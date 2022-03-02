// Created by Omar Oubari <omar@oubari.me>
//
// Information: Implementation of the S-GMM algorithm with learned prior (S-GMM-Prior)
// training model 2

#pragma once

#define _USE_MATH_DEFINES

#include <cmath>
#include <stdexcept>
#include <random>
#include <vector>
#include <array>
#include <limits>
#include <iostream>

#include "../threads.hpp"
#include "../data_types.hpp"
#include "blaze/Blaze.h"

// forward declaration
template <typename T>
class Gmm_core;

template <typename T>
class S_gmm_prior : public Gmm_core<T> {
    
public:
    
    // ----- CONSTRUCTORS AND DESTRUCTORS -----
    S_gmm_prior(const dataset<T>& _set, blaze::DynamicMatrix<T, blaze::rowMajor>& _mean, int _H, int nthreads, T var, int _R, int _verbose) :
    Gmm_core<T>(_set, _mean, _H, nthreads, var, std::min(_R, static_cast<int>(_mean.rows()) - _H), _verbose),
    Qn(this->N, this->H),
    Dc(this->N, this->H+this->R),
    eta(1e-10),
    uniform_toggle(true) {
            
        std::random_device r;
        for (int t = 0; t < this->threads.size(); t++) {
            std::seed_seq seq{r(), r(), r(), r(), r(), r(), r(), r()};
            this->mt[t].seed(seq);
            this->mt[t].discard(1e3);
        }

        // initialising priors
        this->alpha = blaze::DynamicVector<T,  blaze::rowVector>(this->M, 1.0/this->M);
        
        // initialise uniform distributions
        uniform_int_dist = std::uniform_int_distribution<int>(std::uniform_int_distribution<int>(0, this->M - 1));
        
        // initialisation
        init_Dc();
        
        mean_vec = std::vector<blaze::DynamicMatrix<T>>(this->threads.size(), blaze::DynamicMatrix<T>(this->M, this->D, 0.0));
        norm = std::vector<blaze::DynamicVector<T, blaze::rowVector>>(this->threads.size(), blaze::DynamicVector<T,  blaze::rowVector>(this->M, 0.0));
        _variance = std::vector<T>(this->threads.size(), 0.0);
        acc = std::vector<T>(this->threads.size(), 0.0);
    }
    
    virtual ~S_gmm_prior(){}
    
    // ----- PUBLIC METHODS -----
    
    virtual void em(int iteration, int criterion, T threshold, T change) {
        
        if (iteration > 0) {
            this->free_energy = 0;
            for (int t = 0; t < this->threads.size(); t++) {
                blaze::reset(mean_vec[t]);
                blaze::reset(norm[t]);
                _variance[t] = 0.0;
                acc[t] = 0.0;
            }
        }
        
        // E-STEP
        // toggle to switch from uniform sampling to prior distribution
        if (uniform_toggle && change < threshold*10) {
            uniform_toggle = false;
        }
        expectation(iteration);
        
        // M-STEP
        maximisation(criterion);
    }

protected:
    
    // ----- PROTECTED METHODS -----
    
    // initialises Dc randomly
    // Dc is the set of clusters that's going to be considered for each data point
    void init_Dc() {
        // number of threads by number of clusters used to initialise the clusters per datapoint
        auto tbl = std::vector<blaze::DynamicVector<_tab, blaze::rowVector>>(this->threads.size(), blaze::DynamicVector<_tab, blaze::rowVector>(this->M));

        this->threads.parallel(this->N, [&] (int n, int t) {
            // uniform distribution on integers between 0 and M - 1
            for (auto i=0; i<this->H; ++i) {
                int u;
                do {
                    // draw a uniform number for all clusters
                    u = uniform_int_dist(this->mt[t]);

                // keep drawing samples for as long as this cluster has already been assigned to this datapoint
                } while (tbl[t][u].opc == n);

                tbl[t][u].opc = n;
                Dc(n,i).cluster = u;
                Dc(n,i).distance = std::numeric_limits<T>::max();
            }
        });
    }
    
    void expectation(int iteration) {
        
        auto tab = std::vector<blaze::DynamicVector<_tab, blaze::rowVector>>(this->threads.size(), blaze::DynamicVector<_tab,blaze::rowVector>(this->M));
        
        T inv_variance = 1.0/this->variance;
        this->threads.parallel(this->N, [&] (int n, int t) {
            
            if (uniform_toggle) {
                for (auto i=0; i<this->H; ++i) {
                    tab[t][Dc(n,i).cluster].opc = n;
                }
                
                for (auto i=0; i<this->R; ++i) {
                    int u;
                    do {
                        // draw a uniform number for all clusters
                        u = uniform_int_dist(this->mt[t]);

                    // keep drawing samples for as long as this cluster has already been assigned to this datapoint
                    } while (tab[t][u].opc == n);

                    tab[t][u].opc = n;
                    Dc(n,this->H+i).cluster = u;
                }

            } else {
                // set weights of elements already in Dc to 0
                auto _alpha = this->alpha;
                for (auto i=0; i<this->H+this->R; ++i) {
                    _alpha[Dc(n,i).cluster] = 0.0;
                }

                // find non-zero elements
                auto nonzero = std::count_if(_alpha.begin(), _alpha.end(), [&](T w){ return w > eta;});
                
                // make sure we can sample from all clusters not already present in Kn
                if (nonzero < this->R) {
                    for (auto i=0; i<this->R; ++i) {
                        int u;
                        do {
                            // draw a uniform number for all clusters
                            u = uniform_int_dist(this->mt[t]);

                        // keep drawing samples for as long as this cluster has already been assigned to this datapoint
                        } while (tab[t][u].opc == n);

                        tab[t][u].opc = n;
                        Dc(n,this->H+i).cluster = u;
                    }
                } else {
                    // sample from prior distribution
                    std::discrete_distribution<> prior_dist(_alpha.begin(), _alpha.end());
                    for (auto i=0; i<this->R; ++i) {
                        int u;
                        do {
                            // draw a number for all clusters
                            u = prior_dist(this->mt[t]);

                        // keep drawing samples for as long as this cluster has already been assigned to this datapoint
                        } while (tab[t][u].opc == n);

                        tab[t][u].opc = n;
                        Dc(n,this->H+i).cluster = u;
                    }
                }
            }
            
            // calculate distances for Dc
            for (auto& it : blaze::row(Dc, n)) {
                it.distance = blaze::sqrNorm(blaze::row(this->data, n) - blaze::row(this->mean, it.cluster));
            }
            
            // take only H best clusters
            std::nth_element( blaze::row(Dc, n).begin(), blaze::row(Dc, n).begin() + this->H, blaze::row(Dc, n).end(), [&] (auto& lhs, auto& rhs) -> bool { return lhs.distance < rhs.distance; });

            for (auto i=0; i<this->H; i++) {
                Qn(n,i) = Dc(n,i).distance;
            }
            auto qcn = blaze::row(Qn, n);

            qcn *= -0.5 * inv_variance;
            T lim = blaze::max(qcn);
            for (auto i=0; i<this->H; i++) {
                Qn(n,i) = std::exp(Qn(n,i) - lim) * this->alpha[Dc(n,i).cluster];
                if (!std::isfinite(Qn(n,i)) || std::isnan(Qn(n,i))) {
                    Qn(n,i) = 0;
                }
            }
            T sum = blaze::sum(qcn);
            
            if (sum != 0) {
                for (auto i=0; i<this->H; i++) {
                    Qn(n,i) /= sum;
                }
            }
            
            // log likelihood
            acc[t] += this->weight[n] * (-0.5 * this->D * std::log(2.0 * M_PI * this->variance));
            acc[t] += this->weight[n] * (std::log(sum) + lim);
        });
        this->dist_evals += this->N * (this->H + this->R);// O(N(H+R)D) for Dc
    }
    
    void maximisation(int criterion) {

        this->threads.parallel(this->N, [&] (int n, int t) {
            for (int h = 0; h < this->H; ++h) {

                int c = Dc(n,h).cluster;
                T w = Qn(n,h) * this->weight[n];
                
                _variance[t] += w * blaze::sqrNorm(blaze::row(this->mean, c) - blaze::row(this->data, n));
                blaze::row(mean_vec[t], c) += w * blaze::row(this->data, n);
                norm[t][c] += w;
            }
        });
        
        // free energy update
        for (int t = 0; t < this->threads.size(); ++t) {
            this->free_energy += acc[t];
        }
        
        // taking mean of the free energy
        this->free_energy /= blaze::sum(this->weight);
        
        // mean update
        for (int t = 1; t < this->threads.size(); ++t) {
            mean_vec[0] += mean_vec[t];
            norm[0] += norm[t];
            _variance[0] += _variance[t];
        }
        
        T sum = 0.0;
        for (int m = 0; m < this->M; ++m) {
            T tmp = 1.0 / norm[0][m];
            if (std::isfinite(tmp)) {
                blaze::row(mean_vec[0], m) *= tmp;
            } else {
                blaze::row(mean_vec[0], m) = blaze::row(this->mean, m);
            }
            sum += norm[0][m];
        }
        
        if (criterion == 1) {
            this->mean_change = blaze::norm(mean_vec[0] - this->mean) / blaze::norm(this->mean);
        }
        this->mean = mean_vec[0];
        
        // variance update
        this->variance = _variance[0] / (sum * this->D);
        
        // alpha update
        for (int m = 0; m < this->M; ++m) {
            this->alpha[m] = norm[0][m] / sum;
            if (norm[0][m] == 0) {
                this->alpha[m] = std::numeric_limits<T>::min();
            } else {
                this->alpha[m] = norm[0][m] / sum;
            }
        }
    }
    
    // ----- PROTECTED VARIABLES -----
    blaze::DynamicMatrix<T, blaze::rowMajor>                 Qn;
    blaze::DynamicMatrix<_dcn<T>, blaze::rowMajor>           Dc;
    std::uniform_int_distribution<int>                       uniform_int_dist;
    T                                                        eta;
    std::vector<blaze::DynamicVector<T, blaze::rowVector>>   norm;
    std::vector<blaze::DynamicMatrix<T>>                     mean_vec;
    std::vector<T>                                           _variance;
    std::vector<T>                                           acc;
    bool                                                     uniform_toggle;
};
