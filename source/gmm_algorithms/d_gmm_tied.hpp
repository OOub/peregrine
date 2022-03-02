// Created by Omar Oubari <omar@oubari.me>
//
// Information: Implementation of the D-GMM algorithm with tied covariances
// training model 4

#pragma once

#define _USE_MATH_DEFINES

#include <chrono>
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
class D_gmm_tied : public Gmm_core<T> {

public:

    // ----- CONSTRUCTORS AND DESTRUCTORS -----
    D_gmm_tied(const dataset<T>& _set, blaze::DynamicMatrix<T, blaze::rowMajor>& _mean, int _H, int nthreads, T var, int _R, int _verbose) :
    Gmm_core<T>(_set, _mean, _H, nthreads, var, std::min(_R, static_cast<int>(_mean.rows()) - _H), _verbose),
    Qn(this->N, this->H),
    Dc(this->N, this->H+this->R),
    S(this->M, this->M),
    eta(1e-10){
        std::random_device r;
        for (int t = 0; t < this->threads.size(); t++) {
            std::seed_seq seq{r(), r(), r(), r(), r(), r(), r(), r()};
            this->mt[t].seed(seq);
            this->mt[t].discard(1e3);
        }

        // initialise uniform distributions
        uniform_int_dist = std::uniform_int_distribution<int>(std::uniform_int_distribution<int>(0, this->M - 1));
        uniform_real_dist = std::uniform_real_distribution<T>(std::uniform_real_distribution<T>(static_cast<T>(0.0), static_cast<T>(1.0)));

        // initialisation
        init_Dc();
        
        mean_vec = std::vector<blaze::DynamicMatrix<T>>(this->threads.size(), blaze::DynamicMatrix<T>(this->M, this->D, 0.0));
        norm = std::vector<blaze::DynamicVector<T, blaze::rowVector>>(this->threads.size(), blaze::DynamicVector<T,  blaze::rowVector>(this->M, 0.0));
        weights = std::vector<blaze::DynamicVector<T,  blaze::rowVector>>(this->threads.size(), blaze::DynamicVector<T, blaze::rowVector>(this->M, 0.0));
        _covariance = std::vector<blaze::DynamicMatrix<T>>(this->threads.size(), blaze::DynamicMatrix<T>(this->D, this->D, 0.0));
        acc = std::vector<T>(this->threads.size(), 0.0);
    }

    virtual ~D_gmm_tied(){}

    // ----- PUBLIC METHODS -----

    virtual void em(int iteration, int criterion, T threshold, T change) {
        
        if (iteration > 0) {
            this->free_energy = 0;
            for (int t = 0; t < this->threads.size(); t++) {
                blaze::reset(mean_vec[t]);
                blaze::reset(norm[t]);
                blaze::reset(weights[t]);
                blaze::reset(_covariance[t]);
                acc[t] = 0.0;
            }
        }
        
        // E-STEP
        expectation(iteration);

        // reset S matrix
        blaze::reset(S);

        // M-STEP
        maximisation(criterion);
    }

protected:

    // ----- PROTECTED METHODS -----

    // initialises Dc randomly
    // Dc is the set of clusters that's going to be considered for each data point
    void init_Dc() {
        // number of threads by number of clusters used to initialise the clusters per datapoint
        auto tbl = std::vector<blaze::DynamicVector<_tab, blaze::rowVector>>(this->threads.size(),blaze::DynamicVector<_tab, blaze::rowVector>(this->M));

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
        
        // invert the tied covariance matrix via cholesky decomposition
        blaze::DynamicMatrix<T,blaze::rowMajor> inv_covariance = this->covariance;
        blaze::invert<blaze::byLLH>(inv_covariance);
        
        this->threads.parallel(this->N, [&] (int n, int t) {
            if (iteration == 0) {

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

                // find minimum error
                _dcn<T> min_c{
                    std::numeric_limits<int>::max(),
                    std::numeric_limits<T>::max()
                };

                for (int c=0; c<this->H; ++c) {
                    if (min_c.distance > Dc(n,c).distance) {
                        min_c = Dc(n,c);
                    }
                }

                // find weights for proposal distribution
                T denom = 1.0 / blaze::sum(blaze::row(S, min_c.cluster));
                for (auto c=0; c < this->M; ++c) {
                    weights[t][c] = S(min_c.cluster,c) * denom;
                }

                // set weights of elements already in Dc to 0
                for (auto i=0; i<this->H; ++i) {
                    weights[t][Dc(n,i).cluster] = 0.0;
                }

                // find non-zero elements
                auto nonzero = std::count_if(weights[t].begin(), weights[t].end(), [&](T w){ return w >= eta;});

                // make sure we can sample from all clusters not already present in Kn
                if (nonzero < this->R) {
                    weights[t] = blaze::DynamicVector<T,  blaze::rowVector>(this->M, 1.0);
                    for (auto i=0; i<this->H; ++i) {
                        weights[t][Dc(n,i).cluster] = 0.0;
                    }

                    // sum the elements
                    auto p_sum = 1.0 / blaze::sum(weights[t]);

                    // normalise the weights
                    weights[t] *= p_sum;
                }

                // define a discrete distribution according to Kn and S
                tab[t] = blaze::DynamicVector<_tab,blaze::rowVector>(this->M);
                std::discrete_distribution<> proposal_dist(weights[t].begin(), weights[t].end());
                for (auto i=0; i<this->R; ++i) {
                    int u;
                    do {
                        // draw a number for all clusters
                        u = proposal_dist(this->mt[t]);

                    // keep drawing samples for as long as this cluster has already been assigned to this datapoint
                    } while (tab[t][u].opc == n);

                    tab[t][u].opc = n;
                    Dc(n,this->H+i).cluster = u;
                }
            }

            // calculate error for Dc
            for (auto& it : blaze::row(Dc, n)) {
                blaze::DynamicVector<T, blaze::rowVector> error = blaze::row(this->data, n) - blaze::row(this->mean, it.cluster);
                it.distance = error * inv_covariance * blaze::trans(error);
            }
            std::nth_element( blaze::row(Dc, n).begin(), blaze::row(Dc, n).begin() + this->H, blaze::row(Dc, n).end(), [&] (auto& lhs, auto& rhs) -> bool { return lhs.distance < rhs.distance; });

            for (auto i=0; i<this->H; i++) {
                Qn(n,i) = Dc(n,i).distance;
            }
            auto qcn = blaze::row(Qn, n);
            qcn *= -0.5;
            T lim = blaze::max(qcn);
            qcn = blaze::exp(qcn - lim);
            T sum = blaze::sum(qcn);
            qcn /= sum;
            
            acc[t] += this->weight[n] * (-0.5 * this->D * std::log(2.0 * M_PI * det(this->covariance)));
            acc[t] += this->weight[n] * (std::log(sum) + lim);
            acc[t] -= this->weight[n] * (std::log(this->M));
        });
        this->dist_evals += this->N * (this->H + this->R);// O(NH'D) for Kn and O(NKD) for Dc
    }

    void maximisation(int criterion) {

        this->threads.parallel(this->N, [&] (int n, int t) {
            for (int i = 0; i < this->H; ++i) {

                int c = Dc(n,i).cluster;
                T w = Qn(n,i) * this->weight[n];

                blaze::DynamicVector<T, blaze::rowVector> error = blaze::row(this->mean, c) - blaze::row(this->data, n);
                _covariance[t] += w * blaze::trans(error) * error;
                
                blaze::row(mean_vec[t], c) += w * blaze::row(this->data, n);
                norm[t][c] += w;

                for (int j = 0; j < this->H; ++j) {
                    int cp = Dc(n,j).cluster;
                    S(c,cp) +=  - (Dc(n,i).distance + Dc(n,j).distance) / (2 * this->variance);
                }
            }
        });
        S /= this->N;
        
        // free energy update
        for (int t = 0; t < this->threads.size(); t++) {
            this->free_energy += acc[t];
        }

        // taking mean of the free energy
        this->free_energy /= blaze::sum(this->weight);

        for (int t = 1; t < this->threads.size(); ++t) {
            mean_vec[0] += mean_vec[t];
            norm[0] += norm[t];
            _covariance[0] += _covariance[t];
        }
        
        T sum = 0.0;
        for (int h = 0; h < this->M; ++h) {
            T tmp = 1.0 / norm[0][h];
            if (std::isfinite(tmp)) {
                blaze::row(mean_vec[0], h) *= tmp;
            } else {
                blaze::row(mean_vec[0], h) = blaze::row(this->mean, h);
            }
            sum += norm[0][h];
        }

        if (criterion == 1) {
            this->mean_change = blaze::norm(mean_vec[0] - this->mean) / blaze::norm(this->mean);
        }
        this->mean = mean_vec[0];
        
        this->covariance = _covariance[0] / sum;
    }

    // ----- PROTECTED VARIABLES -----
    blaze::DynamicMatrix<T, blaze::rowMajor>                 Qn;
    blaze::DynamicMatrix<_dcn<T>, blaze::rowMajor>           Dc;
    blaze::DynamicMatrix<T, blaze::rowMajor>                 S;
    std::uniform_int_distribution<int>                       uniform_int_dist;
    std::uniform_real_distribution<T>                        uniform_real_dist;
    double                                                   eta;
    std::vector<blaze::DynamicVector<T, blaze::rowVector>>   norm;
    std::vector<blaze::DynamicMatrix<T>>                     mean_vec;
    std::vector<blaze::DynamicMatrix<T>>                     _covariance;
    std::vector<T>                                           acc;
    std::vector<blaze::DynamicVector<T,  blaze::rowVector>>  weights;
};
