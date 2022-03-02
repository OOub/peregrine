// Created by Omar Oubari <omar.oubari@inserm.fr> and Georgios Exarchakis <georgios.exarchakis@ens.fr>
//
// Information: Contains the Gmm_core base class for all the gmm algorithms and the Gmm class used to build the gmm model

#pragma once

#define _USE_MATH_DEFINES

#include <chrono>
#include <string>

#include "threads.hpp"
#include "data_types.hpp"
#include "data_parser.hpp"
#include "utilities.hpp"

// sampling
#include "sampling_algorithms/afkmc2.hpp"

// gmm algorithms
#include "gmm_algorithms/s_gmm.hpp" // training model 1
#include "gmm_algorithms/s_gmm_prior.hpp" // training model 2
#include "gmm_algorithms/d_gmm.hpp" // training model 3
#include "gmm_algorithms/d_gmm_tied.hpp" // training model 4

// external dependency
#include "third_party/numpy.hpp"
#include "blaze/Blaze.h"
#include "tbb/parallel_for.h"

template <typename T>
class Gmm_core {
    
public:
    
    // ----- CONSTRUCTORS and DESTRUCTORS -----
    Gmm_core(const dataset<T>& _set, blaze::DynamicMatrix<T, blaze::rowMajor>& _mean, int _H, int nthreads, T var, int _R, int _verbose) :
    data(_set.data),
    weight(_set.weight),
    mean(_mean),
    R(_R),             // number of Neighboring Clusters Considered
    N(static_cast<int>(data.rows())),    // number of Datapoints
    M(static_cast<int>(mean.rows())),    // number of Clusters
    D(static_cast<int>(mean.columns())), // number of Input dimensions
    H(_H),             // number of Truncated Clusters
    mt(nthreads),      // array of seeds (one for each thread)
    threads(nthreads), // the threads
    variance(var),     // model variance
    free_energy(0),
    dist_evals(0),
    mean_change(0),
    verbose(_verbose),
    covariance(),
    alpha({}) {
        
        // error handling
        if (_set.shape.first != weight.size()) {
             throw std::invalid_argument("data.rows() != weight.size()");
        }
        if (_set.shape.second != mean.columns()) {
             throw std::invalid_argument("data.columns() != mean.columns()");
        }
        if (!(variance > 0)) {
            throw std::invalid_argument("invalid variance");
        }
        if (nthreads == 0) {
            throw std::invalid_argument("invalid number of threads");
        }
        if (N == 0) {
            throw std::invalid_argument("number of data points is zero");
        }
        if (M == 0) {
            throw std::invalid_argument("number of cluster centers is zero");
        }
        if (D == 0) {
            throw std::invalid_argument("number of features is zero");
        }
        if ((H == 0) || (H > M)) {
            throw std::invalid_argument("invalid parameter H");
        }
    }

    virtual ~Gmm_core() {}
    
    // ----- PUBLIC METHODS -----
    virtual void em(int iteration, int criterion, T threshold, T change) {}
    
    // ----- SETTERS AND GETTERS -----
    Tp& get_threads() {
        return threads;
    }
    
    std::vector<std::mt19937_64>& get_mt() {
        return mt;
    }
    
    int64_t get_dist_evals() const {
        return dist_evals;
    }
    
    T get_free_energy() const {
        return free_energy;
    }
    
    void set_variance(T var) {
        variance = var;
    }
    
    T get_variance() const {
        return variance;
    }
    
    blaze::DynamicMatrix<T, blaze::rowMajor>& get_covariance() {
        return covariance;
    }
    
    blaze::DynamicVector<T, blaze::rowVector>& get_alpha() {
        return alpha;
    }
    
    T get_mean_change() const {
        return mean_change;
    }
    
protected:
    
    // ----- PROTECTED VARIABLES -----
    const blaze::DynamicMatrix<T, blaze::rowMajor>&  data;
    const blaze::DynamicVector<T, blaze::rowVector>& weight;
    blaze::DynamicMatrix<T, blaze::rowMajor>&        mean;
    int                                              N;
    int                                              M;
    int                                              D;
    int                                              H;
    int                                              R;
    std::vector<std::mt19937_64>                     mt;
    Tp                                               threads;
    T                                                variance;
    T                                                free_energy;
    T                                                mean_change;
    int64_t                                          dist_evals;
    int                                              verbose;
    blaze::DynamicVector<T, blaze::rowVector>        alpha;
    blaze::DynamicMatrix<T, blaze::rowMajor>         covariance;
};

template <typename T>
class Gmm {
    
public:
    
    // ----- CONSTRUCTORS -----
    Gmm(const dataset<T>& _set, int M, int chain, int H, int R, int nthreads, int _training_model, int _verbose) :
    set(_set),
    mean(M, set.shape.second),
    training_model(_training_model),
    verbose(_verbose),
    alt_criterion(0),
    criterion_scaling(1.0),
    iterations(0),
    dist_evals(0),
    seeding_time(0),
    em_time(0) {
        
        data_size = set.shape.first;
        if (training_model == 1) {
            // S-GMM
            algo = std::make_unique<S_gmm<T>>(set, mean, H, nthreads, 1.0, R, verbose);
        } else if (training_model == 2) {
            // S-GMM-Prior
            algo = std::make_unique<S_gmm_prior<T>>(set, mean, H, nthreads, 1.0, R, verbose);
        } else if (training_model == 3) {
            // D-GMM
            algo = std::make_unique<D_gmm<T>>(set, mean, H, nthreads, 1.0, R, verbose);
        } else if (training_model == 4) {
            // D-GMM with tied covariances
            algo = std::make_unique<D_gmm_tied<T>>(set, mean, H, nthreads, 1.0, R, verbose);
        } else {
             throw std::logic_error("the training model does not exist: 1 = S-GMM | 2 = S-GMM-Prior | 3 = D-GMM | 4 = D-GMM-Tied");
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        if (!blaze::isEmpty(set.data)) {
            afkmc2<T>(set, mean, algo->get_mt(), M, chain, algo->get_threads()); // AFK-MC^2 seeding
        } else if (blaze::isEmpty(set.data) && !set.files.empty()) {
            stream_afkmc2<T>(set, mean, algo->get_mt(), M, chain, algo->get_threads()); // AFK-MC^2 seeding in streaming
        }
        auto end = std::chrono::high_resolution_clock::now();
        seeding_time = (end - start) + set.runtime;
        
        if (training_model == 4) {
            algo->get_covariance() = blaze::IdentityMatrix<T>(set.shape.second) * std::pow(blaze::stddev(set.data),2);
        } else {
            algo->set_variance(std::pow(blaze::stddev(set.data),2));
        }
    }
    
    // ----- PUBLIC METHODS -----
    void fit(T eps, int save_additional_info, std::string save_path) {
        T prv = static_cast<T>(0.0);
        T cur = static_cast<T>(0.0);
        T change = static_cast<T>(1.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        while (true) {
            
            // saving information to numpy arrays
            if (save_additional_info == 1) {
                save_centers(save_path, iterations, mean);
                T quantization = error(set);
                save_quantization<T>(save_path, quantization, true, iterations);
                save_priors(save_path, iterations, algo->get_alpha());
            } else if (save_additional_info == 2) {
                save_priors(save_path, iterations, algo->get_alpha());
            }
            
            std::vector<T> output = {static_cast<T>(iterations), cur, std::sqrt(algo->get_variance())};
            for (auto& it: output) {
                parameter_save.emplace_back(it);
            }
            
            algo->em(iterations, alt_criterion, eps, change);
            ++iterations;
            
            cur = algo->get_free_energy();
//            if (iterations == 2) {
//                criterion_scaling = std::abs((cur - prv));
//            }
                
            if (verbose > 0) {
                std::cout << "iteration " << iterations << " ";
                std::cout << "free energy " << cur << " ";
                std::cout << "\u0394free energy " << change << " ";
                if (!blaze::isEmpty(algo->get_covariance())) {
                    std::cout << "det " << blaze::det(algo->get_covariance()) << std::endl;
                } else {
                    std::cout << "sigma " << std::sqrt(algo->get_variance()) << std::endl;
                }
            }
            
            change = std::abs((cur - prv) / cur);
            if (eps < 1) {
                if (iterations > 1) {
                    if (alt_criterion == 0) {
                        // free energy convergence criterion
                        if (change < eps * criterion_scaling) {
                            break;
                        }
                    } else if (alt_criterion == 1) {
                        // mean shift convergence criterion
                        if (algo->get_mean_change() < eps) {
                            break;
                        }
                    }
                }
            } else {
                // epochs
                if (iterations >= eps) {
                    break;
                }
            }
            prv = cur;
        }
        auto end = std::chrono::high_resolution_clock::now();
        em_time = end - start;
        dist_evals += algo->get_dist_evals();
        
        // save final parameters
        std::vector<T> output = {static_cast<T>(iterations), cur, std::sqrt(algo->get_variance())};
        for (auto& it: output) {
            parameter_save.emplace_back(it);
        }
        
        // save last centers, error and priors
        save_centers(save_path, iterations, mean);
        T quantization = error(set);
        save_quantization<T>(save_path, quantization, true, iterations);
        save_priors(save_path, iterations, algo->get_alpha());
        
        // saving distance evaluations
        save_distance_evaluations(save_path, dist_evals, static_cast<int64_t>(set.shape.second), verbose);

        // saving parameters
        save_parameters(save_path, iterations, parameter_save);

        // saving seeding and em runtime
        save_runtime(save_path, seeding_time, em_time, verbose);

        // saving covariance if available
        save_covariance<T>(save_path, algo->get_covariance());
    }
    
    // computes the quantization error
    T error(const dataset<T>& _set) {
        // checking if there are features
        if (!blaze::isEmpty(_set.data)) {
            int N = static_cast<int>(_set.data.rows());
            int M = static_cast<int>(mean.rows());

            std::vector<T> sum(algo->get_threads().size());
            algo->get_threads().parallel(N, [&] (size_t n, size_t t) {
                T d2 = std::numeric_limits<T>::max();
                for (size_t m = 0; m < M; ++m) {
                    d2 = std::min(d2, blaze::sqrNorm(blaze::row(_set.data, n) - blaze::row(mean, m)));
                }
                sum[t] += d2;
            });

            T result = static_cast<T>(0.0);
            for (auto& it : sum) {
                result += it;
            }
            return result;
        // if no features check that there's data to stream
        } else if (blaze::isEmpty(set.data) && !set.files.empty()) {
            int M = static_cast<int>(mean.rows());

            std::vector<T> sum(algo->get_threads().size());

            int shift = 0;
            blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
            for (auto& f: set.files) {
                // read npy file
                loadBlazeFromNumpy<T>(f, tmp_data);

                int temp_N = static_cast<int>(tmp_data.rows());
                algo->get_threads().parallel(temp_N, [&] (int i, int t) {
                    T d2 = std::numeric_limits<T>::max();
                    for (size_t m = 0; m < M; ++m) {
                        d2 = std::min(d2, blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, m)));
                    }
                    sum[t] += d2;
                });

                // shift by the last number of data points
                shift += temp_N;

                // clear temporary containers
                blaze::clear(tmp_data);
            }

            T result = static_cast<T>(0.0);
            for (auto& it : sum) {
                result += it;
            }
            return result;
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
    
    // predicts the clusters of the given data -- hard clustering
    blaze::DynamicMatrix<int,blaze::rowMajor> predict(std::string save_path, const dataset<T>& _set, int top_k=1) {
        blaze::DynamicMatrix<int,blaze::rowMajor> labels;

        // invert covariance matrix via cholesky if using a tied covariance model
        if (!blaze::isEmpty(algo->get_covariance())) {
            inv_covariance = algo->get_covariance();
            blaze::invert<blaze::byLLH>(inv_covariance);
        }
        
        // checking if there are features
        if (!blaze::isEmpty(_set.data) && !_set.coreset) {
            int N = _set.shape.first;
            int M = static_cast<int>(mean.rows());
            int D = static_cast<int>(mean.columns());
            
            // assuming there is a dataset check if it's associated with labels
            if (!_set.labels.empty()) {

                labels.resize(_set.labels.size(), 1+top_k);
                algo->get_threads().parallel(N, [&] (int n, int t) {
                    std::vector<int> best_m(top_k);
                    if (blaze::isEmpty(algo->get_alpha()) && blaze::isEmpty(algo->get_covariance())) {
                        // condition for algorithms with uniform priors
                        blaze::DynamicVector<T, blaze::rowVector> distances(M);
                        for (int m = 0; m < M; m++) {
                            distances[m] = blaze::sqrNorm(blaze::row(_set.data, n) - blaze::row(mean, m));
                        }
                        
                        // find top k clusters
                        for (auto& bm: best_m) {
                            bm = static_cast<int>(blaze::argmin(distances));
                            distances[bm] = std::numeric_limits<T>::max();
                        }
                        
                    } else if (!blaze::isEmpty(algo->get_alpha()) && blaze::isEmpty(algo->get_covariance())) {
                        // condition for algorithms with prior learning
                        blaze::DynamicVector<T, blaze::rowVector> distances(M);
                        for (int m = 0; m < M; m++) {
                            distances[m] = blaze::sqrNorm(blaze::row(_set.data, n) - blaze::row(mean, m));
                        }
                        distances *= - 0.5 / algo->get_variance();
                        T lim = blaze::max(distances);
                        distances -= lim;
                        distances += - 0.5 * D * blaze::log(2.0 * M_PI * algo->get_variance());
                        distances += blaze::log(algo->get_alpha());
                        
                        // find top k clusters
                        for (auto& bm: best_m) {
                            bm = static_cast<int>(blaze::argmax(distances));
                            distances[bm] = std::numeric_limits<T>::lowest();
                        }
                    } else if (blaze::isEmpty(algo->get_alpha()) && !blaze::isEmpty(algo->get_covariance())) {
                        // condition for algorithms with uniform priors and tied covariances
                        blaze::DynamicVector<T, blaze::rowVector> errors(M);
                        for (int m = 0; m < M; m++) {
                            blaze::DynamicVector<T, blaze::rowVector> error = blaze::row(_set.data, n) - blaze::row(mean, m);
                            errors[m] = error * inv_covariance * blaze::trans(error);
                        }
                        
                        errors *= - 0.5;
                        T lim = blaze::max(errors);
                        errors -= lim;
                        errors += - 0.5 * D * blaze::log(2.0 * M_PI * blaze::det(algo->get_covariance()));
                        errors -= blaze::log(M);
                        
                        // find top k clusters
                        for (auto& bm: best_m) {
                            bm = static_cast<int>(blaze::argmax(errors));
                            errors[bm] = std::numeric_limits<T>::lowest();
                        }
                    }

                    labels(n,0) = _set.labels[n];
                    for (auto i=1; i<top_k+1; ++i) {
                        labels(n,i) = best_m[i-1];
                    }
                });
            
                // save labels
                save_labels(save_path, "labels", _set, labels);
            }

            // saving cells
            save_cells(save_path, _set);

        // if no features check that there's data to stream
        } else if ((!blaze::isEmpty(_set.data) && _set.coreset) || (blaze::isEmpty(_set.data) && !_set.files.empty())) {
            
            int M = static_cast<int>(mean.rows());
            int D = static_cast<int>(mean.columns());
            
            // assuming there is a dataset check if it's associated with labels
            if (!_set.labels.empty()) {
                labels.resize(_set.labels.size(), 1+top_k);

                int shift = 0;
                blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
                for (auto file: _set.files) {
                    // read npy file
                    loadBlazeFromNumpy<T>(file, tmp_data);

                    int temp_N = static_cast<int>(tmp_data.rows());
                    algo->get_threads().parallel(temp_N, [&] (int i, int t) {
                        std::vector<int> best_m(top_k);
                        if (blaze::isEmpty(algo->get_alpha()) && blaze::isEmpty(algo->get_covariance())) {
                            // condition for algorithms with uniform priors
                            blaze::DynamicVector<T, blaze::rowVector> distances(M);
                            for (int m = 0; m < M; m++) {
                                distances[m] = blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, m));
                            }
                            
                            // find top k clusters
                            for (auto& bm: best_m) {
                                bm = static_cast<int>(blaze::argmin(distances));
                                distances[bm] = std::numeric_limits<T>::max();
                            }
                        } else if (!blaze::isEmpty(algo->get_alpha()) && blaze::isEmpty(algo->get_covariance())) {
                            // condition for algorithms with prior learning
                            blaze::DynamicVector<T, blaze::rowVector> distances(M);
                            for (int m = 0; m < M; ++m) {
                                distances[m]  = blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, m));
                            }
                            distances *= - 0.5 / algo->get_variance();
                            T lim = blaze::max(distances);
                            distances -= lim;
                            distances += - 0.5 * D * blaze::log(2.0 * M_PI * algo->get_variance());
                            distances += blaze::log(algo->get_alpha());
                            
                            // find top k clusters
                            for (auto& bm: best_m) {
                                bm = static_cast<int>(blaze::argmax(distances));
                                distances[bm] = std::numeric_limits<T>::lowest();
                            }
                        } else if (blaze::isEmpty(algo->get_alpha()) && !blaze::isEmpty(algo->get_covariance())) {
                            // condition for algorithms with uniform priors and tied covariances
                            blaze::DynamicVector<T, blaze::rowVector> errors(M);
                            for (int m = 0; m < M; m++) {
                                blaze::DynamicVector<T, blaze::rowVector> error = blaze::row(tmp_data, i) - blaze::row(mean, m);
                                errors[m] = error * inv_covariance * blaze::trans(error);
                            }
                            
                            errors *= - 0.5;
                            T lim = blaze::max(errors);
                            errors -= lim;
                            errors += - 0.5 * D * blaze::log(2.0 * M_PI * blaze::det(algo->get_covariance()));
                            errors -= blaze::log(M);
                            
                            // find top k clusters
                            for (auto& bm: best_m) {
                                bm = static_cast<int>(blaze::argmax(errors));
                                errors[bm] = std::numeric_limits<T>::lowest();
                            }
                        }

                        int n = i+shift;
                        labels(n,0) = _set.labels[n];
                        for (auto i=1; i<top_k+1; ++i) {
                            labels(n,i) = best_m[i-1];
                        }
                    });

                    // shift by the last number of data points
                    shift += temp_N;

                    // clear temporary containers
                    blaze::clear(tmp_data);
                }
                
                // save labels
                save_labels(save_path, "labels", _set, labels);
            }

            // saving cells
            save_cells(save_path, _set);
        }

        return labels;
    }
    
    // predicts the clusters of the given data -- soft clustering
    blaze::DynamicMatrix<int,blaze::rowMajor> soft_predict(std::string save_path, const dataset<T>& _set) {
        blaze::DynamicMatrix<T,blaze::rowMajor> labels;
        
        // invert covariance matrix via cholesky if using a tied covariance model
        if (!blaze::isEmpty(algo->get_covariance())) {
            inv_covariance = algo->get_covariance();
            blaze::invert<blaze::byLLH>(inv_covariance);
        }
        
        if (!blaze::isEmpty(_set.data) && !_set.coreset) {
            int N = _set.shape.first;
            int M = static_cast<int>(mean.rows());
            int D = static_cast<int>(mean.columns());
            
            // assuming there is a dataset check if it's associated with labels
            if (!_set.labels.empty()) {
                
                labels.resize(_set.labels.size(), M+1);
                algo->get_threads().parallel(N, [&] (int n, int t) {
                    auto label_row = blaze::subvector(blaze::row(labels, n), 1, M);
                    if (blaze::isEmpty(algo->get_alpha()) && blaze::isEmpty(algo->get_covariance())) {
                        // condition for algorithms with uniform priors
                        for (int m = 0; m < M; m++) {
                            label_row[m] = blaze::sqrNorm(blaze::row(_set.data, n) - blaze::row(mean, m));
                        }
                        
                        label_row *= - 0.5 / algo->get_variance();
                        T lim = blaze::max(label_row);
                        label_row -= lim;
                        label_row += - 0.5 * D * blaze::log(2.0 * M_PI * algo->get_variance());
                        label_row -= blaze::log(M);
                        
                    } else if (!blaze::isEmpty(algo->get_alpha()) && blaze::isEmpty(algo->get_covariance())) {
                        // condition for algorithms with prior learning
                        for (int m = 0; m < M; m++) {
                            label_row[m] = blaze::sqrNorm(blaze::row(_set.data, n) - blaze::row(mean, m));
                        }
                        
                        label_row *= - 0.5 / algo->get_variance();
                        T lim = blaze::max(label_row);
                        label_row -= lim;
                        label_row += - 0.5 * D * blaze::log(2.0 * M_PI * algo->get_variance());
                        label_row += blaze::log(algo->get_alpha());
                        
                    } else if (blaze::isEmpty(algo->get_alpha()) && !blaze::isEmpty(algo->get_covariance())) {
                        // condition for algorithms with uniform priors and tied covariances
                        for (int m = 0; m < M; m++) {
                            blaze::DynamicVector<T, blaze::rowVector> error = blaze::row(_set.data, n) - blaze::row(mean, m);
                            label_row[m] = error * inv_covariance * blaze::trans(error);
                        }
                        
                        label_row *= - 0.5;
                        T lim = blaze::max(label_row);
                        label_row -= lim;
                        label_row += - 0.5 * D * blaze::log(2.0 * M_PI * blaze::det(algo->get_covariance()));
                        label_row -= blaze::log(M);
                    }
                    labels(n,0) = static_cast<T>(_set.labels[n]);
                });
                
                // save labels
                save_labels(save_path, "labels_soft", _set, labels);
            }
            
            // saving cells
            save_cells(save_path, _set);
            
        } else if ((!blaze::isEmpty(_set.data) && _set.coreset) || (blaze::isEmpty(_set.data) && !_set.files.empty())) {
            int M = static_cast<int>(mean.rows());
            int D = static_cast<int>(mean.columns());
            
            // assuming there is a dataset check if it's associated with labels
            if (!_set.labels.empty()) {
                
                labels.resize(_set.labels.size(), M+1);

                int shift = 0;
                blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
                for (auto file: _set.files) {
                    // read npy file
                    loadBlazeFromNumpy<T>(file, tmp_data);

                    int temp_N = static_cast<int>(tmp_data.rows());
                    algo->get_threads().parallel(temp_N, [&] (int i, int t) {
                        int n = i+shift;
                        auto label_row = blaze::subvector(blaze::row(labels, n), 1, M);
                        if (blaze::isEmpty(algo->get_alpha()) && blaze::isEmpty(algo->get_covariance())) {
                            // condition for algorithms with uniform priors
                            for (int m = 0; m < M; m++) {
                                label_row[m] = blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, m));
                            }
                            
                            label_row *= - 0.5 / algo->get_variance();
                            T lim = blaze::max(label_row);
                            label_row -= lim;
                            label_row += - 0.5 * D * blaze::log(2.0 * M_PI * algo->get_variance());
                            label_row -= blaze::log(M);
                            
                        } else if (!blaze::isEmpty(algo->get_alpha()) && blaze::isEmpty(algo->get_covariance())) {
                            // condition for algorithms with prior learning
                            for (int m = 0; m < M; m++) {
                                label_row[m] = blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, m));
                            }
                            
                            label_row *= - 0.5 / algo->get_variance();
                            T lim = blaze::max(label_row);
                            label_row -= lim;
                            label_row += - 0.5 * D * blaze::log(2.0 * M_PI * algo->get_variance());
                            label_row += blaze::log(algo->get_alpha());
                        } else if (blaze::isEmpty(algo->get_alpha()) && !blaze::isEmpty(algo->get_covariance())) {
                            // condition for algorithms with uniform priors and tied covariances
                            for (int m = 0; m < M; m++) {
                                blaze::DynamicVector<T, blaze::rowVector> error = blaze::row(tmp_data, i) - blaze::row(mean, m);
                                label_row[m] = error * inv_covariance * blaze::trans(error);
                            }
                            
                            label_row *= - 0.5;
                            T lim = blaze::max(label_row);
                            label_row -= lim;
                            label_row += - 0.5 * D * blaze::log(2.0 * M_PI * blaze::det(algo->get_covariance()));
                            label_row -= blaze::log(M);
                        }
                        labels(n,0) = static_cast<T>(_set.labels[n]);
                    });
                    
                    // shift by the last number of data points
                    shift += temp_N;

                    // clear temporary containers
                    blaze::clear(tmp_data);
                }
                
                // save labels
                save_labels(save_path, "labels_soft", _set, labels);
            }
            
            // saving cells
            save_cells(save_path, _set);
        }
        
        return labels;
    }
    
protected:
    
    // ----- PROTECTED VARIABLES -----
    const dataset<T>&                               set;                   // dataset
    blaze::DynamicMatrix<T, blaze::rowMajor>        inv_covariance;
    blaze::DynamicMatrix<T, blaze::rowMajor>        mean;                  // cluster center
    std::vector<int>                                indices;               // vector of indices for minibatching
    int                                             data_size;             // size of dataset or coreset
    int                                             verbose;               // print extra information
    std::vector<T>                                  parameter_save;        // vector containing model parameters at every iteration
    int                                             iterations;            // EM iteration number
    int64_t                                         dist_evals;            // distance evaluations
    std::chrono::duration<double>                   seeding_time;          // coreset and seeding time
    std::chrono::duration<double>                   em_time;               // total EM time for all iterations
    std::unique_ptr<Gmm_core<T>>                    algo;                  // gmm algorithm
    int                                             training_model;
    T                                               criterion_scaling;     // scale stopping criterion depending on the first 2 iterations
    int                                             alt_criterion;         // 0 for free energy change, 1 for mean change
};
