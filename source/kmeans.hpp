// Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
// LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE ACADEMIC FREE LICENSE (AFL) v3.0.
//
// Modified by Omar Oubari <omar@oubari.me>
//
// Information: Contains the Kmeans_core class which implements the kmeans algorithm and the Kmeans class used to build the kmeans model
// training model 0

#include <chrono>
#include <cmath>
#include <string>
#include <random>
#include <iostream>

#include "threads.hpp"
#include "utilities.hpp"
#include "data_parser.hpp"

// sampling
#include "sampling_algorithms/d2.hpp"

// external dependency
#include "blaze/Blaze.h"
#include "tbb/parallel_for.h"

template <typename T>
class Kmeans_core {
public:
    // ----- CONSTRUCTOR -----
    Kmeans_core(const dataset<T>& _set, blaze::DynamicMatrix<T, blaze::rowMajor>& _mean, int _M, int nthreads, int _verbose) :
    set(_set),
    mean(_mean),
    old_mean(_mean.rows(), _mean.columns()),
    M(_M),
    threads(nthreads),
    variance(1.0),
    old_variance(1.0),
    sc(set.shape.first),
    dist_evals(0),
    mt(nthreads),
    verbose(_verbose) {

        // error handling
        if (set.shape.first != set.weight.size()) {
             throw std::invalid_argument("data.rows() != weight.size()");
        }
        if (set.shape.second != mean.columns()) {
             throw std::invalid_argument("data.columns() != mean.columns()");
        }
        if (!(variance > 0)) {
            throw std::invalid_argument("invalid variance");
        }
        if (nthreads == 0) {
            throw std::invalid_argument("invalid number of threads");
        }

        if (M == 0) {
            throw std::invalid_argument("number of cluster centers is zero");
        }

        std::random_device r;
        for (int t = 0; t < this->threads.size(); t++) {
            std::seed_seq seq{r(), r(), r(), r(), r(), r(), r(), r()};
            this->mt[t].seed(seq);
            this->mt[t].discard(1e3);
        }
    }

    ~Kmeans_core() {}

    // ----- PUBLIC METHODS -----
    T stream_em() {
        T bound = stream_kmeans_partition();
        old_mean = mean;
        stream_kmeans_mean();
        old_variance = variance;
        stream_kmeans_variance();
        return bound / blaze::sum(set.weight);
    }

    T em() {
        T bound = kmeans_partition();
        old_mean = mean;
        kmeans_mean();
        old_variance = variance;
        kmeans_variance();
        return bound / blaze::sum(set.weight);
    }

    std::vector<std::mt19937>& get_mt() {
        return mt;
    }

    Tp& get_threads() {
        return threads;
    }

    T get_variance() const {
        return variance;
    }

    int64_t get_dist_evals() const {
        return dist_evals;
    }

    T get_mean_change() {
        return blaze::norm(mean - old_mean) / blaze::norm(old_mean);
    }

    T get_var_change() {
        return std::abs( (variance - old_variance) / variance);
    }

protected:

    // ----- PROTECTED METHODS -----
    // stream version
    T stream_kmeans_partition() {
        int N = set.shape.first;
        int D = set.shape.second;

        T sum  = 0;
        T nsum = 0;

        int shift = 0;
        blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
        for (auto& f: set.files) {
            // read npy file
            loadBlazeFromNumpy<T>(f, tmp_data);

            int temp_N = static_cast<int>(tmp_data.rows());
            for (int i = 0; i < temp_N; ++i) {
                int n = i+shift;
                T opt = std::numeric_limits<T>::max();
                int idx = std::numeric_limits<int>::max();
                for (int c = 0; c < M; c++) {
                    T tmp = blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, c));
                    if (opt > tmp) {
                        opt = tmp;
                        idx = c;
                    }
                }
                sc[n] = idx;
                sum += set.weight[n] * opt / (2.0 * variance);
                nsum += set.weight[n];
            }

            // shift by the last number of data points
            shift += temp_N;

            // clear temporary containers
            blaze::clear(tmp_data);
        }

        dist_evals += N * M; // O(NMD)
        return - (nsum * (std::log(M) + (D / 2.0) * std::log(2.0 * M_PI * variance)) + sum);
    }

    // normal version
    T kmeans_partition() {
        int N = set.shape.first;
        int D = set.shape.second;

        T sum  = 0;
        T nsum = 0;

        for (int n = 0; n < N; n++) {
            T opt = std::numeric_limits<T>::max();
            int idx = std::numeric_limits<int>::max();
            for (int c = 0; c < M; c++) {
                T tmp = blaze::sqrNorm(blaze::row(set.data, n) - blaze::row(mean, c));
                if (opt > tmp) {
                    opt = tmp;
                    idx = c;
                }
            }
            sc[n] = idx;
            sum += set.weight[n] * opt / (2.0 * variance);
            nsum += set.weight[n];
        }
        dist_evals += N * M; // O(NMD)
        return - (nsum * (std::log(M) + (D / 2.0) * std::log(2.0 * M_PI * variance)) + sum);
    }

    // stream version
    void stream_kmeans_mean() {
        blaze::DynamicVector<T,  blaze::rowVector> sum(M, 0.0);
        mean = 0;

        int shift = 0;
        blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
        for (auto& f: set.files) {
            // read npy file
            loadBlazeFromNumpy<T>(f, tmp_data);

            int temp_N = static_cast<int>(tmp_data.rows());
            for (int i = 0; i < temp_N; ++i) {
                int n = i+shift;
                blaze::row(mean, sc[n]) += set.weight[n] * blaze::row(tmp_data, i);
                sum[sc[n]] += set.weight[n];
            }

            // shift by the last number of data points
            shift += temp_N;

            // clear temporary containers
            blaze::clear(tmp_data);
        }
        for (int c = 0; c < M; c++) {
            if (sum[c] > 0) {
                blaze::row(mean, c) /= sum[c];
            }
        }
    }

    // normal version
    void kmeans_mean() {
        int N = set.shape.first;

        blaze::DynamicVector<T,  blaze::rowVector> sum(M, 0.0);
        mean = 0;

        for (int n = 0; n < N; n++) {
            blaze::row(mean, sc[n]) += set.weight[n] * blaze::row(set.data, n);
            sum[sc[n]] += set.weight[n];
        }
        for (int c = 0; c < M; c++) {
            if (sum[c] > 0) {
                blaze::row(mean, c) /= sum[c];
            }
        }
    }

    // stream version
    void stream_kmeans_variance() {
        int D = set.shape.second;

        variance = 0;

        T sum = 0;
        int shift = 0;
        blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
        for (auto& f: set.files) {
            // read npy file
            loadBlazeFromNumpy<T>(f, tmp_data);

            int temp_N = static_cast<int>(tmp_data.rows());
            for (int i = 0; i < temp_N; ++i) {
                int n = i+shift;
                variance += set.weight[n] * blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, sc[n]));
                sum += set.weight[n];
            }

            // shift by the last number of data points
            shift += temp_N;

            // clear temporary containers
            blaze::clear(tmp_data);
        }
        variance /= sum * D;
    }

    // normal version
    void kmeans_variance() {

        int N = set.shape.first;
        int D = set.shape.second;

        variance = 0;

        T sum = 0;
        for (int n = 0; n < N; n++) {
            variance += set.weight[n] * blaze::sqrNorm(blaze::row(set.data, n) - blaze::row(mean, sc[n]));
            sum += set.weight[n];
        }
        variance /= sum * D;
    }

    // ----- PROTECTED VARIABLES -----
    const dataset<T>&                                  set;
    blaze::DynamicMatrix<T, blaze::rowMajor>&          mean;
    blaze::DynamicMatrix<T, blaze::rowMajor>           old_mean;
    blaze::DynamicVector<int>                          sc;
    T                                                  variance;
    T                                                  old_variance;
    Tp                                                 threads;
    int                                                M;
    int64_t                                            dist_evals;
    std::vector<std::mt19937>                          mt;
    int                                                verbose;
};

template <typename T>
class Kmeans {

public:

    // ----- CONSTRUCTOR -----
    Kmeans(const dataset<T>& _set, int _M,  int nthreads, int _verbose) :
    set(_set),
    mean(_M, set.shape.second),
    verbose(_verbose),
    alt_criterion(0),
    criterion_scaling(1.0),
    iterations(0),
    dist_evals(0),
    seeding_time(0),
     em_time(0) {
        algo = std::make_unique<Kmeans_core<T>>(set, mean, _M, nthreads, verbose);
        auto start = std::chrono::high_resolution_clock::now();
        if (!blaze::isEmpty(set.data)) {
            d2(set, mean, algo->get_mt(), _M, algo->get_threads()); // D2-sampling
        } else if (blaze::isEmpty(set.data) && !set.files.empty()) {
            stream_d2(set, mean, algo->get_mt(), _M, algo->get_threads()); // D2-sampling in streaming
        }
        auto end = std::chrono::high_resolution_clock::now();
        seeding_time = (end - start) + set.runtime;
    }

    // ----- PUBLIC METHODS -----
    void fit(T eps, int save_additional_info, std::string save_path) {
        T bound = 0;
        T previous;

        // k-means can be described as a variational EM approximation of Gaussian
        // mixture models (with equal mixing proportions and a single variance
        // parameter). With the inclusion of a variance parameter, we can compute
        // a lower bound to the likelihood (as for Gaussian mixture models) and
        // declare convergence based on the relative change of this lower bound.
        // For a fixed number of iterations, the variance does not affect the
        // final cluster centers

        auto start = std::chrono::high_resolution_clock::now();

        while (true) {

            if (save_additional_info > 0) {
                save_centers(save_path, iterations, mean);
                T quantization = error(set);
                save_quantization(save_path, quantization, true, iterations);
            }

            std::vector<T> output = {static_cast<T>(iterations), bound, std::sqrt(algo->get_variance())};
            for (auto& it: output) {
                parameter_save.emplace_back(it);
            }

            if (!blaze::isEmpty(set.data)) {
                bound = algo->em();
            } else if (blaze::isEmpty(set.data) && !set.files.empty()) {
                bound = algo->stream_em();
            }

            ++iterations;
//            if (iterations == 2) {
//                criterion_scaling = std::abs((bound - previous));
//            }

            T sigma = std::sqrt(algo->get_variance());
            if (verbose > 0) {
                std::cout << "iteration " << iterations << " ";
                std::cout << "free energy " << bound << " ";
                std::cout << "sigma " << sigma << std::endl;
            }

            if (eps < 1) {
                if (iterations > 1) {
                    if (alt_criterion == 0) {
                        // free energy criterion
                        if (std::abs((bound - previous) / bound) < eps * criterion_scaling) {
                            break;
                        }
                    } else if (alt_criterion == 1) {
                        if (algo->get_mean_change() < eps) {
                            break;
                        }
                    } else {
                        throw std::logic_error("unrecognised criterion. 0 for free energy change, 1 for mean shift");
                    }
                }
            } else {
                // epochs
                if (iterations >= eps) {
                    break;
                }
            }
            previous = bound;
        }
        auto end = std::chrono::high_resolution_clock::now();
        em_time = end - start;
        dist_evals += algo->get_dist_evals();

        // save final parameters
        std::vector<T> output = {static_cast<T>(iterations), bound, std::sqrt(algo->get_variance())};
        for (auto& it: output) {
            parameter_save.emplace_back(it);
        }

        // save last center and error
        save_centers(save_path, iterations, mean);
        T quantization = error(set);
        save_quantization(save_path, quantization, true, iterations);

        // saving distance evaluations
        save_distance_evaluations(save_path, dist_evals, static_cast<double>(set.shape.second), verbose);

        // saving parameters
        save_parameters(save_path, iterations, parameter_save);

        // saving seeding and em runtime
        save_runtime(save_path, seeding_time, em_time, verbose);
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

    // predicts the clusters of the given data
    blaze::DynamicMatrix<int,blaze::rowMajor> predict(std::string save_path, const dataset<T>& _set) {
        blaze::DynamicMatrix<int,blaze::rowMajor> labels;

        // checking if there are features
        if (!blaze::isEmpty(_set.data) && !_set.coreset) {
            int N = _set.shape.first;
            int M = static_cast<int>(mean.rows());

            // assuming there is a dataset check if it's associated with labels
            if (!_set.labels.empty()) {

                labels.resize(_set.labels.size(), 2);
                algo->get_threads().parallel(N, [&] (int n, int t) {
                    T d2 = std::numeric_limits<T>::max();
                    int best_m = std::numeric_limits<int>::max();

                    // find the cluster with the lowest distance
                    for (int m = 0; m < M; ++m) {
                        T tmp_d2 = blaze::sqrNorm(blaze::row(_set.data, n) - blaze::row(mean, m));
                        if (tmp_d2 < d2) {
                            d2 = tmp_d2;
                            best_m = m;
                        }
                    }

                    labels(n,0) = _set.labels[n];
                    labels(n,1) = best_m;
                });

                // save labels
                save_labels(save_path, "labels", _set, labels);
            }

            // saving cells
            save_cells(save_path, _set);

        // if no features check that there's data to stream
        } else if ((!blaze::isEmpty(_set.data) && _set.coreset) || (blaze::isEmpty(_set.data) && !_set.files.empty())) {
            int M = static_cast<int>(mean.rows());

            // assuming there is a dataset check if it's associated with labels
            if (!_set.labels.empty()) {
                labels.resize(_set.labels.size(), 2);

                int shift = 0;
                blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
                for (auto file: _set.files) {
                    // read npy file
                    loadBlazeFromNumpy<T>(file, tmp_data);

                    int temp_N = static_cast<int>(tmp_data.rows());
                    algo->get_threads().parallel(temp_N, [&] (int i, int t) {
                        T d2 = std::numeric_limits<T>::max();
                        int best_m = std::numeric_limits<int>::max();

                        // find the cluster with the lowest distance
                        for (int m = 0; m < M; ++m) {
                            T tmp_d2 = blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, m));
                            if (tmp_d2 < d2) {
                                d2 = tmp_d2;
                                best_m = m;
                            }
                        }

                        int n = i+shift;
                        labels(n,0) = _set.labels[n];
                        labels(n,1) = best_m;
                    });

                    // shift by the last number of data points
                    shift += temp_N;

                    // clear temporary containers
                    blaze::clear(tmp_data);
                }
            }

            // save labels
            save_labels(save_path, "labels", _set, labels);

            // saving cells
            save_cells(save_path, _set);
        }

        return labels;
    }

protected:

    // ----- PROTECTED VARIABLES -----
    const dataset<T>&                               set;              // dataset
    blaze::DynamicMatrix<T, blaze::rowMajor>        mean;             // cluster center
    int                                             verbose;          // print extra information
    std::vector<T>                                  parameter_save;   // vector containing model parameters at every iteration
    int                                             iterations;       // EM iteration number
    int64_t                                         dist_evals;       // distance evaluations
    std::chrono::duration<double>                   seeding_time;     // coreset and seeding time
    std::chrono::duration<double>                   em_time;          // total EM time for all iterations
    std::unique_ptr<Kmeans_core<T>>                 algo;             // kmeans algorithm
    T                                               criterion_scaling;     // scale stopping criterion depending on the first 2 iterations
    int                                             alt_criterion; // 0 for free energy change, 1 for mean change
};
