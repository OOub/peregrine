// Information: kmeans++ sampling with D^2 weighting

#pragma once

#include <iostream>
#include <vector>
#include <utility>
#include <random>

#include "../data_types.hpp"
#include "../threads.hpp"

// external dependency
#include "blaze/Blaze.h"

// normal version of kmeans ++
// set      - dataset type (check data_types.hpp)
// mean     - cluster centers
// mt       - vector of random number generators
// H        - number of cluster centers
// threads  - threads object for parallelization
template <typename T>
void d2(const dataset<T>& set, blaze::DynamicMatrix<T, blaze::rowMajor>& mean, std::vector<std::mt19937>& mt, int H, Tp& threads) {
    auto& data = set.data;
    auto& weight = set.weight;

    int N = static_cast<int>(data.rows());

    blaze::DynamicVector<T,  blaze::rowVector> min_dst(N);
    blaze::DynamicVector<int> min_idx(N);

    std::discrete_distribution<int> i(weight.begin(), weight.end());
    blaze::row(mean, 0) = blaze::row(data, i(mt[0]));

    threads.parallel(N, [&] (int n, int t) {
        min_dst[n] = weight[n] * blaze::sqrNorm(blaze::row(data, n) - blaze::row(mean, 0));
        min_idx[n] = 0;
    });

    threads.parallel(H, [&] (int c, int t) {
        std::discrete_distribution<int> q(min_dst.begin(), min_dst.end());
        int clu = q(mt[t]);
        blaze::row(mean, c) = blaze::row(data, clu);

        for (int n = 0; n < N; n++) {
            T tmp = weight[n] * blaze::sqrNorm(blaze::row(data, n) - blaze::row(mean, c));

            if (min_dst[n] > tmp) {
                min_dst[n] = tmp;
                min_idx[n] = clu;
            }
        }
    });
}

// stream version of kmeans ++
// set      - dataset type (check data_types.hpp)
// mean     - cluster centers
// mt       - vector of random number generators
// H        - number of cluster centers
// threads  - threads object for parallelization
template <typename T>
void stream_d2(const dataset<T>& set, blaze::DynamicMatrix<T, blaze::rowMajor>& mean, std::vector<std::mt19937>& mt, int H, Tp& threads) {
    auto& files = set.files;
    auto& weight = set.weight;

    int N = set.shape.first;

    blaze::DynamicVector<T,  blaze::rowVector> min_dst(N);
    blaze::DynamicVector<int> min_idx(N);

    std::discrete_distribution<int> i(weight.begin(), weight.end());
    int first_index = i(mt[0]);

    // find index of closest element to n such than element < n. this will help us identify which file contains the index n
    auto upper = std::upper_bound(set.file_search.begin(), set.file_search.end(), first_index);
    auto idx = std::distance(set.file_search.begin(), upper) - 1;

    // read npy file
    blaze::DynamicMatrix<T, blaze::rowMajor> first_data;
    loadBlazeFromNumpy<T>(set.files[idx], first_data);

    // set initial value
    blaze::row(mean, 0) = blaze::row(first_data, first_index-set.file_search[idx]);

    int shift = 0;
    blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
    for (auto& f: files) {
        // read npy file
        loadBlazeFromNumpy<T>(f, tmp_data);

        // fill min_dst and min_idx without using much memory
        int temp_N = static_cast<int>(tmp_data.rows());
        threads.parallel(temp_N, [&] (int i, int t) {
            int n = i+shift;
            min_dst[n] = weight[n] * blaze::sqrNorm(blaze::row(tmp_data, i) - blaze::row(mean, 0));
            min_idx[n] = 0;
        });

        // shift by the last number of data points
        shift += temp_N;

        // clear temporary containers
        blaze::clear(tmp_data);
    }

    for (auto c=0; c<H; ++c) {
        std::discrete_distribution<int> q(min_dst.begin(), min_dst.end());
        int clu = q(mt[0]);

        // find index of closest element to n such than element < clu. this will help us identify which file contains the index clu
        upper = std::upper_bound(set.file_search.begin(), set.file_search.end(), clu);
        idx = std::distance(set.file_search.begin(), upper) - 1;

        // read npy file
        blaze::DynamicMatrix<T, blaze::rowMajor> this_data;
        loadBlazeFromNumpy<T>(set.files[idx], this_data);

        blaze::row(mean, c) = blaze::row(this_data, clu-set.file_search[idx]);

        int shift2 = 0;
        blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data2;
        for (auto& f: files) {
            // read npy file
            loadBlazeFromNumpy<T>(f, tmp_data2);

            int temp_N = static_cast<int>(tmp_data2.rows());
            for (auto i = 0; i < temp_N; ++i) {
                size_t n = i+shift2;
                T tmp = weight[n] * blaze::sqrNorm(blaze::row(tmp_data2, i) - blaze::row(mean, c));

                if (min_dst[n] > tmp) {
                    min_dst[n] = tmp;
                    min_idx[n] = clu;
                }
            }

            // shift by the last number of data points
            shift2 += temp_N;

            // clear temporary containers
            blaze::clear(tmp_data2);

        }
    }
}
