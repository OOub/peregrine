// Information: AFK-MC^2 sampling (O. Bachem, M. Lucic, H. Hassani, and A. Krause. Fast and provably good seedings for k-means. In Proc. Advances in Neural Information Processing Systems, pages 55–63, 2016a)

#pragma once

#include <iostream>
#include <vector>
#include <utility>
#include <random>

#include "../threads.hpp"
#include "../data_types.hpp"

// external dependency
#include "blaze/Blaze.h"

// normal version of AFK-MC^2
// set      - dataset type (check data_types.hpp)
// mean     - cluster centers
// mt       - vector of random number generators
// H        - number of cluster centers
// chain    - Markov chain length
// threads  - threads object for parallelization
template <typename T>
void afkmc2(const dataset<T>& set, blaze::DynamicMatrix<T, blaze::rowMajor>& mean, std::vector<std::mt19937_64>& mt, int H, int chain, Tp& threads) {
    auto& data = set.data;
    auto& weight = set.weight;

    int N = static_cast<int>(data.rows());
    int D = static_cast<int>(data.columns());

    // draw first cluster
    std::discrete_distribution<int> i(weight.begin(), weight.end());
    mean.resize(H, D);
    blaze::row(mean, 0) = blaze::row(data, i(mt[0]));

    // compute proposal distribution
    blaze::DynamicVector<T, blaze::rowVector> q(N);
    {
        threads.parallel(N, [&] (int n, int t) {
            q[n] = blaze::sqrNorm(blaze::row(data, n) - blaze::row(mean, 0)) * weight[n];
        });

        T dsum = static_cast<T>(0.0);
        T wsum = static_cast<T>(0.0);

        threads.parallel(N, [&] (int n, int t) {
            dsum += q[n];
            wsum += weight[n];
        });

        threads.parallel(N, [&] (int n, int t) {
            q[n] = static_cast<T>(0.5) * (q[n] / dsum + weight[n] / wsum);
        });

    }

    std::discrete_distribution<int> draw_q(q.begin(), q.end());
    std::uniform_real_distribution<T> uniform(static_cast<T>(0.0), static_cast<T>(1.0));

    threads.parallel(H, [&] (int h, int t) {

        // initialize a new Markov chain
        int data_idx = draw_q(mt[t]);
        T data_key;

        // compute distance to closest cluster
        T dist = std::numeric_limits<T>::max();
        for (int _h = 0; _h < h; _h++) {
            dist = std::min(dist, blaze::sqrNorm(blaze::row(data, data_idx) - blaze::row(mean, _h)));
        }
        data_key = dist * weight[data_idx];

        // Markov chain
        for (int i = 1; i < chain; i++) {

            // draw new potential cluster center from proposal distribution
            int y_idx = draw_q(mt[t]);
            T y_key;

            // compute distance to closest cluster
            T dist = std::numeric_limits<T>::max();
            for (int _h = 0; _h < h; _h++) {
                dist = std::min(dist, blaze::sqrNorm(blaze::row(data, y_idx) - blaze::row(mean, _h)));
            }
            y_key = dist * weight[y_idx];


            // determine the probability to accept the new sample y_idx
            T y_prob = y_key / q[y_idx];
            T data_prob = data_key / q[data_idx];

            if (((y_prob / data_prob) > uniform(mt[t])) || (data_prob == 0)) {
                data_idx = y_idx;
                data_key = y_key;
            }
        }

        blaze::row(mean, h) = blaze::row(data, data_idx);
    });
}

// stream version of AFK-MC^2
// set      - dataset type (check data_types.hpp)
// mean     - cluster centers
// mt       - vector of random number generators
// H        - number of cluster centers
// chain    - Markov chain length
// threads  - threads object for parallelization
template <typename T>
void stream_afkmc2(const dataset<T>& set, blaze::DynamicMatrix<T, blaze::rowMajor>& mean, std::vector<std::mt19937_64>& mt, int H, int chain, Tp& threads) {
    throw std::logic_error("not implemented yet");
}
