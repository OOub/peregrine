// Information: loading datasets and creating lightweight corsesets

#pragma once

#include <chrono>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <random>

#include "data_types.hpp"

// external dependency
#include "blaze/Blaze.h"
#include "third_party/numpy.hpp"
#include "third_party/filesystem.hpp"

// loads npy files into a blaze matrix. automatically recognizes if the given path is a dataset or a single file
//      - if path is a directory of npy file:
//          * loads data into matrix from path/data (only if stream option if set to FALSE)
//          * loads labels into integer vector from path/labels if available
//          * saves the number of features loaded from each file
//          * saves npy file paths into files string vector
//
//      - if path is one npy file only loads data into matrix
template <typename T>
const dataset<T> load_data(const std::string& path, bool train, int nthreads=64, int Nprime=0, bool stream=false, int sample_percentage=100) {

    // error handling
    if (sample_percentage > 100 || sample_percentage <= 0) {
        throw std::logic_error("your sample is a percentage that needs to be between 1 and 100");
    }

    if (!path.empty() && !ghc::filesystem::exists(path)) {
        throw std::logic_error("wrong path to dataset or npy file");
    }

    // initialise dataset
    dataset<T> set;

    // set training or test set
    set.train = train;

    ghc::filesystem::path given_path = path;
    if (ghc::filesystem::is_regular_file(path) && given_path.extension() == ".npy") {
        // read npy file into blaze matrix
        loadBlazeFromNumpy<T>(path, set.data);

        set.shape.second = static_cast<int>(set.data.columns());

        if (sample_percentage < 100) {
            auto number_of_samples = std::ceil(set.data.rows() * sample_percentage / 100);
            set.data.resize(number_of_samples, set.shape.second);
        }

        set.shape.first = static_cast<int>(set.data.rows());

    } else if (ghc::filesystem::is_directory(given_path)) {

        // read the header text file which contains the dimensions of the dataset (Number of data points (line 1), Dimensions (line 2) and Channels (line 3))
        std::vector<int> temp_shape;
        std::ifstream header(given_path/"header.txt");
        if (header.good()) {
            std::string line;
            while (std::getline(header, line)) {
                temp_shape.emplace_back(std::stoi(line));
            }
        } else {
            throw std::runtime_error("header.txt for the given directory could not be opened. Please check that a header.txt file is present in the dataset directory");
        }

        // save matrix shape
        set.shape.first = temp_shape[0]*temp_shape[2];
        set.shape.second = temp_shape[3];

        // check if labels are present in the directory tree
        bool labels_exist = ghc::filesystem::exists(given_path/"labels");

        // check if cells are present in the directory tree (these basically split timesurfaces into areas of a grid)
        bool cells_exist = ghc::filesystem::exists(given_path/"cells");

        ghc::filesystem::path data_path = given_path/"data";

        if (ghc::filesystem::exists(data_path)) {
            std::vector<std::pair<int, std::string>> file_order;
            for (auto &file : ghc::filesystem::recursive_directory_iterator(data_path)) {
                if (file.path().extension() == ".npy") {
                    // save file path
                    file_order.emplace_back(std::make_pair(std::stoi(file.path().stem()), file.path().c_str()));
                }
            }

            // sort files according to their name (which is an index)
            std::sort(file_order.begin(), file_order.end(), [&](std::pair<int, std::string> a, std::pair<int, std::string> b) {
                return a.first < b.first;});

            // reserve space for file search
            set.files.reserve(file_order.size());
            set.file_search.reserve(file_order.size());

            // maximum number of samples we want to achieve the sample_percentage
            auto samples_limit = std::ceil(set.shape.first * sample_percentage / 100);

            // resize data matrix if not using coresets
            if (!stream) {
                if (sample_percentage < 100) {
                    // reserve space for data with overestimation (adding a maximum of 15% capacity to expected space)
                    auto extra_percentage = 15;
                    if (100 - sample_percentage < extra_percentage) {
                        extra_percentage = 100 - sample_percentage;
                    }
                    auto overestimation = samples_limit+std::ceil(set.shape.first * extra_percentage / 100);
                    set.data.resize(overestimation,set.shape.second);
                } else {
                    set.data.resize(set.shape.first,set.shape.second);
                }
            }

            int shift = 0;
            int number_of_samples = 0;
            int number_of_files = 0;
            for (auto& f: file_order) {
                // create path from f string
                set.files.emplace_back(f.second);
                ghc::filesystem::path file = f.second;

                // read npy file
                std::vector<int> tmp_shape;
                blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
                loadBlazeFromNumpy<T>(file, tmp_shape, tmp_data);

                // get number of new features
                int new_N = std::accumulate(tmp_shape.begin(), tmp_shape.end()-1, 1, std::multiplies<int>());

                // error handling for numpy file specs
                if (tmp_shape.size() != 4) {
                    throw std::logic_error("numpy arrays should have 4 dimensions: [Batch Size x N x C x D]");
                }

                // we do not want to save the data in a large matrix if coresets are used
                if (!stream) {
                    // creating submatrix and filling submatrix of data with the new file
                    blaze::submatrix(set.data, shift, 0, new_N, tmp_shape[3]) = tmp_data;
                }

                // shift by the last number of data points and save in file_search
                set.file_search.emplace_back(shift);
                shift += new_N;

                // add corresponding cells
                if (cells_exist) {
                    auto cell_path = given_path/"cells"/file.filename();
                    std::vector<int32_t> tmp_cell;
                    loadArrayFromNumpy(cell_path, tmp_cell);

                    for (auto& c: tmp_cell) {
                        set.cells.emplace_back(c);
                    }
                }

                // add corresponding labels
                if (labels_exist) {
                    auto label_path = given_path/"labels"/file.filename();
                    std::vector<int32_t> tmp_label;
                    loadArrayFromNumpy(label_path, tmp_label);

                    for (auto& l: tmp_label) {
                        for (auto j=0; j<tmp_shape[1]*tmp_shape[2]; ++j) {
                            set.labels.emplace_back(l);
                        }
                    }

                    for (auto i=0; i<tmp_label.size() * tmp_shape[2]; ++i) {
                        set.count.emplace_back(tmp_shape[1]);
                    }
                }

                // save number of samples used (in case we want a subsample of the data)
                if (sample_percentage < 100 && number_of_samples < samples_limit) {
                    number_of_samples += new_N;
                    number_of_files += 1;
                }

                // get out of the for loop if we have enough samples
                if (sample_percentage < 100 && number_of_samples > samples_limit) {
                    break;
                }
            }

            if (sample_percentage < 100) {
                // calculate actual percentage taken from data
                set.percentage = (number_of_samples * 100) / set.shape.first;

                // changing shape to new matrix size
                set.shape.first = number_of_samples;

                // resize data matrix to remove overestimation
                if (!stream) {
                    set.data.resize(set.shape.first, set.shape.second);
                    set.data.shrinkToFit();
                }

                // take only files we used
                set.files.resize(number_of_files);
            }

        } else {
            throw std::runtime_error("the given directory tree is not in the required format. a dataset folder contains a data folder with all the npy data files and optionally a label folder with all the npy label files");
        }
    }

    if (Nprime > 0) {
        set.dist_evals = set.shape.first;
        auto start = std::chrono::high_resolution_clock::now();
        build_coreset(set, Nprime, nthreads);
        auto end = std::chrono::high_resolution_clock::now();
        set.runtime = end - start;
        set.coreset = true;
    } else {
        set.weight = blaze::DynamicVector<T, blaze::rowVector>(set.shape.first, static_cast<T>(1.0));
    }

    return set;
}

template <typename T>
void build_coreset(dataset<T>& set, int Nprime, int nthreads) {

    std::vector<std::mt19937> mt(nthreads);
    Tp threads(nthreads);

    auto N = set.shape.first;
    auto D = set.shape.second;

    blaze::DynamicVector<T, blaze::rowVector> u(D, static_cast<T>(0.0));
    blaze::DynamicVector<T, blaze::rowVector> q(N);

    auto coreset = blaze::DynamicMatrix<T, blaze::rowMajor>(Nprime, set.shape.second);
    set.weight.resize(Nprime);

    // compute mean
    if (!blaze::isEmpty(set.data)) {
        for (int n = 0; n < N; n++) {
            u += blaze::row(set.data, n);
        }
    } else {
        // if no data then stream from vector of files
        blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
        for (auto file: set.files) {
            // read npy file
            loadBlazeFromNumpy<T>(file, tmp_data);

            // loop through temp data matrix and add to u without using too much memory
            for (auto i=0; i<tmp_data.rows(); ++i) {
                u += blaze::row(tmp_data, i);
            }

            // clear matrix
            blaze::clear(tmp_data);
        }
    }
    u *= static_cast<T>(1.0)/N;

    // compute proposal distribution
    if (!blaze::isEmpty(set.data)) {
        threads.parallel(N, [&] (int n, int t) {
            q[n] = blaze::sqrNorm(blaze::row(set.data, n) - u);
        });
    } else {
        size_t shift = 0;
        blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
        for (auto file: set.files) {
            // read npy file
            loadBlazeFromNumpy<T>(file, tmp_data);

            // fill in q matrix without using much memory
            auto temp_N = static_cast<int>(tmp_data.rows());
            threads.parallel(temp_N, [&] (int i, int t) {
                size_t n = i+shift;
                q[n] = blaze::sqrNorm(blaze::row(tmp_data, i) - u);
            });

            // shift by the last number of data points
            shift += temp_N;


            // clear temporary containers
            blaze::clear(tmp_data);
        }
    }

    T sum = static_cast<T>(0.0);
    for (int n = 0; n < N; n++) {
        sum += q[n];
    }

    threads.parallel(N, [&] (int n, int t) {
        q[n] = static_cast<T>(0.5) * (q[n] / sum + static_cast<T>(1.0) / N);
    });


    std::random_device r;
    for (int t = 0; t < threads.size(); t++) {
        std::seed_seq seq{r(), r(), r(), r(), r(), r(), r(), r()};
        mt[t].seed(seq);
        mt[t].discard(1e3);
    }

    std::discrete_distribution<int> dst(q.begin(), q.end());
    if (!blaze::isEmpty(set.data)) {
        threads.parallel(Nprime, [&] (int m, int t) {
            // get sample and fill coreset
            int n = dst(mt[t]);
            blaze::row(coreset, m) = blaze::row(set.data, n);
            set.weight[m] = static_cast<T>(1.0) / (q[n] * Nprime);
        });
    } else {
        threads.parallel(Nprime, [&] (int m, int t) {
            // get sample
            int n = dst(mt[t]);

            // find index of closest element to n such than element < n. this will help us identify which file contains the index n
            auto upper = std::upper_bound(set.file_search.begin(), set.file_search.end(), n);
            auto idx = std::distance(set.file_search.begin(), upper) - 1;

            // read npy file
            blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
            loadBlazeFromNumpy<T>(set.files[idx], tmp_data);

            // fill coreset
            blaze::row(coreset, m) = blaze::row(tmp_data, n-set.file_search[idx]);
            set.weight[m] = static_cast<T>(1.0) / (q[n] * Nprime);
        });
    }

    // replace data with coreset
    blaze::clear(set.data);
    set.data = coreset;
    set.data.shrinkToFit();
    set.shape.first = Nprime;
}
