// Information: Collection of functions used to save and analyse data

#pragma once

#include <chrono>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <utility>

#include "data_types.hpp"

// external dependency
#include "blaze/Blaze.h"
#include "third_party/numpy.hpp"
#include "third_party/filesystem.hpp"

// saving cells (useful for building spatial histograms)
template <typename T>
void save_cells(std::string save_path, const dataset<T>& set) {
    if (!set.cells.empty()) {
        std::ostringstream cell_filename;
        if (set.train) {
            cell_filename << save_path << "/tr_cells.npy";
        } else {
            cell_filename << save_path << "/te_cells.npy";
        }

        // create save folder if it doesn't exist
        ghc::filesystem::path cell_current_dir(ghc::filesystem::absolute(cell_filename.str()));
        auto cell_parent_path = cell_current_dir.parent_path();
        ghc::filesystem::create_directories(cell_parent_path);

        // saving as a 1D npy file
        const int shape[1] = {static_cast<int>(set.cells.size())};
        saveArrayAsNumpy(cell_filename.str(), false, 1, &shape[0], &set.cells[0]);
    }
}

// save labels
template <typename T>
void save_labels(std::string save_path, std::string name, const dataset<T>& set, blaze::DynamicMatrix<int,blaze::rowMajor> labels) {
    // save labels
    std::ostringstream filename;
    if (set.train) {
        filename << save_path << "/tr_" << name << ".npy";
    } else {
        filename << save_path << "/te_" << name << ".npy";
    }

    // create save folder if it doesn't exist
    ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
    auto parent_path = current_dir.parent_path();
    ghc::filesystem::create_directories(parent_path);

    // saving blaze matrix as a npy file
    saveBlazeAsNumpy(filename.str(), false, labels);

    // save number of time surfaces/image patches per data point (useful for building histograms)
    std::ostringstream count_filename;
    if (set.train) {
        count_filename << save_path << "/tr_count.npy";
    } else {
        count_filename << save_path << "/te_count.npy";
    }

    // create save folder if it doesn't exist
    ghc::filesystem::path count_current_dir(ghc::filesystem::absolute(count_filename.str()));
    auto count_parent_path = count_current_dir.parent_path();
    ghc::filesystem::create_directories(count_parent_path);

    // saving as a 1D npy file
    const int shape[1] = {static_cast<int>(set.count.size())};
    saveArrayAsNumpy(count_filename.str(), false, 1, &shape[0], &set.count[0]);
}

// save percentage
template <typename T>
void save_percentage(std::string save_path, const dataset<T>& set) {
    std::ostringstream filename;
    if (set.train) {
        filename << save_path << "/tr_percentage.npy";
    } else {
        filename << save_path << "/te_percentage.npy";
    }

    ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
    auto parent_path = current_dir.parent_path();
    ghc::filesystem::create_directories(parent_path);

    const int s[1] = {1};
    saveArrayAsNumpy(filename.str(), false, 1, &s[0], &set.percentage);
}

// saves quantization error of provided dataset
template <typename T>
void save_quantization(std::string save_path, T quantization, bool train=false, int iterations=0) {

    if (!std::isinf(quantization)) {
        std::ostringstream filename;

        if (train) {
            filename << save_path << "/tr_quantization/" << std::to_string(iterations) << ".npy";
        } else {
            filename << save_path << "/te_quantization.npy";
        }

        ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
        auto parent_path = current_dir.parent_path();
        ghc::filesystem::create_directories(parent_path);

        const int shape[1] = {1};
        saveArrayAsNumpy(filename.str(), false, 1, &shape[0], &quantization);
    }
}

template <typename T>
void save_covariance(std::string save_path, blaze::DynamicMatrix<T, blaze::rowMajor>& covariance) {
    if (!blaze::isEmpty(covariance)) {
        std::ostringstream filename;
        filename << save_path << "/covariance.npy";

        ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
        auto parent_path = current_dir.parent_path();
        ghc::filesystem::create_directories(parent_path);

        saveBlazeAsNumpy(filename.str(), false, covariance);
    }
}

void save_distance_evaluations(std::string save_path, int64_t dist_evals, double D, int verbose) {
    if (verbose > 0) {
        std::cout << "distance evaluations:" << dist_evals << std::endl;
    }
    std::ostringstream filename;
    filename << save_path << "/distance_evals.npy";

    ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
    auto parent_path = current_dir.parent_path();
    ghc::filesystem::create_directories(parent_path);

    std::vector<int64_t> dist_vector;
    dist_vector.emplace_back(dist_evals);
    dist_vector.emplace_back(D);

    const int shape[1] = {static_cast<int>(dist_vector.size())};
    saveArrayAsNumpy(filename.str(), false, 1, &shape[0], &dist_vector[0]);
}

// save coreset+seeding runtime and EM runtime
void save_runtime(std::string save_path, std::chrono::duration<double> seeding_time, std::chrono::duration<double> em_time, int verbose) {
    if (verbose > 0) {
        std::cout << "seeding time:" << seeding_time.count() <<  " " << "EM time:" << em_time.count() << std::endl;
    }

    std::ostringstream filename;
    filename << save_path << "/runtime.npy";

    ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
    auto parent_path = current_dir.parent_path();
    ghc::filesystem::create_directories(parent_path);

    std::vector<double> runtime_vector{seeding_time.count(), em_time.count()};

    const int shape[1] = {static_cast<int>(runtime_vector.size())};
    saveArrayAsNumpy(filename.str(), false, 1, &shape[0], &runtime_vector[0]);
}

// save model parameters for all iteration
template <typename T>
void save_parameters(std::string save_path, int iterations, std::vector<T>& parameter_save) {
    // creating filename
    std::ostringstream filename;
    filename << save_path << "/parameters.npy";

    // create save folder if it doesn't exist
    ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
    auto parent_path = current_dir.parent_path();
    ghc::filesystem::create_directories(parent_path);

    const int shape[2] = {iterations+1, 3};
    saveArrayAsNumpy(filename.str(), false, 2, &shape[0], &parameter_save[0]);
}

// save centers called at every iteration with save_center option else only the centers at the final iteration are saved
template <typename T>
void save_centers(std::string save_path, int iterations, blaze::DynamicMatrix<T, blaze::rowMajor>& mean) {
    std::ostringstream filename;
    filename << save_path << "/centers/" << std::to_string(iterations) << ".npy";

    ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
    auto parent_path = current_dir.parent_path();
    ghc::filesystem::create_directories(parent_path);

    saveBlazeAsNumpy(filename.str(), false, mean);
}

// saving priors when not uniform
template <typename T>
void save_priors(std::string save_path, int iterations, blaze::DynamicVector<T, blaze::rowVector>& alpha) {
    if (!blaze::isEmpty(alpha)) {
        std::ostringstream filename;
        filename << save_path << "/priors/" << std::to_string(iterations) << ".npy";

        ghc::filesystem::path current_dir(ghc::filesystem::absolute(filename.str()));
        auto parent_path = current_dir.parent_path();
        ghc::filesystem::create_directories(parent_path);

        saveBlazeAsNumpy(filename.str(), false, alpha);
    }
}
