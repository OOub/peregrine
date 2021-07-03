// Information: Data types used throughout the source code

#pragma once

#include <vector>
#include <string>

// external dependency
#include "blaze/Blaze.h"

// ----- INPUT DATA -----
// dataset structure
template <typename T>
struct dataset {
    blaze::DynamicMatrix<T, blaze::rowMajor>  data;
    std::pair<int, int>                       shape;
    blaze::DynamicVector<T, blaze::rowVector> weight;
    bool                                      train;
    std::vector<int>                          labels;
    std::vector<int>                          cells;
    std::vector<int>                          count;
    std::vector<std::string>                  files;
    std::vector<int>                          file_search;
    bool                                      coreset = false;
    int64_t                                   dist_evals = 0;
    std::chrono::duration<double>             runtime = static_cast<std::chrono::duration<double>>(0.0);
    int                                       percentage = 100;
};

// ----- STRUCTURES FOR THE ALGORITHMS -----
// lookup table to know which clusters are assigned to which datapoints
struct _tab {
    int opc;
    int ref;
    _tab() :
        opc(std::numeric_limits<int>::max()),
        ref(std::numeric_limits<int>::max()){}
};

// Kn structure containing:
// 1. clusters
// 2. squared euclidean distance between datapoints and means
template <typename T>
struct _dcn {
    int cluster;
    T distance;
};
