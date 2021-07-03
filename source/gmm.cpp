// Information: Clustering with different variational GMM algorithms applied to image patches and time surfaces for classification with various datasets

#include "gmm.hpp"

int main(int argc, char** argv) {

    // error handling
    if (argc < 19) {
        throw std::runtime_error(std::to_string(argc).append(" received. Expected at least 18 arguments"));
    }

    // data parameters
    std::string path_train_features = argv[1];             // path to training features
    std::string path_test_features  = argv[2];             // path to test features (empty string if we don't want a test set)
    std::string save_path           = argv[3];             // directory where we save the output
    int sample_percentage           = std::atoi(argv[4]);  // take a data sample for the training set instead of full data (between 1 and 100)

    // model parameters
    int training_model              = std::atoi(argv[5]);  // 1:u-S-GMM | 2:S-GMM
    int M                           = std::atoi(argv[6]);  // number of cluster centers
    int H                           = std::atoi(argv[7]);  // number of clusters considered for each data point
    int R                           = std::atoi(argv[8]);  // number of new samples
    int Nprime                      = std::atoi(argv[9]); // size of subset | set to 0 to disable coreset creation
    bool stream                     = std::atoi(argv[10]); // stream data from harddrive to build coreset | training data not loaded and only works when using coresets
    int chain_length                = std::atoi(argv[11]); // chain length for AFK-MC² seeding
    double convergence_threshold    = std::stod(argv[12]); // < 1 for convergence threshold | >= 1 for epochs

    // output parameters
    int trials                      = std::atoi(argv[13]); // number of trials in parallel
    int top_k                       = std::atoi(argv[14]); // hard clustering on the best k features
    bool inference                  = std::atoi(argv[15]); // save cluster assignments for training and test datasets (when streaming doesn't work for training set)
    bool soft_clustering            = std::atoi(argv[16]); // choose between soft and hard clustering
    int save_additional_info        = std::atoi(argv[17]); // 1: save centers, error, and priors across iterations | 2: only priors
    int verbose                     = std::atoi(argv[18]); // printing output across iterations (1 or 2 for even more information)

    int nthreads = std::thread::hardware_concurrency();    // number of C++11 threads

    // strip last / from save_path if present
    if (save_path.back() == '/') {
        save_path.pop_back();
    }

    // reading data
    dataset<double> training_dataset = load_data<double>(path_train_features, true, nthreads, Nprime, stream, sample_percentage); // training dataset
    dataset<double> test_dataset = load_data<double>(path_test_features, false, nthreads); // test dataset

    if (trials <= 1) {
        // fitting model
        auto gmm = Gmm<double>(training_dataset, M, chain_length, H, R, nthreads, training_model, verbose);
        gmm.fit(convergence_threshold, save_additional_info, save_path);

        // inference on training and test datasets
        if (inference) {
            if (soft_clustering) {
                gmm.soft_predict(save_path, training_dataset);
                gmm.soft_predict(save_path, test_dataset);
            } else {
                gmm.predict(save_path, training_dataset, top_k);
                gmm.predict(save_path, test_dataset, top_k);
            }
        }

        // save error of test set
        double error = gmm.error(test_dataset);
        save_quantization<double>(save_path, error);

        // save percentage training data
        save_percentage<double>(save_path, training_dataset);

    } else {
        std::vector<size_t> run_indices(trials);
        std::iota(run_indices.begin(), run_indices.end(), 0);

        tbb::parallel_for(static_cast<size_t>(0), static_cast<size_t>(trials), [&](size_t i) {
            std::string parallel_path = save_path + "/" + std::to_string(i);

            // fitting model
            auto gmm = Gmm<double>(training_dataset, M, chain_length, H, R, nthreads, training_model, verbose);
            gmm.fit(convergence_threshold, save_additional_info, parallel_path);

            // inference on training and test datasets
            if (inference) {
                if (soft_clustering) {
                    gmm.soft_predict(parallel_path, training_dataset);
                    gmm.soft_predict(parallel_path, test_dataset);
                } else {
                    gmm.predict(parallel_path, training_dataset, top_k);
                    gmm.predict(parallel_path, test_dataset, top_k);
                }
            }

            // save error of test set
            double error = gmm.error(test_dataset);
            save_quantization<double>(parallel_path, error);

            // save percentage training data
            save_percentage<double>(parallel_path, training_dataset);

        });
    }

    // exit program
    return 0;
}
