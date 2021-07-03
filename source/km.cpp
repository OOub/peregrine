// Information: clustering with k-means as a baseline

#include "kmeans.hpp"

int main(int argc, char** argv) {

    // error handling
    if (argc < 13) {
        throw std::runtime_error(std::to_string(argc).append(" received. Expected at least 12 arguments"));
    }

    // data parameters
    std::string path_train_features = argv[1];             // path to training features
    std::string path_test_features  = argv[2];             // path to test features (empty string if we don't want a test set)
    std::string save_path           = argv[3];             // directory where we save the output
    int sample_percentage           = std::atoi(argv[4]);  // take a data sample for the training set instead of full data (between 1 and 100)

    // model parameters
    int M                           = std::atoi(argv[5]);  // number of cluster centers
    int Nprime                      = std::atoi(argv[6]);  // size of subset. set to 0 to disable coresets
    bool stream                     = std::atoi(argv[7]);  // stream data from harddrive
    double convergence_threshold    = std::stod(argv[8]);  // < 1 for convergence threshold | >= 1 for epochs

    // output parameters
    int trials                      = std::atoi(argv[9]); // number of trials in parallel
    bool inference                  = std::atoi(argv[10]); // save cluster assignments for training and test datasets
    int save_additional_info        = std::atoi(argv[11]); // 1: save centers and errors across iterations
    int verbose                     = std::atoi(argv[12]); // printing output across iterations

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
        auto km = Kmeans<double>(training_dataset, M, nthreads, verbose);
        km.fit(convergence_threshold, save_additional_info, save_path);

        // inference on training and test datasets
        if (inference) {
            km.predict(save_path, training_dataset);
            km.predict(save_path, test_dataset);
        }

        // save error
        double error = km.error(test_dataset);
        save_quantization<double>(save_path, error);

        // save percentage training data
        save_percentage<double>(save_path, training_dataset);

    } else {
        std::vector<size_t> run_indices(trials);
        std::iota(run_indices.begin(), run_indices.end(), 0);

        tbb::parallel_for(static_cast<size_t>(0), static_cast<size_t>(trials), [&](size_t i) {
            std::string parallel_path = save_path + "/" + std::to_string(i);

            // fitting model
            auto km = Kmeans<double>(training_dataset, M, nthreads, verbose);
            km.fit(convergence_threshold, save_additional_info, parallel_path);

            // inference on training and test datasets
            if (inference) {
                km.predict(parallel_path, training_dataset);
                km.predict(parallel_path, test_dataset);
            }

            // save error
            double error = km.error(test_dataset);
            save_quantization<double>(parallel_path, error);

            // save percentage training data
            save_percentage<double>(parallel_path, training_dataset);
        });
    }

    // exit program
    return 0;
}
