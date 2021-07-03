![Logo](resources/peregrine.png)

This repository contains code for an implementation of an efficient variational gaussian mixture model on multiple datasets:
* Data Similarity Gaussian Mixture Model (D-GMM)
* Stochastic Gaussian Mixture Model (S-GMM)

## Dependencies
* [CMake](https://cmake.org)
* [Premake4](https://premake.github.io)
* [Blaze](https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation)
* [Intel TBB](https://github.com/oneapi-src/oneTBB)
* [pytorch 1.4+](https://pytorch.org)

## Running a classification task

First, generate time surfaces, which we are going to cluster afterwards.
* time surfaces from the N-MNIST, NCARS and DVSGesture event-based datasets: time_surfaces.ipynb
* time surfaces from the N-CALTECH101 event-based dataset: N-Caltech101.ipynb

Main entry point is variational-gmm.ipynb, which runs the GMM clustering algorithm and then uses standard scikit-learn or pytorch classifiers to analyse the results.

1. We first need to make sure the C++ source code is compiled (C++14 required) by running

~~~~
premake4 gmake && cd build && make
~~~~

2. We can then open the python notebook and start classifying datasets. There's no need to run the compilation step again unless you make changes to the C++ source code

#### Dependencies for working with time surfaces

* Tonic (Spike manipulation and augmentation): https://github.com/neuromorphs/tonic.git

~~~
pip install tonic
~~~

## Installation

#### CMake and Premake4

###### On Mac

You can install cmake and premake via the homebrew package manager
~~~~
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install cmake premake
~~~~

###### On Linux
~~~~
sudo apt-get install cmake premake4
~~~~

#### Blaze

Make sure you have BLAS and LAPACK installed first
~~~~
sudo apt-get install libopenblas-dev
~~~~

Proceed to the blaze installation
~~~~
git clone https://bitbucket.org/blaze-lib/blaze.git
cd blaze && cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
sudo make install
~~~~

#### Intel TBB

###### On Mac

You can install intel TBB via the homebrew package manager
~~~~
brew install tbb
~~~~

###### On Linux
~~~~
sudo apt install libtbb-dev
~~~~
