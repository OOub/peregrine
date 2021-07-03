![Logo](resources/peregrine.png)

This repository contains code which implements the Stochastic Gaussian Mixture Model (S-GMM) for event-based datasets

## Dependencies
* [CMake](https://cmake.org)
* [Premake4](https://premake.github.io)
* [Blaze](https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation)
* [Intel TBB](https://github.com/oneapi-src/oneTBB)
* [pytorch 1.4+](https://pytorch.org)
* [Tonic](https://github.com/neuromorphs/tonic.git)

## Running a classification task

1. Generate time-surfaces using: time_surfaces.ipynb (look at tonic for more details on what datasets are support)
2. Compile C++14 source code
~~~~
premake4 gmake && cd build && make
~~~~
3. Main entry point for clustering and classification: variational-gmm.ipynb

## Installation

#### Tonic
~~~
pip install tonic
~~~

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
