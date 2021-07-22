#### Description ####

The SimSearch library offers efficient CPU and GPU implementations of a low-complexity approximation of Earth/Word Mover's Distance. The details of the approximation algorithm are outlined in "http://proceedings.mlr.press/v97/atasu19a.html

This repository consists of 
1) dynamically linked shared object library (.so) and include files that implement a C API,
2) C and python source files that implement a python/C interface and a python API,
3) examples that demonstrate how to use the python and the C APIs.

#### Hardware requirements ####

An NVIDIA GPU with at least 8GB main memory available when using the GPU-accelerated functions

#### Directory structure ####

lib: compiled SimSearch C library

include: header files of SimSearch C library

test_mnist: a test module that exercises SimSearch API on the MNIST dataset using the C interface

test_20news: a test module that exercises SimSearch API on the 20 Newsgroups dataset using the C interface

python: SimSearch python API and test modules to exercise the python API on MNIST and 20 Newsgroups datasets

MNIST: MNIST image dataset of handwritten digits (copy the data files into this directory)

20news-18828: 20 Newsgroups text dataset (expected location of the uncompressed data files)

Embeddings: Word2Vec Embeddings (copy pre-trained word embeddings into this directory)

#### Using precompiled libraries

Dependencies: Eigen (http://eigen.tuxfamily.org/) and NVIDIA CUDA 10.2 libraries

cd lib 

To use X86 libraries: cp X86_64_linux_gcc4.8.5_openmpi/*.* ./

To use PPC libraries: cp PPC_64_gcc6.4.1_at10.0_openmpi/*.* ./

cd ..

#### Running the MNIST test using the C interface ####

Download train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, and t10k-labels-idx1-ubyte.gz  from http://yann.lecun.com/exdb/mnist/, uncompress and move into SimSearch/MNIST directory.

cd test_mnist

cd build

cmake ..

make

If MPI is to be enabled, then pass USE_MPI=1 through -DUSE_MPI=1 to the above cmake command.

We can run the test as follows:

If mpi was enabled during the build process: mpirun -np 4 ./test_mnist_mpi

If mpi was not enabled during the build process: ./test_mnist

This test will compare 10000 test images with 60000 training images. For each image in the test set, the top-32 nearest neighbors in the training set will be computed and averaged precision results will be printed.

#### Running the 20 Newsgroups test using the C interface ####

Download 20news-18828.tar.gz from http://qwone.com/~jason/20Newsgroups/, uncompress and move into SimSearch/20news-18828 directory.

Download GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/, uncompress and move into SimSearch/Embeddings directory.

cd test_20news

cd build

cmake ..

make

If MPI is to be enabled, then pass USE_MPI=1 through -DUSE_MPI=1 to the above cmake command.

We can run the test as follows:

If mpi was enabled during the build process: mpirun -np 4 ./test_20news_mpi

If mpi was not enabled during the build process: ./test_20news

This test will compare 18828 20Newsgroups documents with each other. For each document in the database, the top-32 nearest neighbors will be computed and averaged precision results will be printed.

#### Running the MNIST test using the Python interface ####

Download train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, and t10k-labels-idx1-ubyte.gz  from http://yann.lecun.com/exdb/mnist/, uncompress and move into SimSearch/MNIST directory.

cd python

cd build

cmake ..

make

cd ..

If MPI is to be enabled, then pass USE_MPI=1 through -DUSE_MPI=1 to the above cmake command.

We can run the test as follows:

If mpi was enabled during the build process: mpirun -np 4 python test_mnist.py

If mpi was not enabled during the build process: python test_mnist.py

This test will compare 10000 test images with 60000 training images. For each image in the test set, the top-32 nearest neighbors in the training set will be computed and averaged precision results will be printed.

#### Running the 20 Newsgroups test using the Python interface ####

Download 20news-18828.tar.gz from http://qwone.com/~jason/20Newsgroups/, uncompress and move into SimSearch/20news-18828 directory.

Download GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/, uncompress and move into SimSearch/Embeddings directory.

cd python

cd build

cmake ..

make

cd ..

If MPI is to be enabled, then pass USE_MPI=1 through -DUSE_MPI=1 to the above cmake command.

Before running the test, it is necessary to preprocess the embedding vectors: 

python preprocess_word2vec_embeddings.py

We can run the test as follows:

If mpi was enabled during the build process: mpirun -np 4 python test_20news.py

If mpi was not enabled during the build process: python test_20news.py

This test will compare 18828 20Newsgroups documents with each other. For each document in the database, the top-32 nearest neighbors will be computed and averaged precision results will be printed.

&copy; Copyright IBM Corporation 2019, 2020, 2021
