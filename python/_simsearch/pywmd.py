# (c) Copyright IBM Corporation 2019, 2020, 2021

from scipy import sparse
import numpy as np
from _simsearch import libpywmd

import os
import multiprocessing

def _get_mpi_local_size():
    val = os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE')
    if val is None:
        return 1
    else:
        return int(val)

def _compute_num_threads(verbose=False):

    affinity_count = None
    if hasattr(os, 'sched_getaffinity'):
        affinity_count = len(os.sched_getaffinity(0))
        if (verbose):
            print("os supports sched_getaffinity: affinity_count = " + str(affinity_count))

    cpu_count = None

    if hasattr(multiprocessing, 'cpu_count'):
        cpu_count = multiprocessing.cpu_count()
        if (verbose):
            print("multiprocessing supports cpu_count: cpu_count = " + str(cpu_count))
    elif hasattr(os, 'cpu_count'):
        cpu_count = os.cpu_count()
        if (verbose):
            print("os supports cpu_count: cpu_count = " + str(cpu_count))
    else:
        cpu_count = affinity_count
        if (verbose):
            print("cpu_count = " + str(cpu_count))
    
    process_count = _get_mpi_local_size()  
    if (verbose):
        print("process_count = " + str(process_count))

    if cpu_count is not None:
        num_threads = int(cpu_count / process_count)
        #if affinity_count is not None:
        #    num_threads = min(num_threads, affinity_count)
        num_threads = min(40, num_threads)
        num_threads = max(1, num_threads) 
    else:
        num_threads = 1

    if (verbose):
        print("num_threads = " + str(num_threads))

    return num_threads

#######################################
# COMPUTE WORD MOVERS DISTANCE
def word_movers_distance(W, X, Y=None, use_gpu=True, optimization_level=1, num_cpu_threads=None, use_cosine=False):
    """Computes Word Mover's Distance between samples in X and Y given embedding vectors W.

    The details of the approximation algorithm can be found here:
    "http://export.arxiv.org/abs/1812.02091".

    Parameters
    ----------
    W : ndarray, shape: (n_features, n_dimensions)
        Embedding vectors

    X : array-like, sparse-matrix, shape (n_samples_x, n_features)
        Input dataset. When MPI is used, it is preferred to have X of larger
        size than Y when X and Y are of unequal sizes as the MPI distribution
        of data is performed for X only.

    Y : array-like, sparse-matrix, shape (n_samples_y, n_features), optional
        Input dataset. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.

    optimization_level : integer, mimimum is 0 and maximum is 4. Using a higher value increases runtime.

    num_cpu_threads : integer, maximum number of CPU threads used by each process. 

    use_cosine : boolean, when false computes Euclidean ground distances between the embedding vectors,
		 when true computes (1 - Cosine Similarity) after normalizing the embedding vectors. 

    Returns
    -------
    D : array-like, shape (n_samples_x, n_samples_y)
        Word Movers Distance
    """

    if num_cpu_threads is None:
        num_cpu_threads = _compute_num_threads()

    if Y is None:
        Y=X

    cosine_flag = 0
    if (use_cosine):
        cosine_flag=1;

    gpu_flag = 0
    if (use_gpu):
        gpu_flag = 1

    D = np.zeros((X.shape[0], Y.shape[0]), dtype='float32')

    libpywmd.wmd_wrapper(D, W, X.nnz, X.indptr, X.indices, X.data, Y.nnz, Y.indptr, Y.indices, Y.data, gpu_flag, optimization_level, num_cpu_threads, cosine_flag)

    return(D)

#######################################
# COMPUTE EARTH MOVERS DISTANCE
def earth_movers_distance(D_ground, X, Y=None, use_gpu=True, optimization_level=1, num_cpu_threads=None, symmetric=True):
    """Computes Earth Mover's Distance between samples in X and Y given ground distances D_ground.

    The details of the approximation algorithm can be found here:
    "http://export.arxiv.org/abs/1812.02091".

    Parameters
    ----------
    D_ground : ndarray, shape: (n_features, n_features2)
        Ground distances

    X : array-like, sparse-matrix, shape (n_samples_x, n_features)
        Input dataset. When MPI is used, it is preferred to have X of larger
        size than Y when X and Y are of unequal sizes as the MPI distribution
        of data is performed for X only.

    Y : array-like, sparse-matrix, shape (n_samples_y, n_features2), optional
        Input dataset. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.

    optimization_level : integer, mimimum is 0 and maximum is 4. Using a higher value increases runtime.

    num_cpu_threads : integer, maximum number of CPU threads used by each process. 

    symmetric: whether D_ground is a symmetric array or not

    Returns
    -------
    D : array-like, shape (n_samples_x, n_samples_y)
        Earth Movers Distance
    """

    if num_cpu_threads is None:
        num_cpu_threads = _compute_num_threads()

    if Y is None:
        Y=X

    gpu_flag = 0
    if (use_gpu):
        gpu_flag = 1

    symmetric_flag = 0
    if (symmetric):
        symmetric_flag = 1

    D = np.zeros((X.shape[0], Y.shape[0]), dtype='float32')

    libpywmd.emd_wrapper(D, D_ground, X.nnz, X.indptr, X.indices, X.data, Y.nnz, Y.indptr, Y.indices, Y.data, gpu_flag, optimization_level, num_cpu_threads, symmetric_flag)

    return(D)

#######################################
# COMPUTE GROUND DISTANCES (EUCLIDEAN OR COSINE SIMILARITY BASED)
def compute_ground_distances(W, W2=None, use_gpu=True, use_cosine=False):
    """Computes pairwise distances between the vectors in W.

    Parameters
    ----------
    W : ndarray, shape: (n_features, n_dimensions)
        Embedding vectors

    W2 : ndarray, shape: (n2_features, n_dimensions)
        Second set of embedding vectors (optional)

    use_cosine : boolean, when false computes Euclidean ground distances between the embedding vectors,
		 when true computes (1 - Cosine Similarity) after normalizing the embedding vectors. 

    Returns
    -------
    D : array-like, shape (n_features, n_features) or (n_features, n2_features)
        Ground Distances between Embedding Vectors
    """

    cosine_flag = 0
    if (use_cosine):
        cosine_flag=1

    gpu_flag = 0
    if (use_gpu):
        gpu_flag = 1

    if W2 is None:
        D = np.zeros((W.shape[0], W.shape[0]), dtype='float32')
    else: 
        D = np.zeros((W.shape[0], W2.shape[0]), dtype='float32')

    libpywmd.compute_ground_distances(D, W, W2, gpu_flag, cosine_flag)

    return(D)

#######################################

