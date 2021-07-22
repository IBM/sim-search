# (c) Copyright IBM Corporation 2019, 2020, 2021

import errno
import os
import numpy as np
from scipy import sparse
from _simsearch import libpywmd
from sklearn.preprocessing import normalize

#######################################
# LOAD 20 NEWSGROUPS DATA
def load_20News(directory_path, vocabulary_file_path = "vocabulary.txt", norm='l1', max_histogram_size=500, stop_word_threshold=900):
    """ Loads 20News dataset into arrays X, labels, ids

    Parameters
    ----------
    directory_path : string
        Source location for the data

    vocabulary_file_path : string
        Vocabulary that is used in tokenization and vectorization of text
    
    norm : string (default='l1')
        l2 if the user wants to normalize the data using l2 normalization

    max_histogram_size : integer (default=500)
        Each document histogram stores only the top max_histogram_size most-frequent words of the respective document.

    stop_word_threshold : integer (default=900)
        The first stop_word_threshold words of the vocabulary are treated as stop words, and omitted in the histograms.

    Returns
    -------
    X : array-like, sparse_matrix, shape (n_samples, n_features)
        Feature vectors      

    labels : array-like, shape (n_samples,)
        labels are the class labels of the samples in X

    ids: array_like, shape (n_samples,)
        ids are the ids of the samples in X
    """
    
    if(not os.path.exists(directory_path)):
        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), directory_path)
    if(not os.path.exists(vocabulary_file_path)):
        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), vocabulary_file_path)
    else:
        vocab_size, nrow, nnz, indptr, indices, data, labels, ids = libpywmd.load_20News(directory_path, vocabulary_file_path, max_histogram_size, stop_word_threshold)
        if (nnz==-1):
            raise IOError("Failed to parse input files")
        else:
            X = sparse.csr_matrix( (data,indices,indptr), shape=(nrow, vocab_size))
            X = normalize(X, norm, copy=False)
            return(X, labels, ids)

#######################################
# LOAD MNIST DATA WITH EMBEDDINGS
def load_MNIST_with_embeddings(image_file_path, label_file_path, norm='l1', use_sparse=True):
    """ Loads MNIST dataset into arrays W, X, labels, ids

    Parameters
    ----------
    image_file_path : string
        file path location for the MNIST images

    label_file_path : string
        file path location for the MNIST labels

    norm : string (default='l1')
        l2 if the user wants to normalize the data using l2 normalization

    use_sparse : bool (default='True')
        False if the user wants to load MNIST in non-sparse format

    Returns
    -------
    W : array_like, shape (n_features, n_dimensions)
        Embedding vectors

    X : array_like, sparse_matrix, shape (n_samples, n_features)
        Feature vectors

    labels : array-like, shape (n_samples,)
        labels are the class labels of the samples in X

    ids: array_like, shape (n_samples,)
        ids are the ids of the samples in X
    """

    if(not os.path.exists(image_file_path)):
        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), image_file_path)
    if(not os.path.exists(label_file_path)):
        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), label_file_path)
    else:
        W, nrow, nnz, indptr, indices, data, labels, ids = libpywmd.load_mnist_with_embeddings(image_file_path, label_file_path)
        if (nnz==-1):
            raise IOError("Failed to parse input files")
        else:
            X = sparse.csr_matrix( (data,indices,indptr), shape=(nrow, W.shape[0]))
            X = normalize(X, norm, copy=False)
            if not (use_sparse):
                X = X.todense()
            return(W, X, labels, ids)
