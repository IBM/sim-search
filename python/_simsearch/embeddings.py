# (c) Copyright IBM Corporation 2019, 2020, 2021

import errno
import os
import numpy as np
from _simsearch import libpywmd

#######################################
# LOAD EMBEDDINGS FROM WORD2VEC BINARY OUTPUT FILE
def load_word2vec_embeddings(embedding_file_path, vectors_file_path="vectors.npy", vocabulary_file_path = "vocabulary.txt"):
    """ Loads Word2Vec word-embedding vectors and vocabulary from embedding_file_path
    Saves the vectors into vectors_file_path in .npy format
    Saves the vocabulary into vocabulary_file_path in .txt format

    Parameters
    ----------

    embedding_file_path : string
        Indicates the absolute file path location for the embeddings file
    vectors_file_path : string (default="vectors.npy")
        Indicates the absolute file path location for storing the output ``vectors.npy`` file
    vocabulary_file_path : string (default="vocabulary.txt")
        Indicates the absolute file path location for storing the output ``vocabulary.txt`` file

    Returns
    -------

    W : array-like, shape (n_features, n_dimensions)
        Embedding vectors
    """
    
    if(not os.path.exists(embedding_file_path)):
        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), embedding_file_path)
    else:
        W = libpywmd.load_embeddings(embedding_file_path, vocabulary_file_path)
        np.save(vectors_file_path, W)
        return(W)

#######################################
# LOAD EMBEDDING VECTORS FROM NUMPY FILE 
def load_embedding_vectors(vectors_file_path="vectors.npy"):
    """ Loads embedding vectors from vectors_file_path 

    Parameters
    ----------
    
    vectors_file_path : string (default="vectors.npy")
        Indicates the relative file path location for the file "vectors.npy"
    
    Returns
    -------

    W : array-like, shape (n_features, n_dimensions)
        Embedding vectors
    """

    if(not os.path.exists(vectors_file_path)):
        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), vectors_file_path)
    else:
        W = np.load(vectors_file_path)
        return(W)
