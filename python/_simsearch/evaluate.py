# (c) Copyright IBM Corporation 2019, 2020, 2021

import numpy as np
from _simsearch import libpywmd

#######################################
# EVALUATE TOP-K SEARCH ACCURACY
def evaluate_topk(D, K, labels_X, labels_Y=None):
    """ Evaluates search accuracy for the top K samples

    Parameters
    ----------
    D : ndarray, shape (n_samples_x, n_samples_y)
        A two-dimensional distance matrix with distances between two datasets (X and Y)

    K : int
        Number of top samples for which we want to calculate the accuracy of the similarity search algorithm

    labels_X : array-like, shape (n_samples_x,)
        Labels corresponding to dataset X

    labels_Y : array-like, shape (n_samples_y,)
        Labels corresponding to dataset Y
    
    Notes
    -----
    labels_Y = None if we want to evaluate the precision of the similarity search algorithm for documents/images within a 
        single dataset    
 
    Returns
    -------
    k_vec : ndarray, shape (log(K)+1,) 
        Indicates the top-K number for which we are calculating the precision values (in powers of 2: 1,2,4,8,16,32 if the value of input argument K is 32)

    prec_vec : ndarray, shape (log(K)+1,)
        Indicates the corresponding precision values 
   
    topk_indices : ndarray, shape (n_samples_x, K)
        Indicates the number K for which we store the precision values 
  
    topk_values : ndarray, shape (n_samples_x, K)
        Indicates the precision values corresponding to the topk_indices
    
    """

    if (labels_Y is None):
        # comparing with the same dataset: exclude top-1 results from the precision evaluation
        K = K+1

    # make sure k is not larger than n_samples_y
    k = min(D.shape[1], K)

    # allocate some of the output vectors
    topk_indices = np.zeros((D.shape[0],k), np.int32)
    topk_values = np.zeros((D.shape[0],k), np.float32)

    if (labels_Y is None):
        # comparing with the same dataset: exclude top-1 results from the precision evaluation
        k_vec, prec_vec = libpywmd.compute_topk(D, topk_indices, topk_values, labels_X, labels_X, 1)
    else:
        # comparing with a different dataset: include top-1 results in the precision evaluation
        k_vec, prec_vec = libpywmd.compute_topk(D, topk_indices, topk_values, labels_X, labels_Y, 0)

    return (k_vec, prec_vec, topk_indices, topk_values)
