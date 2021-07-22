// (c) Copyright IBM Corporation 2019, 2020, 2021
// Author: Kubilay Atasu

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <sys/time.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <assert.h>

#ifndef DOCSIM_H
#define DOCSIM_H

namespace docsim {

/*********************************************************************/ 

/* Data structure for storing sparse histograms */

typedef std::vector < std::pair < unsigned int, float > > sparse_vec;

/*********************************************************************/ 

int remd2_twoway_ground(int V_old, int V_new, float *D_ground, bool symmetric, int K, int delta_iterations,
			std::vector < sparse_vec > old_docs, std::vector < uint64_t > old_ids,
			std::vector < sparse_vec > new_docs, std::vector < uint64_t > new_ids, 
			float *D, bool export_results, const char *results_file, bool use_gpu);

int remd2_twoway_wrapper(int V, float *W, int M, bool symmetric, int K, int delta_iterations,
			std::vector < sparse_vec > old_docs, std::vector < uint64_t > old_ids,
			std::vector < sparse_vec > new_docs, std::vector < uint64_t > new_ids, 
			float *D, bool export_results, const char *results_file, bool use_gpu);

int rwmd2_twoway_wrapper(int V, float *W, int M, int K, int delta_iterations,
			std::vector < sparse_vec > old_docs, std::vector < uint64_t > old_ids,
			std::vector < sparse_vec > new_docs, std::vector < uint64_t > new_ids, 
			float *D, bool export_results, const char *results_file, bool use_gpu);

/* 
rwmd2_twoway_wrapper compares two sets of sparse histograms given the embedding vectors
remd2_twoway_ground compares two sets of sparse histograms given the ground distances
remd2_twoway_wrapper computes ground distances from embeddings and calls remd2_twoway_ground

Input Parameters

V: Size of the complete vocabulary for the given word embeddings
V_old: Size of the vocabulary used by the first set of documents
V_new: Size of the vocabulary used by the second set of documents
M: Size of the embedding vectors
delta_iterations: number of iterations to be performed
use_gpu: set true to use the GPU, otherwise the CPU will be used
symmetric: set true when using the same vocabulary for old and new sets

export_results: set true to export the top-K results into a file
K: number of results to be returned by Top-K calculation
results_file: a csv file storing the top-K results (uses doc ids) 

Input Data Structures

W: Embedding vectors for the vocabulary elements (VxM matrix of floats)
D_ground: Ground distances between vocabulary elements (V_old x V_new matrix of floats)

old_docs: first set of documents stored as sparse histograms
old_ids:  integer ids of the documents in the first set

new_docs: second set of documents stored as sparse histograms
new_ids:  integer ids of the documents in the second set

Output Data Structures

D: a num_docs_old x num_docs_new matrix of distances
In MPI mode, old_docs are distributed across p MPI processes. 
Process i receives size_i docs and produces size_i rows of D,
where (size_1 + size_2 + ... + size_p) = num_docs_old.

*/

/*********************************************************************/

int sort_cpu(int n, int m, float *D, int k, int *ind_out, float *val_out);

/* 
Computes top-k smallest distances in each row of an nxm distance matrix D

Outputs: 
val_out is an nxk matrix that stores the k-smallest values in each row
ind_out is an nxk matrix that stores the respective column indices 
*/

/*********************************************************************/ 

void cleanup_distribution_info();

/* Cleanup information related to distributed processing */

}

#endif
