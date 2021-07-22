// (c) Copyright IBM Corporation 2019, 2020, 2021
// Author: Kubilay Atasu

#ifndef LIBWMD_H
#define LIBWMD_H

namespace docsim {

// GPU functions (libwmd.cu)
int copy_vocabulary(int device_id, int voc_size_, int vec_size_, float *W, bool second);
int remove_vocabulary(bool second);
int copy_raw_data(int n, int *ind, float *weight, int num_docs, int *start);
int remove_raw_data();
int copy_raw_data2(int n2, int *ind2, float *weight2, int num_docs2, int *start2);
int remove_raw_data2();
int copy_vectors2(int n2, int m, float * X2, float *weights2, int num_docs2_, int * start2_);
int remove_vectors2();
int set_device_id(int device_id);
int ed_min_vocab_multi(float * Z, int delta_iterations, bool precompute, bool symmetric, bool second);

int compute_ground_distances(float * D_ground, bool symmetric);
int copy_ground_distances(int voc_size_, int voc_size2_, float * D_ground, bool symmetric);
int remove_ground_distances();

// CPU functions (libwmd.cpp)
int copy_vocabulary_cpu(int & voc_size_, int vec_size_, float *W_, bool second);
int remove_vocabulary_cpu(bool second);
int copy_raw_data_cpu(int n_, int * ind_, float * weight_, int num_docs_, int * start_);
int remove_raw_data_cpu();
int copy_raw_data2_cpu(int n2_, int * ind2_, float * weight2_, int num_docs2_, int * start2_);
int remove_raw_data2_cpu();
int copy_vectors2_cpu(int n2_, int m, float * X2_, float *weight2_, int num_docs2_, int * start2_);
int remove_vectors2_cpu();
int ed_min_vocab_cpu_multi(float * Z, int delta_iterations, bool precompute, bool symmetric, bool second); 

int compute_ground_distances_cpu(float * D_ground, bool symmetric);
int copy_ground_distances_cpu(int voc_size_, int voc_size2_, float * D_ground, bool symmetric);
int remove_ground_distances_cpu();

}

#endif
