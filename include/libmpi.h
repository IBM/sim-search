// (c) Copyright IBM Corporation 2019, 2020, 2021
// Author: Kubilay Atasu

#ifndef LIBMPI_H
#define LIBMPI_H

namespace docsim {

void init_mpi(int *myid_p, int *num_procs_p, int *local_id_p, int *num_local_procs_p);
void finalize_mpi();
void barrier_mpi();
void broadcast_mpi(void *buffer, int byte_count, int root);
void all_gather_mpi(int *data);
void reduce_mpi_min_float(void* send_data, void* recv_data, int count);
void reduce_mpi_sum_int(void* send_data, void* recv_data, int count);
float compute_kth_parallel(int k, float *val);
void cleanup_distribution_info();
void sort_cpu_parallel(int n, int m, int k, float * D, int *ind_out, float * val_out);
void gather_distances_mpi(float *D, int nrows, int ncols);

}

#endif
