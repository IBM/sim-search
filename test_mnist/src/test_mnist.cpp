// Copyright IBM Corporation 2019, 2020, 2021
// Author: Kubilay Atasu

#include "docsim.h"
#include "evaluate.h"
#include "parse.h"
#ifdef USE_MPI
#include "libmpi.h"
#endif

// ENVIRONMENT VARIABLES USED IN DISTRIBUTED EXECUTION
namespace docsim {
extern int myid, num_procs;
extern int local_id, num_local_procs;
extern int num_cpu_threads;
}

using namespace docsim;
using namespace std;

double When() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return( double(t.tv_sec) + double(t.tv_usec)/1000.0/1000.0 );
}

/*************************************************************************************************
NEAREST-NEIGHBORS SEARCH
*************************************************************************************************/

int main(int argc, char * argv[]) 
{
#ifdef USE_ESSL
	#ifdef USE_MPI
	num_cpu_threads = 1;
	#else
	num_cpu_threads = 20;
	#endif
#else
	num_cpu_threads = 16;
#endif

	const char *results_file = "topk_mnist";

	// ALGORITHM PARAMETERS 
	long long M; // DIMENSIONALITY OF THE EMBEDDING VECTORS
	long long V; // NUMBER OF WORDS IN THE VOCABULARY
	int K = 32; // COMPUTE TOP-K NEAREST NEIGHBORS

	// Main memory constraint
	const uint64_t max_memory_single_cpu = ((uint64_t) (1024*1024*1024))*((uint64_t) (40/4)); // 40 GB

	// INTIALIZE THE MPI ENVIRONMENT
	#ifdef USE_MPI
	init_mpi(&myid, &num_procs, &local_id, &num_local_procs);
	#else
	myid = 0; num_procs = 1; local_id = 0; num_local_procs = 1;
	#endif

	uint64_t max_memory_cpu =  ((uint64_t) num_procs) * max_memory_single_cpu;

	vector < sparse_vec > old_docs;
	vector < uint64_t > old_ids;
	vector < uint64_t > old_labels;
	int num_docs_old;
	int num_words_old;

	vector < sparse_vec > cur_docs;
	vector < uint64_t > cur_ids;
	vector < uint64_t > cur_labels;
	int num_docs_cur;
	int num_words_cur;

	vector < sparse_vec > new_docs;
	vector < uint64_t > new_ids;
	vector < uint64_t > new_labels;
	int num_docs_new;
	int num_words_new;

	double t0 = When();
	float *W = NULL;
	parse_docs_mnist("../../MNIST/train-images-idx3-ubyte", "../../MNIST/train-labels-idx1-ubyte", old_docs, old_ids, old_labels, num_docs_old, num_words_old, &W, M, V);
	cout << "Parsing training docs took " <<  When() - t0 << " secs" << endl;

	t0 = When();
	parse_docs_mnist("../../MNIST/t10k-images-idx3-ubyte", "../../MNIST/t10k-labels-idx1-ubyte", new_docs, new_ids, new_labels, num_docs_new, num_words_new, &W, M, V);
	cout << "Parsing test docs took " <<  When() - t0 << " secs" << endl;

	cout << "num_docs_old  = " << num_docs_old << endl;
	cout << "num_words_old = " << num_words_old << endl;
	cout << "num_docs_new  = " << num_docs_new << endl;
	cout << "num_words_new = " << num_words_new << endl;

	vector < vector < pair < uint32_t , float > > > topk;

	t0 = When();
	int i = 0;
	int p = 1;

	// RUN NEAREST-NEIGHBOR SEARCH IN BATCHES TO AVOID EXCEEDING THE MAIN MEMORY CONSTRAINT
	uint64_t max_docs_cur = max_memory_cpu / num_docs_old;
	while (i < num_docs_new) {
		double t1 = When();
		cur_docs.clear();
		cur_ids.clear();
		cur_labels.clear();
		num_docs_cur = 0;
		num_words_cur = 0;
		while ((i < num_docs_new) && (num_docs_cur < max_docs_cur)) {
			cur_docs.push_back(new_docs[i]);
			cur_ids.push_back(new_ids[i]);
			cur_labels.push_back(new_labels[i]);
			num_docs_cur++;
			num_words_cur += new_docs[i].size();
			i++;
		}
		char filename[512];
		sprintf(filename, "%s.part%d", results_file, p);
		if (myid==0) cout << "Comparing " <<  num_docs_cur << " new docs against " << num_docs_old << " old docs (" << num_words_old << " words)" << endl;

		float *D = new float [num_docs_cur*num_docs_old];
		// remd2_twoway_wrapper(V, W, M, false, K, 1, cur_docs, cur_labels, old_docs, old_labels, D, false, filename, true); // GPU
		// remd2_twoway_wrapper(V, W, M, false, K, 1, cur_docs, cur_labels, old_docs, old_labels, D, false, filename, false); // CPU
		rwmd2_twoway_wrapper(V, W, M, K, 1, cur_docs, cur_labels, old_docs, old_labels, D, false, filename, true); // GPU
		// rwmd2_twoway_wrapper(V, W, M, K, 1, cur_docs, cur_labels, old_docs, old_labels, D, false, filename, false); // CPU
		// cout << "Comparing " <<  num_docs_cur << " new docs against " << num_docs_old << " old docs (" << num_words_old << " words)" << " took "  << When() - t1 << " secs" << endl;

		if (D != NULL) {
			int *ind_out;
			float *val_out;
			ind_out = new int[num_docs_cur*K];
			val_out = new float[num_docs_cur*K];
			sort_cpu(num_docs_cur, num_docs_old, D, K, ind_out, val_out);
			evaluate_topk_precision(num_docs_cur, K, ind_out, &(cur_labels[0]), &(old_labels[0]), false, true);
			delete [] ind_out;
			delete [] val_out;
			delete [] D;
		}

		p++;
	}

	if (myid==0) cout << "Final: Comparing " <<  num_docs_new << " new docs against all " << num_docs_old << " old docs took "  << When() - t0 << " secs" << endl;

	cleanup_distribution_info();

	delete [] W; 

	#ifdef USE_MPI
	finalize_mpi();
	#endif

	return(0);
}
