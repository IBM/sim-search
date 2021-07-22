// (c) Copyright IBM Corporation 2019, 2020, 2021
// Author: Kubilay Atasu

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <exception>
#include <math.h>
#include "libwmd.h"
#include "docsim.h"
#include "parse.h"
#include "evaluate.h"
#ifdef USE_MPI
#include "libmpi.h"
#endif

struct module_state {
    PyObject *type_error;
    PyObject *io_error;
    PyObject *other_error;
};

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define GET_MODULE_STATE(m) ((struct module_state*)PyModule_GetState(m))
#define INITERROR return NULL
#else
#define GET_MODULE_STATE(m) (&_state)
static struct module_state _state;
#define INITERROR return
#endif

namespace docsim {
// GLOBAL VARIABLES USED IN DISTRIBUTED EXECUTION
extern int myid, num_procs;
extern int local_id, num_local_procs;
// GLOBAL VARIABLES USED TO CONFIGURE ALGORITHMS
extern int num_cpu_threads;
extern bool use_cosine;
extern bool normalize_embeddings;
}

#include <iostream>
using namespace std;
using namespace docsim;

static PyObject * pywmd_load_embeddings(PyObject *self, PyObject *args) {
	char *embedding_file_path;
	char *vocabulary_file_path;

	float *W;
	long long M, V;
	map <string, int > vocab;
	
	char message[512];

	if (!PyArg_ParseTuple(args, "ss", &embedding_file_path, &vocabulary_file_path)) {
		strcpy(message, "Arguments are not valid");
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	if (!parse_embedding_file(embedding_file_path, true, vocabulary_file_path, vocab, &W, M, V)) {
		// strcpy(message, "IOError: Failed to parse the embedding file and to create the vocabulary file");
		// struct module_state *st = GET_MODULE_STATE(self);
		// PyErr_SetString(st->io_error, message);
		// cout << message << endl;
		return PyInt_FromLong(-1);
	}

	// cout << "Parsed " << embedding_file_path << " and created "<< vocabulary_file_path << endl;

	npy_intp dimensions[2]; dimensions[0] = V; dimensions[1] = M;
	PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNewFromData(2, dimensions, NPY_FLOAT32, (char *) W);
	return PyArray_Return(result);
}

static PyObject * pywmd_load_20News(PyObject *self, PyObject *args) {

	char *directory_path;
	char *vocabulary_file_path;

	char message[512];

	npy_int64  py_max_histogram_size;
	npy_int64  py_stop_word_threshold;

	if (!PyArg_ParseTuple(args,"ssll", &directory_path, &vocabulary_file_path, &py_max_histogram_size, &py_stop_word_threshold)) {
		strcpy(message, "Arguments are not valid");
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	map <string, int > vocab;
	long long V = parse_vocabulary(vocabulary_file_path, vocab);
	if (V == 0) {
		// strcpy(message, "IOError: Failed to parse the vocabulary file");
		// struct module_state *st = GET_MODULE_STATE(self);
		// PyErr_SetString(st->io_error, message);
		// cout << message << endl;
		npy_int64 t = -1;
		return Py_BuildValue("llllllll", t, t, t, t, t, t, t, t);
	}

	vector < sparse_vec > docs;
	vector < uint64_t > ids;
	vector < uint64_t > labels;
	int num_docs;
	int num_words;

	if (!parse_docs_20(directory_path, py_max_histogram_size, py_stop_word_threshold, docs, ids, labels, num_docs, num_words, vocab)) {
		// strcpy(message, "IOError: Found no text files in the directory path");
		// struct module_state *st = GET_MODULE_STATE(self);
		// PyErr_SetString(st->io_error, message);
		// cout << message << endl;
		npy_int64 t = -1;
		return Py_BuildValue("llllllll", t, t, t, t, t, t, t, t);
	}

	// cout << "Parsed " << directory_path << " and " << vocabulary_file_path << endl;

	npy_int64 py_V = V;
	npy_int64 py_nrow = num_docs;
	npy_int64 py_nnz = num_words;

	npy_intp dimensions[1]; dimensions[0] = num_docs+1;
	PyArrayObject  *py_indptr = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_INT32);
	int32_t *indptr = (int32_t *) PyArray_DATA(py_indptr);

	dimensions[0] = num_words;
	PyArrayObject  *py_indices = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_INT32);
	int32_t *indices = (int32_t *) PyArray_DATA(py_indices);

	dimensions[0] = num_words;
	PyArrayObject  *py_data = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_FLOAT32);
	float *data = (float *) PyArray_DATA(py_data);

	dimensions[0] = num_docs;
	PyArrayObject  *py_labels = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_UINT64);
	uint64_t *labels_vec = (uint64_t *) PyArray_DATA(py_labels);
	for (uint32_t j=0; j<num_docs; j++) labels_vec[j] = labels[j];

	PyArrayObject  *py_ids = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_UINT64);
	uint64_t *ids_vec = (uint64_t *) PyArray_DATA(py_ids);
	for (uint32_t j=0; j<num_docs; j++) ids_vec[j] = ids[j];

	int32_t cur_idx = 0;
	for (uint32_t j=0; j<num_docs; j++) {
		indptr[j] = cur_idx;
		for(uint32_t i = 0; i < docs[j].size(); i++) {
			indices[cur_idx] = docs[j][i].first;
			data[cur_idx] = docs[j][i].second;
			cur_idx++;
		}
	}
	indptr[num_docs] = num_words;

	return Py_BuildValue("lllOOOOO", py_V, py_nrow, py_nnz, py_indptr, py_indices, py_data, py_labels, py_ids);
}

static PyObject * pywmd_load_mnist_with_embeddings(PyObject *self, PyObject *args) {

	char *image_file_path;
	char *label_file_path;

	char message[512];

	if (!PyArg_ParseTuple(args,"ss", &image_file_path, &label_file_path)) {
		strcpy(message, "Arguments are not valid");
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	vector < sparse_vec > docs;
	vector < uint64_t > ids;
	vector < uint64_t > labels;
	int num_docs;
	int num_words;

	long long M; // DIMENSIONALITY OF THE EMBEDDING VECTORS
	long long V; // NUMBER OF WORDS IN THE VOCABULARY
	float *W = NULL;

	if (!parse_docs_mnist(image_file_path, label_file_path, docs, ids, labels, num_docs, num_words, &W, M, V)) {
		// strcpy(message, "IOError: Failed to parse MNIST files");
		// struct module_state *st = GET_MODULE_STATE(self);
		// PyErr_SetString(st->io_error, message);
		// cout << message << endl;
		npy_int64 t = -1;
		return Py_BuildValue("llllllll", t, t, t, t, t, t, t, t);
	}

	// cout << "Parsed " << image_file_path << " and " << label_file_path << endl;

	// uint64_t *nnz = (uint64_t *) PyArray_DATA(py_nnz);
	// *nnz = num_words;

	npy_int64 py_nrow = num_docs;
	npy_int64 py_nnz = num_words;

	npy_intp dimensions[1]; dimensions[0] = num_docs+1;
	PyArrayObject  *py_indptr = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_INT32);
	int32_t *indptr = (int32_t *) PyArray_DATA(py_indptr);

	dimensions[0] = num_words;
	PyArrayObject  *py_indices = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_INT32);
	int32_t *indices = (int32_t *) PyArray_DATA(py_indices);

	dimensions[0] = num_words;
	PyArrayObject  *py_data = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_FLOAT32);
	float *data = (float *) PyArray_DATA(py_data);

	dimensions[0] = num_docs;
	PyArrayObject  *py_labels = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_UINT64);
	uint64_t *labels_vec = (uint64_t *) PyArray_DATA(py_labels);
	for (uint32_t j=0; j<num_docs; j++) labels_vec[j] = labels[j];

	PyArrayObject  *py_ids = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_UINT64);
	uint64_t *ids_vec = (uint64_t *) PyArray_DATA(py_ids);
	for (uint32_t j=0; j<num_docs; j++) ids_vec[j] = ids[j];

	int32_t cur_idx = 0;
	for (uint32_t j=0; j<num_docs; j++) {
		indptr[j] = cur_idx;
		for(uint32_t i = 0; i < docs[j].size(); i++) {
			indices[cur_idx] = docs[j][i].first;
			data[cur_idx] = docs[j][i].second;
			cur_idx++;
		}
	}
	indptr[num_docs] = num_words;

	npy_intp dimensions2[2]; dimensions2[0] = V; dimensions2[1] = M;
	PyArrayObject *py_embeddings = (PyArrayObject *) PyArray_SimpleNewFromData(2, dimensions2, NPY_FLOAT32, (char *) W);

	return Py_BuildValue("OllOOOOO", py_embeddings, py_nrow, py_nnz, py_indptr, py_indices, py_data, py_labels, py_ids);
}

static PyObject * pywmd_compute_ground_distances(PyObject *self, PyObject *args) {
	PyArrayObject  *py_D_ground;
	PyArrayObject  *py_W;
	PyArrayObject  *py_W2;
	npy_int64       py_use_gpu;

	float * D_ground = NULL;
	float * W_arr = NULL;
	float * W2_arr = NULL;
	int v = 0;
	int v2 = 0;
	int m = 0;

	char message[512];
	bool report_error = false;

	npy_int64 py_use_cosine;

	#if !(defined (USE_ESSL) || defined (USE_MKL) || defined (USE_EIGEN) || defined (USE_CUDA))
	cout << "!!No matrix library has been found, skipping distance calculation!!" << endl;
	return PyInt_FromLong(-1);
	#endif

	bool symmetric = false;
	if (!PyArg_ParseTuple(args,"OOOll", &py_D_ground, &py_W, &py_W2, &py_use_gpu, &py_use_cosine)) {
		strcpy(message, "Arguments are not valid");
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	bool use_gpu = py_use_gpu;
	use_cosine = py_use_cosine;
	if (use_cosine) {
		normalize_embeddings = true;
	}

	if (PyArray_NDIM(py_D_ground) != 2) {
		strcpy(message, "D_ground must have two dimensions.");
		report_error = true;
	} else { 
		npy_intp * D_shape = PyArray_DIMS(py_D_ground);
		if (PyArray_TYPE(py_D_ground) == NPY_FLOAT32) {
			D_ground  = (float*)PyArray_DATA(py_D_ground);
		} else {
			strcpy(message, "The elements of D_ground have the wrong type. Expected type: float32.");
			report_error = true;
		}
	}
	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	int ndim = PyArray_NDIM(py_W);
	if (ndim != 2) {
		strcpy(message, "W must have two dimensions.");
		report_error = true;
	} else { 
		npy_intp * W_shape = PyArray_DIMS(py_W);
		v = W_shape[0];
		m = W_shape[1];
		if ((v >0) && (m>0)) {
			if (PyArray_TYPE(py_W) == NPY_FLOAT32) {
				W_arr  = (float*)PyArray_DATA(py_W);
			} else {
				strcpy(message, "The elements of W have the wrong type. Expected type: float32.");
				report_error = true;
			}
		} else {
			strcpy(message, "W does not have two dimensions.");
			report_error = true;
		}
	}
	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	int ndim2 = PyArray_NDIM(py_W2);
	if (ndim2 != 2) {
		symmetric = true;
	} else { 
		npy_intp * W2_shape = PyArray_DIMS(py_W2);
		v2 = W2_shape[0];
		int m2 = W2_shape[1];
		if ((v2 >0) && (m2>0)) {
			if (PyArray_TYPE(py_W2) == NPY_FLOAT32) {
				W2_arr  = (float*)PyArray_DATA(py_W2);
			} else {
				strcpy(message, "The elements of W2 have the wrong type. Expected type: float32.");
				report_error = true;
			}
			if (m != m2) {
				strcpy(message, "The dimensions of W and W2 are not compatible.");
				report_error = true;
			}
		} else {
			strcpy(message, "W2 does not have two dimensions.");
			report_error = true;
		}
	}
	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	if (use_gpu) {
		#if (defined (USE_CUDA)) 
		// Copy vocabulary to GPU
		copy_vocabulary(local_id, v, m, W_arr, false);
		if (!symmetric) {
			copy_vocabulary(local_id, v2, m, W2_arr, true);
		}
		#endif
	} else {
		#if (defined (USE_ESSL) || defined (USE_MKL) || defined (USE_EIGEN)) 
		copy_vocabulary_cpu(v, m, W_arr, false);	
		if (!symmetric) {
			copy_vocabulary_cpu(v2, m, W2_arr, true);
		}
		#endif
	}

	if (symmetric) v2 = v;

	if (use_gpu) {
		#if (defined (USE_CUDA)) 
		compute_ground_distances(D_ground, symmetric);
		#endif
	} else {
		#if (defined (USE_ESSL) || defined (USE_MKL) || defined (USE_EIGEN)) 
		compute_ground_distances_cpu(D_ground, symmetric);
		#endif
	}

	if (use_gpu) {
		#if (defined (USE_CUDA)) 
		remove_vocabulary(false);
		if (!symmetric) {
			remove_vocabulary(true);
		}
		#endif
	} else {
		#if (defined (USE_ESSL) || defined (USE_MKL) || defined (USE_EIGEN)) 
		remove_vocabulary_cpu(false);
		if (!symmetric) {
			remove_vocabulary_cpu(true);
		}
		#endif
	}

	return PyInt_FromLong(-1);
}

static PyObject * pywmd_emd_wrapper(PyObject *self, PyObject *args) {
	PyArrayObject  *py_D;
	PyArrayObject  *py_D_ground;

	npy_int64       py_nnzx;
	PyArrayObject  *py_indptrx;
	PyArrayObject  *py_indicesx;
	PyArrayObject  *py_datax;

	npy_int64       py_nnzy;
	PyArrayObject  *py_indptry;
	PyArrayObject  *py_indicesy;
	PyArrayObject  *py_datay;

	npy_int64       py_use_gpu;
	npy_int64       py_optimization_level;
	npy_int64       py_num_cpu_threads;
	npy_int64       py_symmetric;

	float * D = NULL;
	float * D_ground = NULL;
	int v = 0;
	int v2 = 0;
	int nx = 0;
	int ny = 0;

	int32_t *indptrx_arr = NULL;
	int32_t *indptry_arr = NULL;
	int32_t *indicesx_arr = NULL;
	int32_t *indicesy_arr = NULL;
	float *datax_arr = NULL;
	float *datay_arr = NULL;

	char message[512];
	bool report_error = false;

	#if !(defined (USE_ESSL) || defined (USE_MKL) || defined (USE_EIGEN) || defined (USE_CUDA))
	cout << "!!No matrix library has been found, skipping distance calculation!!" << endl;
	return PyInt_FromLong(-1);
	#endif

	if (!PyArg_ParseTuple(args,"OOlOOOlOOOllll", &py_D, &py_D_ground, &py_nnzx, &py_indptrx, &py_indicesx, &py_datax,  &py_nnzy, &py_indptry, &py_indicesy, &py_datay, &py_use_gpu, &py_optimization_level, &py_num_cpu_threads, &py_symmetric)) {
		strcpy(message, "Arguments are not valid");
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	int delta_iterations = pow(2,py_optimization_level) -1;
	delta_iterations = max(0, delta_iterations); // can't be smaller than 0
	delta_iterations = min(15, delta_iterations); // can't be larger than 15
	// cout << "Performing " << delta_iterations << " iterations" << endl;	

	bool use_gpu = py_use_gpu;

	num_cpu_threads = py_num_cpu_threads;

	bool symmetric = py_symmetric;

	if (PyArray_NDIM(py_D) != 2) {
		strcpy(message, "D must have two dimensions.");
		report_error = true;
	} else { 
		npy_intp * D_shape = PyArray_DIMS(py_D);
		if ((D_shape[0]>0) && (D_shape[1]>0)) {
			if (PyArray_TYPE(py_D) == NPY_FLOAT32) {
				D  = (float*)PyArray_DATA(py_D);
			} else {
				strcpy(message, "The elements of D have the wrong type. Expected type: float32.");
				report_error = true;
			}
		} else {
			strcpy(message, "D does not have two dimensions.");
			report_error = true;
		}
	}
	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	int ndim = PyArray_NDIM(py_D_ground);
	if (ndim != 2) {
		strcpy(message, "D_ground must have two dimensions.");
		report_error = true;
	} else { 
		npy_intp * D_shape = PyArray_DIMS(py_D_ground);
		v = D_shape[0];
		v2 = D_shape[1];
		if (PyArray_TYPE(py_D_ground) == NPY_FLOAT32) {
			D_ground  = (float*)PyArray_DATA(py_D_ground);
			if ((symmetric) && (v != v2)) {
				strcpy(message, "D_ground is not a symmetric matrix. Expected: symmetric.");
				report_error = true;
			}
		} else {
			strcpy(message, "The elements of D_ground have the wrong type. Expected type: float32.");
			report_error = true;
		}
	}
	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	if (symmetric) v2 = v;

	npy_intp validx = PyArray_SIZE(py_indptrx);
	if (validx == 0) {
		strcpy(message, "Size of indptr is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_indptrx) != NPY_INT32) {
		strcpy(message, "The elements of indptr have the wrong type. Expected type: int32.");
		report_error = true;
	} else {
		indptrx_arr = (int32_t *) PyArray_DATA(py_indptrx);
		nx = validx-1;
	}

	npy_intp validy = PyArray_SIZE(py_indptry);
	if (validy == 0) {
		strcpy(message, "Size of indptr is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_indptry) != NPY_INT32) {
		strcpy(message, "The elements of indptr have the wrong type. Expected type: int32.");
		report_error = true;
	} else {
		indptry_arr = (int32_t *) PyArray_DATA(py_indptry);
		ny = validy-1;
	}

	validx = PyArray_SIZE(py_indicesx);
	if (validx == 0) {
		strcpy(message, "Size of py_indices is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_indicesx) != NPY_INT32) {
		strcpy(message, "The elements of py_indices have the wrong type. Expected type: int32.");
		report_error = true;
	} else {
		indicesx_arr = (int32_t *) PyArray_DATA(py_indicesx);
	}

	validy = PyArray_SIZE(py_indicesy);
	if (validy == 0) {
		strcpy(message, "Size of py_indices is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_indicesy) != NPY_INT32) {
		strcpy(message, "The elements of py_indices have the wrong type. Expected type: int32.");
		report_error = true;
	} else {
		indicesy_arr = (int32_t *) PyArray_DATA(py_indicesy);
	}

	validx = PyArray_SIZE(py_datax);
	if (validx == 0) {
		strcpy(message, "Size of py_data is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_datax) != NPY_FLOAT32) {
		strcpy(message, "The elements of py_data have the wrong type. Expected type: float32.");
		report_error = true;
	} else {
		datax_arr = (float *) PyArray_DATA(py_datax);
	}

	validy = PyArray_SIZE(py_datay);
	if (validy == 0) {
		strcpy(message, "Size of py_data is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_datay) != NPY_FLOAT32) {
		strcpy(message, "The elements of py_data have the wrong type. Expected type: float32.");
		report_error = true;
	} else {
		datay_arr = (float *) PyArray_DATA(py_datay);
	}

	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	std::vector < sparse_vec > new_docs;
	std::vector < uint64_t > new_ids;
	for (int i=0; i<nx; i++) {
		sparse_vec doc;
		for (int32_t j=indptrx_arr[i]; j<indptrx_arr[i+1]; j++) {
			std::pair<unsigned int, float> cur_pair(indicesx_arr[j], datax_arr[j]);
                	doc.push_back(cur_pair);
		}
		new_docs.push_back(doc);
		new_ids.push_back(0);
	}

	std::vector < sparse_vec > old_docs;
	std::vector < uint64_t > old_ids;
	for (int i=0; i<ny; i++) {
		sparse_vec doc;
		for (int32_t j=indptry_arr[i]; j<indptry_arr[i+1]; j++) {
			std::pair<unsigned int, float> cur_pair(indicesy_arr[j], datay_arr[j]);
                	doc.push_back(cur_pair);
		}
		old_docs.push_back(doc);
		old_ids.push_back(0);
	}

	cleanup_distribution_info();

	remd2_twoway_ground(v, v2, D_ground, symmetric, 1, delta_iterations, new_docs, new_ids, old_docs, old_ids, D, false, "./tmp", use_gpu);

	return PyInt_FromLong(-1);
}

static PyObject * pywmd_wmd_wrapper(PyObject *self, PyObject *args) {
	PyArrayObject  *py_D;
	PyArrayObject  *py_W;

	npy_int64       py_nnzx;
	PyArrayObject  *py_indptrx;
	PyArrayObject  *py_indicesx;
	PyArrayObject  *py_datax;

	npy_int64       py_nnzy;
	PyArrayObject  *py_indptry;
	PyArrayObject  *py_indicesy;
	PyArrayObject  *py_datay;

	npy_int64       py_use_gpu;
	npy_int64       py_optimization_level;
	npy_int64       py_num_cpu_threads;
	npy_int64       py_use_cosine;

	float * D = NULL;
	float * W_arr = NULL;
	int v = 0;
	int m = 0;
	int nx = 0;
	int ny = 0;

	int32_t *indptrx_arr = NULL;
	int32_t *indptry_arr = NULL;
	int32_t *indicesx_arr = NULL;
	int32_t *indicesy_arr = NULL;
	float *datax_arr = NULL;
	float *datay_arr = NULL;

	char message[512];
	bool report_error = false;

	#if !(defined (USE_ESSL) || defined (USE_MKL) || defined (USE_EIGEN) || defined (USE_CUDA))
	cout << "!!No matrix library has been found, skipping distance calculation!!" << endl;
	return PyInt_FromLong(-1);
	#endif

	if (!PyArg_ParseTuple(args,"OOlOOOlOOOllll", &py_D, &py_W, &py_nnzx, &py_indptrx, &py_indicesx, &py_datax, &py_nnzy, &py_indptry, &py_indicesy, &py_datay, &py_use_gpu, &py_optimization_level, &py_num_cpu_threads, &py_use_cosine)) {
		strcpy(message, "Arguments are not valid");
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	int delta_iterations = pow(2,py_optimization_level) -1;
	delta_iterations = max(0, delta_iterations); // can't be smaller than 0
	delta_iterations = min(15, delta_iterations); // can't be larger than 15
	// cout << "Performing " << delta_iterations << " iterations" << endl;	

	bool use_gpu = py_use_gpu;

	num_cpu_threads = py_num_cpu_threads;
	use_cosine = py_use_cosine;
	if (use_cosine) {
		normalize_embeddings = true;
	}

	if (PyArray_NDIM(py_D) != 2) {
		strcpy(message, "D must have two dimensions.");
		report_error = true;
	} else { 
		npy_intp * D_shape = PyArray_DIMS(py_D);
		if ((D_shape[0]>0) && (D_shape[1]>0)) {
			if (PyArray_TYPE(py_D) == NPY_FLOAT32) {
				D  = (float*)PyArray_DATA(py_D);
			} else {
				strcpy(message, "The elements of D have the wrong type. Expected type: float32.");
				report_error = true;
			}
		} else {
			strcpy(message, "D does not have two dimensions.");
			report_error = true;
		}
	}
	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	int ndim = PyArray_NDIM(py_W);
	if (ndim != 2) {
		strcpy(message, "W must have two dimensions.");
		report_error = true;
	} else { 
		npy_intp * W_shape = PyArray_DIMS(py_W);
		v = W_shape[0];
		m = W_shape[1];
		if ((v >0) && (m>0)) {
			if (PyArray_TYPE(py_W) == NPY_FLOAT32) {
				W_arr  = (float*)PyArray_DATA(py_W);
			} else {
				strcpy(message, "The elements of W have the wrong type. Expected type: float32.");
				report_error = true;
			}
		} else {
			strcpy(message, "W does not have two dimensions.");
			report_error = true;
		}
	}
	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	npy_intp validx = PyArray_SIZE(py_indptrx);
	if (validx == 0) {
		strcpy(message, "Size of indptr is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_indptrx) != NPY_INT32) {
		strcpy(message, "The elements of indptr have the wrong type. Expected type: int32.");
		report_error = true;
	} else {
		indptrx_arr = (int32_t *) PyArray_DATA(py_indptrx);
		nx = validx-1;
	}

	npy_intp validy = PyArray_SIZE(py_indptry);
	if (validy == 0) {
		strcpy(message, "Size of indptr is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_indptry) != NPY_INT32) {
		strcpy(message, "The elements of indptr have the wrong type. Expected type: int32.");
		report_error = true;
	} else {
		indptry_arr = (int32_t *) PyArray_DATA(py_indptry);
		ny = validy-1;
	}

	validx = PyArray_SIZE(py_indicesx);
	if (validx == 0) {
		strcpy(message, "Size of py_indices is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_indicesx) != NPY_INT32) {
		strcpy(message, "The elements of py_indices have the wrong type. Expected type: int32.");
		report_error = true;
	} else {
		indicesx_arr = (int32_t *) PyArray_DATA(py_indicesx);
	}

	validy = PyArray_SIZE(py_indicesy);
	if (validy == 0) {
		strcpy(message, "Size of py_indices is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_indicesy) != NPY_INT32) {
		strcpy(message, "The elements of py_indices have the wrong type. Expected type: int32.");
		report_error = true;
	} else {
		indicesy_arr = (int32_t *) PyArray_DATA(py_indicesy);
	}

	validx = PyArray_SIZE(py_datax);
	if (validx == 0) {
		strcpy(message, "Size of py_data is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_datax) != NPY_FLOAT32) {
		strcpy(message, "The elements of py_data have the wrong type. Expected type: float32.");
		report_error = true;
	} else {
		datax_arr = (float *) PyArray_DATA(py_datax);
	}

	validy = PyArray_SIZE(py_datay);
	if (validy == 0) {
		strcpy(message, "Size of py_data is zero.");
		report_error = true;
	} else if (PyArray_TYPE(py_datay) != NPY_FLOAT32) {
		strcpy(message, "The elements of py_data have the wrong type. Expected type: float32.");
		report_error = true;
	} else {
		datay_arr = (float *) PyArray_DATA(py_datay);
	}

	if (report_error) {
		struct module_state *st = GET_MODULE_STATE(self);
		PyErr_SetString(st->type_error, message);
		return PyInt_FromLong(-1);
	}

	std::vector < sparse_vec > new_docs;
	std::vector < uint64_t > new_ids;
	for (int i=0; i<nx; i++) {
		sparse_vec doc;
		for (int32_t j=indptrx_arr[i]; j<indptrx_arr[i+1]; j++) {
			std::pair<unsigned int, float> cur_pair(indicesx_arr[j], datax_arr[j]);
                	doc.push_back(cur_pair);
		}
		new_docs.push_back(doc);
		new_ids.push_back(0);
	}

	std::vector < sparse_vec > old_docs;
	std::vector < uint64_t > old_ids;
	for (int i=0; i<ny; i++) {
		sparse_vec doc;
		for (int32_t j=indptry_arr[i]; j<indptry_arr[i+1]; j++) {
			std::pair<unsigned int, float> cur_pair(indicesy_arr[j], datay_arr[j]);
                	doc.push_back(cur_pair);
		}
		old_docs.push_back(doc);
		old_ids.push_back(0);
	}

	cleanup_distribution_info();

	rwmd2_twoway_wrapper(v, W_arr, m, 1, delta_iterations, new_docs, new_ids, old_docs, old_ids, D, false, "./tmp", use_gpu);

	return PyInt_FromLong(-1);
}

static PyObject * pywmd_compute_topk(PyObject *self, PyObject *args) {

	PyArrayObject * D_py_arr;
	PyArrayObject * Y_py_arr;
	PyArrayObject * Z_py_arr;
	PyArrayObject * l_test_py_arr;
	PyArrayObject * l_train_py_arr;
	npy_int64       py_exclude_first;

	if(!PyArg_ParseTuple(args,"OOOOOl", &D_py_arr, &Y_py_arr, &Z_py_arr, &l_test_py_arr, &l_train_py_arr, &py_exclude_first)) return NULL;	

	bool exclude_first = py_exclude_first;

	npy_intp * D_py_shape = PyArray_DIMS(D_py_arr);
	int n = D_py_shape[0];
	int m = D_py_shape[1];

	npy_intp * Y_py_shape = PyArray_DIMS(Y_py_arr);
	int k = Y_py_shape[1];

	float * D       = (float*)PyArray_DATA(D_py_arr);
	int   * ind_out = (int*)PyArray_DATA(Y_py_arr);
	float * val_out = (float*)PyArray_DATA(Z_py_arr);

	uint64_t * labels_test = (uint64_t *) PyArray_DATA(l_test_py_arr);
	uint64_t * labels_train = (uint64_t *) PyArray_DATA(l_train_py_arr);

	// cout << "Sorting the distances" << endl;
	// cout << n << " " <<  m << endl;
	// cout << Y_py_shape[0] << " " << Y_py_shape[1] << endl;

	int status = sort_cpu(n, m, D, k, ind_out, val_out);

	int num_test = n;

	vector <float> precision_vector = evaluate_topk_precision(num_test, k, ind_out, labels_test, labels_train, exclude_first, false);

	npy_intp dimensions[1]; dimensions[0] = precision_vector.size();

	PyArrayObject  *py_prec_vec = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_FLOAT32);
	float *prec_vec = (float *) PyArray_DATA(py_prec_vec);
	for (int i=0; i<precision_vector.size(); i++) prec_vec[i] = precision_vector[i]; 

	PyArrayObject  *py_k_vec = (PyArrayObject *)PyArray_SimpleNew(1, dimensions, NPY_INT32);
	int *k_vec = (int *) PyArray_DATA(py_k_vec);
	for (int i=0; i<precision_vector.size(); i++) k_vec[i] = pow(2, i);

	if (exclude_first)
		k_vec[precision_vector.size()-1] = k-1;
	else
		k_vec[precision_vector.size()-1] = k;

	return Py_BuildValue("OO", py_k_vec, py_prec_vec);
}

void python_cleanup() {
	cleanup_distribution_info();
	#ifdef USE_MPI
	finalize_mpi();
	#endif
}

static PyMethodDef pywmdMethods[] = {
	// INTERFACE FUNCTIONS
	{"load_mnist_with_embeddings", pywmd_load_mnist_with_embeddings, METH_VARARGS, "Load MNIST dataset and embeddings"}, 

	{"load_20News", pywmd_load_20News, METH_VARARGS, "Load 20 Newsgroups dataset"}, 

	{"load_embeddings", pywmd_load_embeddings, METH_VARARGS, "Load pre-trained embeddings"}, 

	{"wmd_wrapper", pywmd_wmd_wrapper,  METH_VARARGS, "Python interface for WMD"},

	{"compute_topk", pywmd_compute_topk,  METH_VARARGS, "Find top-k most similar documents in training data (on CPU)"},

	{"emd_wrapper", pywmd_emd_wrapper,  METH_VARARGS, "Python interface for EMD"},

	{"compute_ground_distances", pywmd_compute_ground_distances,  METH_VARARGS, "Compute ground distances between members of the vocabulary"},

	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int mytraverse(PyObject *m, visitproc visit, void *arg) {
	Py_VISIT(GET_MODULE_STATE(m)->type_error);
	Py_VISIT(GET_MODULE_STATE(m)->io_error);
	Py_VISIT(GET_MODULE_STATE(m)->other_error);
	return 0;
}

static int myclear(PyObject *m) {
	Py_CLEAR(GET_MODULE_STATE(m)->type_error);
	Py_CLEAR(GET_MODULE_STATE(m)->io_error);
	Py_CLEAR(GET_MODULE_STATE(m)->other_error);
	return 0;
}

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	#ifdef USE_MPI
        "pywmdmpi",                        /* m_name */
	#else
        "pywmd",                        /* m_name */
	#endif
	"This is a module",  		/* m_doc */
	sizeof(struct module_state),	/* m_size */
	pywmdMethods,        		/* m_methods */
	NULL,                		/* m_reload */
	mytraverse,                	/* m_traverse */
	myclear,                	/* m_clear */
	NULL,                		/* m_free */
};
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
	#ifdef USE_MPI
    	PyInit_libpywmdmpi(void)
	#else
    	PyInit_libpywmd(void)
	#endif
#else
	#ifdef USE_MPI
	initlibpywmdmpi(void)
	#else
	initlibpywmd(void)
	#endif
#endif
{
	//cout << "Initializing module libpywmd " << endl;

#if PY_MAJOR_VERSION >= 3
	PyObject *module = PyModule_Create(&moduledef);
#else
	#ifdef USE_MPI
	PyObject *module = Py_InitModule("libpywmdmpi", pywmdMethods);
	#else
	PyObject *module = Py_InitModule("libpywmd", pywmdMethods);
	#endif
#endif
	import_array();

	if (module == NULL) INITERROR;

	struct module_state *st = GET_MODULE_STATE(module);

	// Setting up the errors
	char other_error[] = "pywmd.Error";
	st->other_error = PyErr_NewException(other_error, NULL, NULL);
	if (st->other_error == NULL) {
		Py_DECREF(module);
		INITERROR;
	}

	char type_error[] = "pywmd.TypeError";
	st->type_error = PyErr_NewException(type_error, NULL, NULL);
	if (st->type_error == NULL) {
		Py_DECREF(module);
		INITERROR;
	}

	char io_error[] = "pywmd.IOError";
	st->io_error = PyErr_NewException(io_error, NULL, NULL);
	if (st->io_error == NULL) {
		Py_DECREF(module);
		INITERROR;
	}

	#ifdef USE_MPI
	// SET-UP THE MPI ENVIRONMENT
	init_mpi(&myid, &num_procs, &local_id, &num_local_procs);
	#else
	myid = 0; num_procs = 1; local_id = 0; num_local_procs = 1;
	#endif

	Py_AtExit(python_cleanup);	

	//cout << "Initialized module libpywmd " << endl;

#if PY_MAJOR_VERSION >= 3
	return module;
#endif

}
