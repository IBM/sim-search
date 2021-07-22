// (c) Copyright IBM Corporation 2019, 2020, 2021
// Author: Kubilay Atasu

#ifndef PARSE_H
#define PARSE_H

#include <map>
#include "docsim.h"

namespace docsim {

	void L1norm(std::vector < sparse_vec > &docs, double scale);
	int parse_embedding_file(const char *embedding_file_name, bool export_vocab, const char *vocab_file_name, std::map <std::string, int > & vocab, float **Wp, long long & M, long long & V);
	int parse_vocabulary(const char *vocab_file_name, std::map <std::string, int > & vocab);
	int parse_docs_20(const char *path, const int max_histogram_size, const int stop_word_threshold,
				std::vector <sparse_vec> &docs, std::vector <uint64_t> &doc_ids, std::vector <uint64_t> &doc_labels, int &num_docs, int &num_words, std::map <std::string,int> &vocab);
	int parse_docs_mnist(const char *image_file_name, const char *label_file_name, 
				std::vector < sparse_vec > &docs, std::vector < uint64_t > &doc_ids, std::vector < uint64_t > &doc_labels, int &num_docs, int &num_words, float **Wp, long long & M, long long & V);
}

#endif //PARSE_H
