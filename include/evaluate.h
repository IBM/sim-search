// (c)Copyright IBM Corporation 2019, 2020, 2021
// Author: Kubilay Atasu

#ifndef EVALUATE_H
#define EVALUATE_H

#include <vector>

using namespace std;

namespace docsim {

vector <float> evaluate_topk_precision(int N, int K, int * topk_indices, uint64_t *new_labels, uint64_t *old_labels, bool exclude_first, bool verbose);

}

#endif //EVALUATE_H
