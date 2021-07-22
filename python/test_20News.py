# (c) Copyright IBM Corporation 2019, 2020, 2021
# Author: Kubilay Atasu

from __future__ import print_function
from _simsearch import pywmd
from _simsearch import loaders
from _simsearch import embeddings
from _simsearch import evaluate
import time

# LOAD EMBEDDINGS
#W = embeddings.load_word2vec_embeddings('../Embeddings/GoogleNews-vectors-negative300.bin')
W = embeddings.load_embedding_vectors('vectors.npy')

# LOAD DATA FILES
X, labels, ids = loaders.load_20News('../20news-18828/', 'vocabulary.txt')

# RUN WMD ON GPU
print("Running the GPU library")
start = time.time()
D = pywmd.word_movers_distance(W, X)
end = time.time()
print("Took " + str(end-start) + " seconds")

# RUN WMD ON CPU
#start = time.time()
#print("Running the CPU library")
#D = pywmd.word_movers_distance(W, X, Y=None, use_gpu=False)
#end = time.time()
#print("Took " + str(end-start) + " seconds")

# EVALUATE TOP-K SEARCH ACCURACY
K = 32
k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, K, labels)
print("\nTop-K precision results")
print("K:\t\t" + str(k_vec))
print("Precision:\t" + str(prec_vec))

# WRITE RESULTS INTO A FILE
#with open("distances_20news.dat", "w") as ff:
#    for i in range(D.shape[0]):
#        for j in range(K):
#            print((str(labels[i]), str(labels[topk_indices[i][j]]), topk_values[i][j]), file=ff)
