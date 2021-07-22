# (c) Copyright IBM Corporation 2019, 2020, 2021
# Author: Kubilay Atasu

from __future__ import print_function
from _simsearch import pywmd
from _simsearch import loaders
from _simsearch import evaluate
import time

#######################################
# CREATE EMBEDDING TABLE (COORDINATES OF PIXELS)
#W = simsearch_utils.create_mnist_embeddings(28, 28)
#PARSE MNIST IMAGES
#X, labels, ids = simsearch_utils.load_MNIST("../MNIST/mnist_train.csv")

W, X, labels_X, ids_X = loaders.load_MNIST_with_embeddings("../MNIST/t10k-images-idx3-ubyte", "../MNIST/t10k-labels-idx1-ubyte")

W, Y, labels_Y, ids_Y = loaders.load_MNIST_with_embeddings("../MNIST/train-images-idx3-ubyte", "../MNIST/train-labels-idx1-ubyte")

#######################################
# RUN WMD ON GPU
print("Running the GPU library")
start = time.time()
D = pywmd.word_movers_distance(W, X, Y)
end = time.time()
print("Took " + str(end-start) + " seconds")

# RUN WMD ON CPU
#start = time.time()
#print("Running the CPU library")
#D = pywmd.word_movers_distance(W, X, Y, use_gpu=False)
#end = time.time()
#print("Took " + str(end-start) + " seconds")

# EVALUATE TOP-K SEARCH ACCURACY
K = 32
k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, K, labels_X, labels_Y)
print("\nTop-K precision results")
print("K:\t\t" + str(k_vec))
print("Precision:\t" + str(prec_vec))

# WRITE RESULTS INTO A FILE
#with open("distances_mnist.dat", "w") as ff:
#    for i in range(D.shape[0]):
#        for j in range(K):
#            print((str(labels_X[i]), str(labels_Y[topk_indices[i][j]]), topk_values[i][j]), file=ff)


