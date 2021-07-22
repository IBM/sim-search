# (c) Copyright IBM Corporation 2019, 2020, 2021
# Author: Kubilay Atasu

import unittest
from _simsearch import pywmd
from _simsearch import loaders
from _simsearch import evaluate

class TestSimSearchMNIST(unittest.TestCase):

    def setUp(self):
        self.error_tolerance = 0.0001
        import os
        self.exp_prec0 = os.getenv("EXPECTED_PREC_VAL_MNIST_OPT0", "0.9703,0.96615,0.9614,0.9550375,0.9471875,0.9357844").split(",")
        self.exp_prec1 = os.getenv("EXPECTED_PREC_VAL_MNIST_OPT1", "0.9707,0.96775,0.96415,0.9580875,0.9499437,0.9388781").split(",")
        self.exp_prec2 = os.getenv("EXPECTED_PREC_VAL_MNIST_OPT2", "0.9712,0.9687,0.9648,0.9589375,0.951225,0.94011563").split(",")
        self.exp_prec_small0 = os.getenv("EXPECTED_PREC_VAL_MNIST_SMALL_OPT0", "0.882,0.8535,0.82175,0.778875,0.7178125,0.63521874").split(",")
        self.exp_prec_small1 = os.getenv("EXPECTED_PREC_VAL_MNIST_SMALL_OPT1", "0.886,0.857,0.822,0.785875,0.72475,0.64009374").split(",")
        self.exp_prec_small2 = os.getenv("EXPECTED_PREC_VAL_MNIST_SMALL_OPT2", "0.884,0.858,0.82575,0.7865,0.7264375,0.64146876").split(",")
        self.W, self.X, self.labels_X, self.ids_X = loaders.load_MNIST_with_embeddings("../MNIST/t10k-images-idx3-ubyte", "../MNIST/t10k-labels-idx1-ubyte")
        self.W, self.Y, self.labels_Y, self.ids_Y = loaders.load_MNIST_with_embeddings("../MNIST/train-images-idx3-ubyte", "../MNIST/train-labels-idx1-ubyte")

    def test_mnist_gpu0(self):
        D = pywmd.word_movers_distance(self.W, self.X, self.Y, optimization_level=0)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X, self.labels_Y)
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec0[i])), self.error_tolerance)

    def test_mnist_gpu1(self):
        D = pywmd.word_movers_distance(self.W, self.X, self.Y, optimization_level=1)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X, self.labels_Y)
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec1[i])), self.error_tolerance)

    def test_mnist_gpu2(self):
        D = pywmd.word_movers_distance(self.W, self.X, self.Y, optimization_level=2)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X, self.labels_Y)
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec2[i])), self.error_tolerance)

    def test_mnist_gpu0_emd(self):
        D_ground = pywmd.compute_ground_distances(self.W) 
        D = pywmd.earth_movers_distance(D_ground, self.X, self.Y, optimization_level=0)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X, self.labels_Y)
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec0[i])), self.error_tolerance)

    def test_mnist_gpu1_emd(self):
        D_ground = pywmd.compute_ground_distances(self.W) 
        D = pywmd.earth_movers_distance(D_ground, self.X, self.Y, optimization_level=1)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X, self.labels_Y)
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec1[i])), self.error_tolerance)

    def test_mnist_gpu2_emd(self):
        D_ground = pywmd.compute_ground_distances(self.W) 
        D = pywmd.earth_movers_distance(D_ground, self.X, self.Y, optimization_level=2)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X, self.labels_Y)
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec2[i])), self.error_tolerance)

    def test_mnist_cpu0(self):
        D = pywmd.word_movers_distance(self.W, self.X[0:1000], self.Y[0:1000], use_gpu=False, optimization_level=0)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X[0:1000], self.labels_Y[0:1000])
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec_small0[i])), self.error_tolerance)

    def test_mnist_cpu1(self):
        D = pywmd.word_movers_distance(self.W, self.X[0:1000], self.Y[0:1000], use_gpu=False, optimization_level=1)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X[0:1000], self.labels_Y[0:1000])
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec_small1[i])), self.error_tolerance)

    def test_mnist_cpu2(self):
        D = pywmd.word_movers_distance(self.W, self.X[0:1000], self.Y[0:1000], use_gpu=False, optimization_level=2)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X[0:1000], self.labels_Y[0:1000])
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec_small2[i])), self.error_tolerance)

    def test_mnist_cpu0_emd(self):
        D_ground = pywmd.compute_ground_distances(self.W, use_gpu=False) 
        D = pywmd.earth_movers_distance(D_ground, self.X[0:1000], self.Y[0:1000], use_gpu=False, optimization_level=0)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X[0:1000], self.labels_Y[0:1000])
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec_small0[i])), self.error_tolerance)

    def test_mnist_cpu1_emd(self):
        D_ground = pywmd.compute_ground_distances(self.W, use_gpu=False) 
        D = pywmd.earth_movers_distance(D_ground, self.X[0:1000], self.Y[0:1000], use_gpu=False, optimization_level=1)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X[0:1000], self.labels_Y[0:1000])
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec_small1[i])), self.error_tolerance)

    def test_mnist_cpu2_emd(self):
        D_ground = pywmd.compute_ground_distances(self.W, use_gpu=False) 
        D = pywmd.earth_movers_distance(D_ground, self.X[0:1000], self.Y[0:1000], use_gpu=False, optimization_level=2)
        k_vec, prec_vec, topk_indices, topk_values = evaluate.evaluate_topk(D, 32, self.labels_X[0:1000], self.labels_Y[0:1000])
        for i in range(0,6):
            self.assertLess(abs(prec_vec[i] - float(self.exp_prec_small2[i])), self.error_tolerance)

if __name__ == '__main__':
    unittest.main()
