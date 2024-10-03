# model.py - the neural network, only numpy
#
# Convention: every ROW of X is one image, so X is (m, 576).
# A layer is  Z = A_prev @ W + b  with W of shape (n_in, n_out).
# Hidden layers use ReLU, the last layer uses softmax.

import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    # subtract the biggest value in each row first so exp() doesn't overflow
    # (doesn't change the answer because softmax only cares about differences)
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)


def cross_entropy(P, y):
    # P is (m, 2) probabilities, y is (m,) with the correct class 0 or 1
    m = y.shape[0]
    p_correct = P[np.arange(m), y]
    return -np.mean(np.log(p_correct + 1e-12))  # 1e-12 so we never do log(0)


def one_hot(y, num_classes):
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1
    return Y


