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


class NeuralNet:
    def __init__(self, layer_sizes, init="he", l2=0.0, seed=0):
        # layer_sizes is like [576, 64, 2]  (input, hidden..., output)
        # [576, 2] means no hidden layer = logistic regression
        self.sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # number of weight matrices
        self.l2 = l2
        self.params = {}
        rng = np.random.default_rng(seed)

        for i in range(1, self.L + 1):
            n_in = layer_sizes[i - 1]
            n_out = layer_sizes[i]
            if init == "he":
                # He init: variance 2/n_in, keeps the ReLU outputs from shrinking
                W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
            else:
                # "small": what I tried first, just tiny random numbers
                W = rng.standard_normal((n_in, n_out)) * 0.01
            self.params["W" + str(i)] = W
            self.params["b" + str(i)] = np.zeros((1, n_out))

    def forward(self, X):
        # saves all the Z's and A's because backward() needs them
        self.cache = {"A0": X}
        A = X
        for i in range(1, self.L + 1):
            Z = A @ self.params["W" + str(i)] + self.params["b" + str(i)]
            if i == self.L:
                A = softmax(Z)
            else:
                A = relu(Z)
            self.cache["Z" + str(i)] = Z
            self.cache["A" + str(i)] = A
        return A

    def loss(self, X, y):
        P = self.forward(X)
        loss = cross_entropy(P, y)
        if self.l2 > 0:
            for i in range(1, self.L + 1):
                loss += 0.5 * self.l2 * np.sum(self.params["W" + str(i)] ** 2)
        return loss

    def backward(self, y):
        # must call forward() on the same batch right before this
        m = y.shape[0]
        grads = {}

        # softmax + cross entropy together give the nice (P - Y) / m
        P = self.cache["A" + str(self.L)]
        dZ = (P - one_hot(y, self.sizes[-1])) / m

        for i in range(self.L, 0, -1):
            A_prev = self.cache["A" + str(i - 1)]
            W = self.params["W" + str(i)]

            grads["W" + str(i)] = A_prev.T @ dZ + self.l2 * W
            grads["b" + str(i)] = dZ.sum(axis=0, keepdims=True)

            if i > 1:
                dA_prev = dZ @ W.T
                # ReLU derivative: 1 where Z was positive, 0 otherwise
                dZ = dA_prev * (self.cache["Z" + str(i - 1)] > 0)

        return grads

    def update(self, grads, lr):
        # plain gradient descent step
        for key in self.params:
            self.params[key] -= lr * grads[key]

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def save(self, path):
        np.savez(path, sizes=np.array(self.sizes), **self.params)

    @staticmethod
    def load(path):
        f = np.load(path)
        net = NeuralNet(list(f["sizes"]))
        for key in net.params:
            net.params[key] = f[key]
        return net
