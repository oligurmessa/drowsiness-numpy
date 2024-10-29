# gradcheck.py - checks that backward() is actually right
#
# Check 1: compare my backprop gradients against numerical gradients
#          (f(w + eps) - f(w - eps)) / (2 eps)  on a tiny random network.
# Check 2: train on only 50 real images. If the network can't memorize
#          50 images then something is broken.
#
# run:  python src/gradcheck.py

import os
import numpy as np
from model import NeuralNet


def numerical_gradients(net, X, y, eps=1e-5):
    num_grads = {}
    for key in net.params:
        W = net.params[key]
        g = np.zeros_like(W)
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                old = W[r, c]
                W[r, c] = old + eps
                loss_plus = net.loss(X, y)
                W[r, c] = old - eps
                loss_minus = net.loss(X, y)
                W[r, c] = old
                g[r, c] = (loss_plus - loss_minus) / (2 * eps)
        num_grads[key] = g
    return num_grads


def gradient_check():
    print("Check 1: numerical vs backprop gradients")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 6))
    y = rng.integers(0, 2, size=8)

    # small net with 2 hidden layers and some L2 so everything gets tested
    net = NeuralNet([6, 5, 4, 2], init="he", l2=0.1, seed=1)
    # biases start at 0, make them random so their gradients are tested too
    for key in net.params:
        if key.startswith("b"):
            net.params[key] = rng.standard_normal(net.params[key].shape) * 0.1

    net.forward(X)
    grads = net.backward(y)
    num_grads = numerical_gradients(net, X, y)

    ok = True
    for key in sorted(net.params):
        a = grads[key]
        n = num_grads[key]
        rel_error = np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n))
        print("  %s  relative error = %.2e" % (key, rel_error))
        if rel_error > 1e-6:
            ok = False
    print("  PASSED" if ok else "  FAILED")
    return ok


def overfit_check():
    print("Check 2: overfit 50 images")
    if not os.path.exists("data/eyes_24x24.npz"):
        print("  skipped (run src/data.py first)")
        return True
    from data import load_data
    d = load_data()
    # random 50, not the first 50 (the first 50 are all the same person with
    # the same label, which would be way too easy)
    pick = np.random.default_rng(0).choice(len(d["y_train"]), 50, replace=False)
    X = d["X_train"][pick]
    y = d["y_train"][pick]
    print("  %d closed, %d open" % (np.sum(y == 0), np.sum(y == 1)))

    net = NeuralNet([X.shape[1], 64, 2], init="he", seed=0)
    for step in range(500):
        net.forward(X)
        net.update(net.backward(y), lr=0.1)
    acc = net.accuracy(X, y)
    print("  loss = %.4f, accuracy on the 50 images = %.1f%%" % (net.loss(X, y), 100 * acc))
    ok = acc == 1.0
    print("  PASSED" if ok else "  FAILED")
    return ok


if __name__ == "__main__":
    gradient_check()
    overfit_check()
