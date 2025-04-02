# random_split_check.py - what if I had split by image instead of by person?
#
# Same final settings, but train/val/test are a random shuffle of all images,
# so the same people (and almost identical neighbouring video frames) are in
# both train and test. This number is NOT a fair result, I only ran it to see
# how much the easy split inflates the accuracy.
#
# run:  python experiments/random_split_check.py

import sys
sys.path.append("src")
import numpy as np
from data import OUT_FILE, shift
from train import train

f = np.load(OUT_FILE)
X = f["X"].astype(np.float64) / 255.0
y = f["y"]

rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
n_train = int(0.7 * len(y))
n_val = int(0.15 * len(y))
tr = perm[:n_train]
va = perm[n_train:n_train + n_val]
te = perm[n_train + n_val:]

shifts = [(2, 0), (-2, 0), (0, 2), (0, -2)]
X_train = np.vstack([X[tr]] + [shift(X[tr], dx, dy) for dx, dy in shifts])
y_train = np.concatenate([y[tr]] * 5)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

d = {
    "X_train": (X_train - mean) / std, "y_train": y_train,
    "X_val": (X[va] - mean) / std, "y_val": y[va],
}
net, history, best_epoch = train(d, [64], 0.1, 64, 8, "he", 0.003, 0, decay=0.6)
print("\nrandom image split:  val acc %.4f   test acc %.4f" % (
    net.accuracy(d["X_val"], y[va]), net.accuracy((X[te] - mean) / std, y[te])))
