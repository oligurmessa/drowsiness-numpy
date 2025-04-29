# pca.py - the linear algebra side quest: PCA of the eye images with the SVD
#
# If X is the centered training matrix (one image per row), the SVD
#   X = U S V^T
# gives the principal directions as the rows of V^T, and the variance along
# each one is s^2 / m. Each direction is a 576-vector, so it can be shown as a
# 24x24 image ("eigen-eyes").
#
# Also tries training the network on only the top k components.
#
# run:  python src/pca.py

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_data, IMG_SIZE
from train import train

d = load_data(standardize=False)
X = d["X_train"]
mean = X.mean(axis=0)
Xc = X - mean

U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var = S ** 2 / len(X)
explained = np.cumsum(var) / np.sum(var)
for frac in [0.90, 0.95, 0.99]:
    k = np.argmax(explained >= frac) + 1
    print("%d of 576 components explain %.0f%% of the variance" % (k, 100 * frac))

# check that the rows of Vt really are orthonormal
print("max |V^T V - I| =", np.abs(Vt @ Vt.T - np.eye(len(Vt))).max())

fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(Vt[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    ax.set_title("PC %d" % (i + 1), fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig("results/eigen_eyes.png", dpi=120)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(np.arange(1, 577), explained)
ax.set_xscale("log")
ax.set_xlabel("number of components")
ax.set_ylabel("fraction of variance explained")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/pca_variance.png", dpi=120)
print("saved results/eigen_eyes.png and results/pca_variance.png")

# ---- train on the top k components instead of all 576 pixels ----
# project: Z = (X - mean) V_k, then divide by the std of each component
print("\ntraining on PCA features (same settings as step 8, no augmentation):")
for k in [20, 50, 100]:
    Vk = Vt[:k].T
    scale = np.sqrt(var[:k])
    dk = {
        "X_train": Xc @ Vk / scale, "y_train": d["y_train"],
        "X_val": (d["X_val"] - mean) @ Vk / scale, "y_val": d["y_val"],
    }
    net, history, best_epoch = train(dk, [64], 0.1, 64, 30, "he", 0.01, 0, decay=0.85)
    print("k = %3d  val acc best %.4f / last %.4f\n" % (k, max(history["val_acc"]), history["val_acc"][-1]))
