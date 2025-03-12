# evaluate.py - final numbers on the TEST set
#
# The test set is 7 people the network has never seen. I only ran this after
# I was done tuning on the validation set.
#
# run:  python src/evaluate.py

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_data, IMG_SIZE
from model import NeuralNet

NAME = "final"

d = load_data(standardize=False)
norm = np.load("results/%s_norm.npz" % NAME)
X_test = (d["X_test"] - norm["mean"]) / norm["std"]
y_test = d["y_test"]

net = NeuralNet.load("results/%s_model.npz" % NAME)
pred = net.predict(X_test)

acc = np.mean(pred == y_test)
majority = max(np.mean(y_test == 0), np.mean(y_test == 1))
print("test images:", len(y_test), " test subjects:", sorted(d["test_ids"].tolist()))
print("majority class baseline: %.2f%%" % (100 * majority))
print("test accuracy:           %.2f%%" % (100 * acc))

# confusion matrix, rows = true label, columns = predicted (0 closed, 1 open)
cm = np.zeros((2, 2), dtype=int)
for t in [0, 1]:
    for p in [0, 1]:
        cm[t, p] = np.sum((y_test == t) & (pred == p))
print("\nconfusion matrix (rows true, cols predicted; 0 = closed, 1 = open)")
print(cm)

# "closed" is the class that matters for drowsiness, so treat it as positive
closed_recall = cm[0, 0] / cm[0].sum()
closed_precision = cm[0, 0] / cm[:, 0].sum()
print("\nclosed-eye recall:    %.2f%%  (of the really closed eyes, how many I caught)" % (100 * closed_recall))
print("closed-eye precision: %.2f%%  (when I say closed, how often it's right)" % (100 * closed_precision))

print("\naccuracy for each test subject:")
for s in sorted(d["test_ids"]):
    mask = d["subject_test"] == s
    print("  subject %2d  %5d images  %.2f%%" % (s, mask.sum(), 100 * np.mean(pred[mask] == y_test[mask])))

# ---- plots ----
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_xticklabels(["closed", "open"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["closed", "open"])
ax.set_xlabel("predicted")
ax.set_ylabel("true")
for t in [0, 1]:
    for p in [0, 1]:
        ax.text(p, t, str(cm[t, p]), ha="center", va="center",
                color="white" if cm[t, p] > cm.max() / 2 else "black")
ax.set_title("test set, accuracy %.1f%%" % (100 * acc))
fig.tight_layout()
fig.savefig("results/confusion_matrix.png", dpi=120)

# some of the mistakes, to see what kind of images it gets wrong
wrong = np.where(pred != y_test)[0]
rng = np.random.default_rng(0)
show = rng.choice(wrong, size=min(24, len(wrong)), replace=False)
names = ["closed", "open"]
fig, axes = plt.subplots(3, 8, figsize=(12, 5))
for ax, i in zip(axes.ravel(), show):
    ax.imshow(d["raw_test"][i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    ax.set_title("true %s\npred %s" % (names[y_test[i]], names[pred[i]]), fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig("results/mistakes.png", dpi=120)
print("\nsaved results/confusion_matrix.png and results/mistakes.png")
