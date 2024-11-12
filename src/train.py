# train.py - trains the network with (mini-batch) gradient descent
#
# examples:
#   python src/train.py --name baseline --hidden 32 --batch 0 --init small --no-standardize
#   python src/train.py --name final --hidden 64 --decay 0.6 --l2 0.003 --augment --epochs 8 --save
#
# Every run adds one row to experiments/log.csv so I can compare them later.
# NOTE: this file never looks at the test set. Only evaluate.py does.

import argparse
import csv
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_data
from model import NeuralNet

LOG_FILE = "experiments/log.csv"


def train(d, hidden, lr, batch_size, epochs, init, l2, seed, decay=1.0):
    X, y = d["X_train"], d["y_train"]
    m = X.shape[0]
    if batch_size == 0:
        batch_size = m  # full batch

    net = NeuralNet([X.shape[1]] + hidden + [2], init=init, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0
    best_params = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(m)  # reshuffle every epoch
        for start in range(0, m, batch_size):
            idx = perm[start:start + batch_size]
            net.forward(X[idx])
            grads = net.backward(y[idx])
            net.update(grads, lr)

        history["train_loss"].append(net.loss(X, y))
        history["val_loss"].append(net.loss(d["X_val"], d["y_val"]))
        history["train_acc"].append(net.accuracy(X, y))
        history["val_acc"].append(net.accuracy(d["X_val"], d["y_val"]))

        if history["val_acc"][-1] > best_val_acc:
            best_val_acc = history["val_acc"][-1]
            best_params = {k: v.copy() for k, v in net.params.items()}
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print("epoch %3d  train loss %.4f  val loss %.4f  train acc %.4f  val acc %.4f" % (
                epoch, history["train_loss"][-1], history["val_loss"][-1],
                history["train_acc"][-1], history["val_acc"][-1]))

        lr = lr * decay  # learning rate decay (decay=1 means constant lr)

    # go back to the weights that were best on validation
    net.params = best_params
    return net, history, best_epoch


def plot_curves(history, path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, history["train_loss"], label="train")
    ax[0].plot(ep, history["val_loss"], label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[1].plot(ep, history["train_acc"], label="train")
    ax[1].plot(ep, history["val_acc"], label="val")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="run")
    p.add_argument("--hidden", default="64", help="hidden sizes like 64 or 64,32. use 0 for no hidden layer")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64, help="0 = full batch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--init", default="he", choices=["he", "small"])
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--decay", type=float, default=1.0, help="multiply lr by this after every epoch")
    p.add_argument("--augment", action="store_true", help="add shifted copies of the training images")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    hidden = [int(h) for h in args.hidden.split(",") if int(h) > 0]
    d = load_data(standardize=not args.no_standardize, augment=args.augment)

    t0 = time.time()
    net, history, best_epoch = train(d, hidden, args.lr, args.batch, args.epochs,
                                     args.init, args.l2, args.seed, args.decay)
    seconds = time.time() - t0

    # val_best = the best epoch (these are the weights I keep)
    # val_last = the last epoch. If these two are far apart the training was
    # jumping around and val_best is probably a bit lucky.
    train_acc = net.accuracy(d["X_train"], d["y_train"])
    val_best = net.accuracy(d["X_val"], d["y_val"])
    val_last = history["val_acc"][-1]
    print("\n%s: best epoch %d, train acc %.4f, val acc best %.4f / last %.4f (%.0fs)" % (
        args.name, best_epoch, train_acc, val_best, val_last, seconds))

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["name", "hidden", "lr", "decay", "batch", "init", "standardize", "l2",
                        "augment", "epochs", "best_epoch", "train_acc", "val_best", "val_last"])
        w.writerow([args.name, args.hidden, args.lr, args.decay, args.batch if args.batch else "full",
                    args.init, not args.no_standardize, args.l2, args.augment, args.epochs,
                    best_epoch, "%.4f" % train_acc, "%.4f" % val_best, "%.4f" % val_last])

    if args.save:
        net.save("results/%s_model.npz" % args.name)
        # evaluate.py needs the same mean/std to normalize the test images
        np.savez("results/%s_norm.npz" % args.name, mean=d["mean"], std=d["std"])
        plot_curves(history, "results/%s_curves.png" % args.name)
        print("saved results/%s_model.npz" % args.name)
