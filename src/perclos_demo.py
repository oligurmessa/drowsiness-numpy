# perclos_demo.py - turning "is the eye closed" into "is the person drowsy"
#
# One closed eye doesn't mean drowsy (people blink). PERCLOS = the fraction of
# recent frames where the eye is closed. Blinking gives a small PERCLOS, long
# eye closures give a big one.
#
# I don't have a real video of a drowsy driver, so this SIMULATES one:
# 60 seconds at 10 frames per second, first half "alert" (short blinks),
# second half "drowsy" (eyes stay closed for seconds). Every frame is a real
# test-set image with the right label, and the network predicts each frame.
#
# run:  python src/perclos_demo.py

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_data
from model import NeuralNet

FPS = 10
WINDOW = 3 * FPS    # look at the last 3 seconds
THRESHOLD = 0.4     # drowsy if closed more than 40% of the window (my choice)

rng = np.random.default_rng(0)

# ---- make the true open/closed sequence (1 = open, 0 = closed) ----
states = []
# alert: open about 4 s, then a blink of 2-3 frames
while len(states) < 30 * FPS:
    states += [1] * int(rng.integers(30, 50))
    states += [0] * int(rng.integers(2, 4))
states = states[:30 * FPS]
# drowsy: open 1-2 s, then closed 2-4 s
while len(states) < 60 * FPS:
    states += [1] * int(rng.integers(10, 20))
    states += [0] * int(rng.integers(20, 40))
states = np.array(states[:60 * FPS])

# ---- pick a real test image for every frame and predict it ----
d = load_data(standardize=False)
norm = np.load("results/final_norm.npz")
X_test = (d["X_test"] - norm["mean"]) / norm["std"]
y_test = d["y_test"]
closed_idx = np.where(y_test == 0)[0]
open_idx = np.where(y_test == 1)[0]
frames = np.array([rng.choice(open_idx) if s == 1 else rng.choice(closed_idx) for s in states])

net = NeuralNet.load("results/final_model.npz")
pred = net.predict(X_test[frames])
print("per-frame accuracy in this sequence: %.1f%%" % (100 * np.mean(pred == states)))


def perclos(seq):
    # fraction of closed frames in the last WINDOW frames
    out = np.zeros(len(seq))
    for t in range(len(seq)):
        recent = seq[max(0, t - WINDOW + 1):t + 1]
        out[t] = np.mean(recent == 0)
    return out


p_true = perclos(states)
p_pred = perclos(pred)
drowsy = p_pred > THRESHOLD

t = np.arange(len(states)) / FPS
alert_part = t < 30
print("alert half:  flagged drowsy in %.1f%% of frames (should be 0)" % (100 * drowsy[alert_part].mean()))
print("drowsy half: flagged drowsy in %.1f%% of frames" % (100 * drowsy[~alert_part].mean()))
first = np.argmax(drowsy & ~alert_part)
print("first drowsy flag at t = %.1f s (drowsy part starts at 30 s)" % t[first])

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].step(t, 1 - states, label="true", linewidth=1)
ax[0].step(t, (1 - pred) * 0.9, label="predicted", linewidth=1, alpha=0.7)
ax[0].set_ylabel("eye closed")
ax[0].legend(loc="upper left")
ax[1].plot(t, p_true, label="true PERCLOS")
ax[1].plot(t, p_pred, label="predicted PERCLOS")
ax[1].axhline(THRESHOLD, color="red", linestyle="--", label="threshold")
ax[1].fill_between(t, 0, 1, where=drowsy, color="red", alpha=0.15, label="flagged drowsy")
ax[1].set_ylabel("PERCLOS (3 s window)")
ax[1].set_xlabel("time (s)")
ax[1].legend(loc="upper left")
fig.tight_layout()
fig.savefig("results/perclos_demo.png", dpi=120)
print("saved results/perclos_demo.png")
