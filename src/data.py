# data.py - turns the MRL eye images into numpy arrays
#
# File names in the dataset look like  s0001_00001_0_0_0_0_0_01.png
#   subject _ image number _ gender _ glasses _ eye state _ reflections _ lighting _ sensor
# eye state: 0 = closed, 1 = open   (that's my label)
#
# run once:  python src/data.py
# it reads the images straight out of the zip so I don't have to unzip 85k files

import io
import zipfile
import numpy as np
from PIL import Image

IMG_SIZE = 24
ZIP_FILE = "data/mrlEyes_2018_01.zip"
OUT_FILE = "data/eyes_24x24.npz"


def prepare():
    z = zipfile.ZipFile(ZIP_FILE)
    names = sorted(n for n in z.namelist() if n.endswith(".png"))
    print("found", len(names), "images")

    X = np.zeros((len(names), IMG_SIZE * IMG_SIZE), dtype=np.uint8)
    y = np.zeros(len(names), dtype=np.int64)
    subject = np.zeros(len(names), dtype=np.int64)
    frame = np.zeros(len(names), dtype=np.int64)

    for i, name in enumerate(names):
        parts = name.split("/")[-1].replace(".png", "").split("_")
        subject[i] = int(parts[0][1:])  # "s0001" -> 1
        frame[i] = int(parts[1])
        y[i] = int(parts[4])

        img = Image.open(io.BytesIO(z.read(name))).convert("L")  # grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        X[i] = np.asarray(img).reshape(-1)  # flatten 24x24 -> 576

        if i % 10000 == 0:
            print("  ", i)

    np.savez_compressed(OUT_FILE, X=X, y=y, subject=subject, frame=frame)
    print("saved", OUT_FILE, X.shape)
    print("closed:", np.sum(y == 0), " open:", np.sum(y == 1))


def split_subjects(subject, seed=0):
    # Split by PERSON, not by image. The images are frames from videos so
    # neighbouring frames look almost the same. With a random image split the
    # test set would have near copies of training images.
    ids = np.unique(subject)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(0.7 * len(ids))
    n_val = int(0.15 * len(ids))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def shift(X, dx, dy):
    # moves every image dx pixels right and dy pixels down.
    # np.roll wraps the edge pixels around to the other side, which isn't
    # really correct, but it's only 2 pixels so I left it.
    imgs = X.reshape(-1, IMG_SIZE, IMG_SIZE)
    imgs = np.roll(imgs, dx, axis=2)
    imgs = np.roll(imgs, dy, axis=1)
    return imgs.reshape(-1, IMG_SIZE * IMG_SIZE)


def load_data(standardize=True, augment=False, seed=0):
    f = np.load(OUT_FILE)
    X = f["X"].astype(np.float64) / 255.0  # pixels to [0, 1]
    y = f["y"]
    subject = f["subject"]
    frame = f["frame"]

    train_ids, val_ids, test_ids = split_subjects(subject, seed)
    tr = np.isin(subject, train_ids)
    va = np.isin(subject, val_ids)
    te = np.isin(subject, test_ids)

    X_train = X[tr]
    y_train = y[tr]
    if augment:
        # training set x5: the original + shifted 2 pixels left/right/up/down.
        # the eye isn't in exactly the same spot for every person, and a fully
        # connected net has no idea that a shifted eye is still the same eye
        shifts = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        X_train = np.vstack([X_train] + [shift(X[tr], dx, dy) for dx, dy in shifts])
        y_train = np.concatenate([y_train] * 5)

    mean = np.zeros(X.shape[1])
    std = np.ones(X.shape[1])
    if standardize:
        # mean and std come from the TRAINING set only, then get reused on
        # val and test (otherwise I'd be peeking at the test data)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X = (X - mean) / std

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X[va], "y_val": y[va],
        "X_test": X[te], "y_test": y[te],
        "subject_test": subject[te], "frame_test": frame[te],
        "raw_test": f["X"][te],  # un-normalized pixels, for plotting
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "mean": mean, "std": std,
    }


if __name__ == "__main__":
    prepare()
    d = load_data()
    print("subjects  train/val/test:", len(d["train_ids"]), len(d["val_ids"]), len(d["test_ids"]))
    for s in ["train", "val", "test"]:
        ys = d["y_" + s]
        print("  %-5s %6d images, %.1f%% open" % (s, len(ys), 100 * ys.mean()))
