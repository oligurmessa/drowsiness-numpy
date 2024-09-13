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








