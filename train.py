"""Train the classifier.

Expected dataset layout (folder name = player name):
    dataset/
        Virat_Kohli/    img1.jpg img2.jpg ...
        Rohit_Sharma/   ...
        MS_Dhoni/       ...

Usage:  python train.py --data_dir dataset
"""
import argparse, os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from common import IMG_EXTS, embed, load_image


def build_embeddings(data_dir, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"Loaded cached embeddings from {cache}")
        return d["X"], d["y"]
    X, y = [], []
    for person in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        files = [f for f in person.rglob("*") if f.suffix.lower() in IMG_EXTS]
        ok = 0
        for f in files:
            try:
                e = embed(load_image(f))
            except Exception as ex:
                print(f"  skip {f.name}: {ex}")
                continue
            if e is None:
                continue  # no face found
            X.append(e); y.append(person.name); ok += 1
        print(f"{person.name}: {ok}/{len(files)} usable images")
    X, y = np.array(X), np.array(y)
    np.savez(cache, X=X, y=y)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--model_out", default="model.joblib")
    ap.add_argument("--cache", default="embeddings.npz")
    ap.add_argument("--test_size", type=float, default=0.2)
    a = ap.parse_args()

    X, y = build_embeddings(a.data_dir, a.cache)

    # drop players with too few usable images
    names, counts = np.unique(y, return_counts=True)
    for n, c in zip(names, counts):
        if c < 5:
            print(f"WARNING: dropping {n} (only {c} usable images, need >=5)")
    keep = np.isin(y, names[counts >= 5])
    X, y = X[keep], y[keep]
    print(f"\n{len(X)} images, {len(np.unique(y))} players")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=a.test_size, stratify=y, random_state=42)

    clf = SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print("\nTest accuracy: %.3f\n" % (pred == yte).mean())
    print(classification_report(yte, pred))

    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay.from_predictions(yte, pred, ax=ax, xticks_rotation=60, colorbar=False)
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)

    clf.fit(X, y)  # refit on everything for the final model
    joblib.dump(clf, a.model_out)
    print(f"Saved model -> {a.model_out}, confusion matrix -> confusion_matrix.png")


if __name__ == "__main__":
    main()
