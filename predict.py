"""Usage: python predict.py path/to/image.jpg [--threshold 0.5]"""
import argparse
import joblib
import numpy as np
from common import embed, load_image


def predict(img, clf, threshold=0.5, top_k=3):
    e = embed(img)
    if e is None:
        return None, []
    proba = clf.predict_proba([e])[0]
    order = np.argsort(proba)[::-1][:top_k]
    top = [(clf.classes_[i], float(proba[i])) for i in order]
    label = top[0][0] if top[0][1] >= threshold else "Unknown"
    return label, top


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    clf = joblib.load(a.model)
    label, top = predict(load_image(a.image), clf, a.threshold)
    if label is None:
        print("No face detected.")
    else:
        print(f"Prediction: {label.replace('_', ' ')}")
        for n, p in top:
            print(f"  {n.replace('_', ' '):25s} {p:.1%}")
