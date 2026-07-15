"""
evaluate.py - evaluate the geometric-inconsistency score on a labelled dataset.
Expected folder:
    dataset/
    |-- fake/   -> label 1
    |-- real/   -> label 0
Computes the score for each image and measures how well it separates the two
classes: AUC, best threshold (from the ROC), accuracy, confusion matrix, ROC curve.
Use:  python evaluate.py path/to/dataset
"""
from __future__ import annotations
import os
import argparse
import numpy as np
from PIL import Image

import tamper_detect as td

FAKE_NAMES = {"fake", "manipulated", "tampered"}
REAL_NAMES = {"real", "authentic", "original"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _label_of(subfolder: str):
    s = subfolder.lower()
    return 1 if s in FAKE_NAMES else 0 if s in REAL_NAMES else None


def collect(folder: str):
    # gather paths and labels from the real/ and fake/ subfolders
    paths, labels = [], []
    for sub in sorted(os.listdir(folder)):
        subdir = os.path.join(folder, sub)
        lab = _label_of(sub)
        if not os.path.isdir(subdir) or lab is None:
            continue
        for fn in sorted(os.listdir(subdir)):
            if os.path.splitext(fn)[1].lower() in EXTS:
                paths.append(os.path.join(subdir, fn)); labels.append(lab)
    return paths, np.array(labels)


def score_images(paths, grid=4, max_lines=24, progress=None):
    # compute the inconsistency score for each image
    scores = []
    for i, p in enumerate(paths):
        try:
            img = np.array(Image.open(p).convert("RGB"))
            scores.append(td.inconsistency_score(img, grid=grid, max_lines=max_lines))
        except Exception as e:
            print(f"  ! skipped {p}: {e}"); scores.append(np.nan)
        if progress:
            progress((i + 1) / len(paths))
    return np.array(scores)


def roc_auc(labels, scores):
    # ROC curve (fpr, tpr, thresholds) and AUC
    order = np.argsort(-scores)
    y = labels[order]
    P = max(1, y.sum()); N = max(1, len(y) - y.sum())
    tpr, fpr, thr = [0.0], [0.0], [np.inf]
    tp = fp = 0
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P); fpr.append(fp / N); thr.append(scores[order][i])
    tpr = np.array(tpr); fpr = np.array(fpr); thr = np.array(thr)
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, thr, auc


def evaluate(folder, grid=4, max_lines=24, out="results", progress=None):
    paths, labels = collect(folder)
    if len(paths) == 0:
        raise SystemExit("No images found. Expected real/ and fake/ subfolders.")
    scores = score_images(paths, grid=grid, max_lines=max_lines, progress=progress)

    ok = ~np.isnan(scores)
    labels, scores = labels[ok], scores[ok]
    fpr, tpr, thr, auc = roc_auc(labels, scores)

    # threshold maximising (tpr - fpr) on the ROC
    j = np.argmax(tpr - fpr)
    best_thr = thr[j] if np.isfinite(thr[j]) else np.median(scores)
    pred = (scores >= best_thr).astype(int)   # 1 = inconsistent, 0 = consistent
    acc = float((pred == labels).mean())

    # breakdown: how many images of each folder fall into each outcome
    fake_inconsistent = int(((pred == 1) & (labels == 1)).sum())
    fake_consistent   = int(((pred == 0) & (labels == 1)).sum())
    real_inconsistent = int(((pred == 1) & (labels == 0)).sum())
    real_consistent   = int(((pred == 0) & (labels == 0)).sum())

    os.makedirs(out, exist_ok=True)
    _plot_roc(fpr, tpr, auc, os.path.join(out, "roc.png"))
    return dict(n=len(labels), n_real=int((labels == 0).sum()), n_fake=int((labels == 1).sum()),
                auc=auc, threshold=float(best_thr), accuracy=acc,
                breakdown=dict(fake_inconsistent=fake_inconsistent,
                               fake_consistent=fake_consistent,
                               real_inconsistent=real_inconsistent,
                               real_consistent=real_consistent),
                mean_real=float(scores[labels == 0].mean()) if (labels == 0).any() else 0.0,
                mean_fake=float(scores[labels == 1].mean()) if (labels == 1).any() else 0.0,
                roc_png=os.path.join(out, "roc.png"))


def _plot_roc(fpr, tpr, auc, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=110)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC - geometric inconsistency score")
    ax.legend(); ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def _print(r):
    b = r["breakdown"]
    print(f"\nImages: {r['n']}  ({r['n_real']} real, {r['n_fake']} fake)")
    print(f"AUC        : {r['auc']:.3f}")
    print(f"Threshold  : {r['threshold']:.1f}deg  (score >= threshold -> inconsistent)")
    print(f"Mean score : real {r['mean_real']:.1f}deg, fake {r['mean_fake']:.1f}deg")
    print()
    print(f"{'':>8}| {'inconsistent':>13} | {'consistent':>11}")
    print("-" * 38)
    print(f"{'fake/':>8}| {b['fake_inconsistent']:>13} | {b['fake_consistent']:>11}")
    print(f"{'real/':>8}| {b['real_inconsistent']:>13} | {b['real_consistent']:>11}")
    print("-" * 38)
    print(f"ROC saved  : {r['roc_png']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="dataset folder with real/ and fake/ subfolders")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--max-lines", type=int, default=24)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    _print(evaluate(args.folder, grid=args.grid, max_lines=args.max_lines, out=args.out))