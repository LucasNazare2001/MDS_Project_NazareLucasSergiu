"""
tamper_detect.py - geometric inconsistency analysis.
Applies the paper's method (orientation estimation) to blocks of the image and
measures how consistent the local orientations are with each other.
"""
from __future__ import annotations
import numpy as np
import bispectral as bs

# Coarse angle grid: only relative differences between regions matter here, not
# absolute precision, so a few angles are enough (faster).
DETECTOR_ANGLES = np.arange(-45, 46, 9)
MAX_SIDE = 384   # downscale large images for speed


def _to_gray01(image) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.max() > 1.5:
        img = img / 255.0
    return img


def _texture_energy(block: np.ndarray) -> float:
    # mean gradient: flat (unreliable) regions are ignored
    gx = np.abs(np.diff(block, axis=1)).mean() if block.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(block, axis=0)).mean() if block.shape[0] > 1 else 0.0
    return 0.5 * (gx + gy)


def analyze(image,
            grid: int = 4,                      # blocks per axis
            overlap: float = 0.34,              # overlap between blocks
            max_lines: int = 24,                # scan lines per block
            texture_percentile: float = 40.0,   # texture threshold (skip flat blocks)
            flag_sigma: float = 2.0,            # threshold (in sigma) to flag a block
            topk: int = 3) -> dict:
    """Estimate each block's orientation, compute the reference orientation
    (median) and flag the blocks that depart from it."""
    gray = _to_gray01(image)
    mx = max(gray.shape)
    if mx > MAX_SIDE:                            # downscale large images
        gray = gray[::int(np.ceil(mx / MAX_SIDE)), ::int(np.ceil(mx / MAX_SIDE))]
    h, w = gray.shape
    bh = max(16, int(round(h / grid)))
    bw = max(16, int(round(w / grid)))
    sy = max(1, int(bh * (1 - overlap)))
    sx = max(1, int(bw * (1 - overlap)))

    # estimate the local orientation of each block
    centers, tv, th_, energy = [], [], [], []
    for y in range(0, max(1, h - bh + 1), sy):
        for x in range(0, max(1, w - bw + 1), sx):
            block = gray[y:y + bh, x:x + bw]
            if block.shape[0] < 16 or block.shape[1] < 16:
                continue
            out = bs.estimate_orientation_2d(block, angles=DETECTOR_ANGLES,
                                             max_lines=max_lines, calibrate=True)
            centers.append((y + bh // 2, x + bw // 2))
            tv.append(out["theta_vertical"]); th_.append(out["theta_horizontal"])
            energy.append(_texture_energy(block))

    centers = np.array(centers)
    tv = np.array(tv); th_ = np.array(th_); energy = np.array(energy)
    if len(centers) == 0:
        vis = (np.stack([gray] * 3, -1) * 255).astype(np.uint8)
        return dict(vis=vis, score=0.0, max_deviation=0.0, dispersion=0.0,
                    frac_inconsistent=0.0, n_blocks=0, ref_v=0.0, ref_h=0.0)

    # keep only sufficiently textured blocks
    trust = energy >= np.percentile(energy, texture_percentile)
    if trust.sum() < 3:
        trust = np.ones_like(energy, dtype=bool)

    # reference orientation and per-block deviation
    ref_v = float(np.median(tv[trust])); ref_h = float(np.median(th_[trust]))
    dev = np.hypot(tv - ref_v, th_ - ref_h)
    dtr = dev[trust]

    # robust threshold to flag anomalous blocks
    spread = 1.4826 * (np.median(np.abs(dtr - np.median(dtr))) + 1e-6)
    flagged = trust & (dev > np.median(dtr) + flag_sigma * spread)

    # score = mean of the top-k most deviating blocks (robust to noise)
    score = float(np.sort(dtr)[::-1][:max(1, topk)].mean())

    vis = _render(gray, centers, dev, flagged, bh, bw)
    return dict(vis=vis, score=score, max_deviation=float(dtr.max()),
                dispersion=float(spread),
                frac_inconsistent=float(flagged.sum() / max(1, trust.sum())),
                n_blocks=int(trust.sum()), ref_v=ref_v, ref_h=ref_h)


def inconsistency_score(image, **kw) -> float:
    # single scalar per image (used by the dataset evaluation)
    return analyze(image, **kw)["score"]


def _render(gray, centers, dev, flagged, bh, bw) -> np.ndarray:
    # overlay a heat-map (red = high deviation) and outline flagged blocks
    h, w = gray.shape
    d = (dev - dev.min()) / (dev.max() - dev.min() + 1e-9)
    heat = np.zeros((h, w)); wsum = np.zeros((h, w))
    for (cy, cx), dv in zip(centers, d):
        y0, y1 = max(0, cy - bh // 2), min(h, cy + bh // 2)
        x0, x1 = max(0, cx - bw // 2), min(w, cx + bw // 2)
        heat[y0:y1, x0:x1] += dv; wsum[y0:y1, x0:x1] += 1
    heat /= np.maximum(wsum, 1e-9)

    out = 0.55 * np.stack([gray] * 3, -1)
    out[..., 0] += 0.45 * heat
    out[..., 2] += 0.45 * (1 - heat)
    out = np.clip(out, 0, 1)
    for (cy, cx), fl in zip(centers, flagged):
        if not fl:
            continue
        y0, y1 = max(0, cy - bh // 2), min(h - 1, cy + bh // 2)
        x0, x1 = max(0, cx - bw // 2), min(w - 1, cx + bw // 2)
        out[y0:y1, x0] = [1, 0, 0]; out[y0:y1, x1] = [1, 0, 0]
        out[y0, x0:x1] = [1, 0, 0]; out[y1, x0:x1] = [1, 0, 0]
    return (out * 255).astype(np.uint8)


def make_test_image(n: int = 256, host_angle: float = 25.0,
                    patch_angle: float = -30.0, seed: int = 0):
    # test image: background at one orientation, central region at another
    def norm(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-9)
    host = norm(bs.synth_projected_image(n // 2, theta_y=host_angle, seed=seed))
    patch = norm(bs.synth_projected_image(n // 2, theta_y=patch_angle, seed=seed + 99))
    H = host.shape[0]; ph = H // 3; y0 = x0 = H // 2 - ph // 2
    host[y0:y0 + ph, x0:x0 + ph] = patch[y0:y0 + ph, x0:x0 + ph]
    return (host * 255).astype(np.uint8), (y0, x0, ph, ph)


if __name__ == "__main__":
    img, bbox = make_test_image()
    r = analyze(img)
    print(f"score={r['score']:.1f}  max_dev={r['max_deviation']:.1f}  n_blocks={r['n_blocks']}")
