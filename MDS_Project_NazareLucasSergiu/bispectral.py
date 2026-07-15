"""
bispectral.py - the paper's method (Farid & Kosecka, IEEE TIP 2007).
Estimates the slant of a planar surface from a single image by minimising the
residual bicoherence over candidate rectification angles.
"""
from __future__ import annotations
import numpy as np

# Angle grid searched for the orientation (degrees).
DEFAULT_ANGLES = np.arange(-45, 46, 3)

# Calibration constant: the raw angle underestimates the true slant by a roughly
# constant factor. Fitted by fit_calibration() for the default settings.
CALIB = 0.880


# --- Bicoherence (third-order phase coupling), scalar in [0, 1] ---
def mean_bicoherence(signal: np.ndarray,
                     seg_len: int = 64,      # segment length
                     overlap: int = 32) -> float:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    step = max(1, seg_len - overlap)
    win = np.hanning(seg_len)

    num = np.zeros((seg_len, seg_len), dtype=np.complex128)
    den1 = np.zeros((seg_len, seg_len), dtype=np.float64)
    den2 = np.zeros((seg_len, seg_len), dtype=np.float64)
    idx = (np.arange(seg_len)[:, None] + np.arange(seg_len)[None, :]) % seg_len

    n_seg = 0
    for start in range(0, len(signal) - seg_len + 1, step):
        seg = (signal[start:start + seg_len] - signal[start:start + seg_len].mean()) * win
        F = np.fft.fft(seg)
        F1, F2, F12 = F[:, None], F[None, :], F[idx]
        num += F1 * F2 * np.conj(F12)
        den1 += np.abs(F1 * F2) ** 2
        den2 += np.abs(F12) ** 2
        n_seg += 1

    if n_seg == 0:
        return 0.0
    num /= n_seg; den1 /= n_seg; den2 /= n_seg
    bic = np.abs(num) / (np.sqrt(den1 * den2) + 1e-12)
    return float(bic.mean())


# --- Perspective projection and rectification of a 1-D signal ---
def _project_coords(x: np.ndarray, theta_deg: float) -> np.ndarray:
    # fronto-parallel coord -> image coord for a plane tilted by theta
    t = np.deg2rad(theta_deg)
    return (np.cos(t) * x) / (-np.sin(t) * x + 2.0)


def rectify_signal(g: np.ndarray, theta_deg: float) -> np.ndarray:
    # hypothesise tilt theta and warp the signal back to fronto-parallel
    g = np.asarray(g, dtype=np.float64).ravel()
    n = len(g)
    x = np.linspace(-1, 1, n)
    proj = _project_coords(x, theta_deg)
    img_grid = np.linspace(proj.min(), proj.max(), n)
    return np.interp(proj, img_grid, g)


# --- Orientation estimation ---
def _refined_min(angles, curve):
    # angle minimising the curve, refined with a local 3-point parabola
    j = int(np.argmin(curve))
    if 0 < j < len(angles) - 1:
        a0, a1 = angles[j - 1], angles[j]
        b0, b1, b2 = curve[j - 1], curve[j], curve[j + 1]
        denom = (b0 - 2 * b1 + b2)
        if abs(denom) > 1e-12:
            return a1 + 0.5 * (b0 - b2) / denom * (a1 - a0)
    return float(angles[j])


def estimate_orientation_1d(signal: np.ndarray,
                            angles=DEFAULT_ANGLES,
                            seg_len: int = 64,
                            overlap: int = 32,
                            calibrate: bool = True) -> float:
    # slant = angle that minimises the bicoherence after rectification
    signal = np.asarray(signal, dtype=np.float64).ravel()
    curve = np.array([mean_bicoherence(rectify_signal(signal, th), seg_len, overlap)
                      for th in angles])
    theta = -_refined_min(angles, curve)
    if calibrate:
        theta = theta / CALIB
    return float(np.clip(theta, angles.min(), angles.max()))


def estimate_orientation_2d(image: np.ndarray,
                            angles=DEFAULT_ANGLES,
                            max_lines: int = 48,   # scan lines used per axis
                            calibrate: bool = True) -> dict:
    # estimate vertical (rows) and horizontal (columns) slant
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.max() > 1.5:
        img = img / 255.0
    h, w = img.shape

    rows = np.linspace(0, h - 1, min(max_lines, h)).astype(int)
    v_est = [estimate_orientation_1d(img[r, :], angles, calibrate=calibrate) for r in rows]
    cols = np.linspace(0, w - 1, min(max_lines, w)).astype(int)
    # columns have the opposite sign convention -> flip
    h_est = [-estimate_orientation_1d(img[:, c], angles, calibrate=calibrate) for c in cols]

    return {
        "theta_vertical": float(np.median(v_est)),
        "theta_horizontal": float(np.median(h_est)),
        "vertical_estimates": np.array(v_est),
        "horizontal_estimates": np.array(h_est),
    }


# --- Projected-texture synthesis (to reproduce the paper's figures) ---
def synth_projected_signal(n: int = 256, theta_deg: float = 0.0,
                           seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, 2 * n, endpoint=False)
    u = _project_coords(x, theta_deg)
    g = np.zeros_like(u)
    for k in range(1, n + 1):
        g += (1.0 / k) * np.cos(2 * np.pi * k * u + rng.uniform(-np.pi, np.pi))
    return g


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def synth_projected_image(n: int = 64, theta_x: float = 0.0, theta_y: float = 0.0,
                          seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1, 1, 2 * n, endpoint=False)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    R = _rot_y(np.deg2rad(theta_y)) @ _rot_x(np.deg2rad(theta_x))
    P = R @ pts + np.array([[0], [0], [2.0]])
    xp = (P[0] / P[2]).reshape(X.shape)
    yp = (P[1] / P[2]).reshape(X.shape)
    img = np.zeros_like(xp)
    for k in range(1, n + 1):
        th = rng.uniform(-np.pi, np.pi); ph = rng.uniform(-np.pi, np.pi)
        img += (1.0 / k) * np.cos(2 * np.pi * k * (np.cos(th) * xp + np.sin(th) * yp) + ph)
    return img


# --- 2-D perspective rectification of an image (for the visual demo) ---
def rectify_image_2d(image, theta_vertical, theta_horizontal, focal=None, depth=2.0):
    import cv2
    img = np.asarray(image)
    h, w = img.shape[:2]
    f = float(focal if focal is not None else max(h, w))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    R = _rot_y(np.deg2rad(theta_vertical)) @ _rot_x(np.deg2rad(theta_horizontal))
    t = np.array([0, 0, depth])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    Hinv = np.linalg.inv(H)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    proj = Hinv @ corners
    proj = (proj[:2] / proj[2]).T
    minxy = proj.min(0); span = (proj.max(0) - minxy); span[span == 0] = 1
    S = np.array([[w / span[0], 0, -minxy[0] * w / span[0]],
                  [0, h / span[1], -minxy[1] * h / span[1]],
                  [0, 0, 1]])
    return cv2.warpPerspective(img, S @ Hinv, (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


# --- Fit the calibration constant (run once) ---
def fit_calibration(n_signal=256, seeds=range(12),
                    true_angles=(-30, -20, -10, 10, 20, 30)) -> float:
    raw, tru = [], []
    for td in true_angles:
        for s in seeds:
            g = synth_projected_signal(n_signal, theta_deg=td, seed=s)
            raw.append(estimate_orientation_1d(g, calibrate=False)); tru.append(td)
    return float(np.polyfit(tru, raw, 1)[0])


if __name__ == "__main__":
    print(f"fitted CALIB = {fit_calibration():.3f}  (current in file: {CALIB})")
