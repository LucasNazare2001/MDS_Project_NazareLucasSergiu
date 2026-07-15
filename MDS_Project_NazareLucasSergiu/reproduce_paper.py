"""
reproduce_paper.py - reproduce the paper's results on synthetic data
(with known slant, so the error can be measured).
Produces: fig3, fig5 and Table I.
Run:  python reproduce_paper.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bispectral as bs


def figure3():
    # bicoherence vs rectification angle: minimal at the true slant
    angles = np.arange(-45, 46, 3)
    true_tilt = 25.0
    curves = [[bs.mean_bicoherence(bs.rectify_signal(
                 bs.synth_projected_signal(256, theta_deg=true_tilt, seed=s), a))
               for a in angles] for s in range(8)]
    curve = np.mean(curves, axis=0)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    ax.plot(angles, curve, "-o", ms=3)
    ax.axvline(-true_tilt, color="r", ls="--", label=f"expected min ({-true_tilt:.0f}deg)")
    ax.set_xlabel("rectification angle (deg)"); ax.set_ylabel("mean bicoherence")
    ax.set_title("Fig. 3 - bicoherence vs angle")
    ax.legend(); fig.tight_layout()
    fig.savefig("fig3_bicoherence_vs_orientation.png")
    print("saved fig3_bicoherence_vs_orientation.png")


def figure5():
    # estimated angle vs true angle
    true_angles = np.arange(-35, 36, 5)
    est_mean, est_sd = [], []
    for td in true_angles:
        e = [bs.estimate_orientation_1d(bs.synth_projected_signal(256, theta_deg=td, seed=s))
             for s in range(10)]
        est_mean.append(np.mean(e)); est_sd.append(np.std(e))
    est_mean = np.array(est_mean)
    slope = np.polyfit(true_angles, est_mean, 1)[0]
    mae = np.mean(np.abs(est_mean - true_angles))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    ax.errorbar(true_angles, est_mean, yerr=est_sd, fmt="o", ms=4, capsize=2, label="estimated")
    ax.plot(true_angles, true_angles, "k--", lw=1, label="ideal (y=x)")
    ax.set_xlabel("true orientation (deg)"); ax.set_ylabel("estimated orientation (deg)")
    ax.set_title(f"Fig. 5 - estimated vs true (slope={slope:.2f}, err={mae:.1f}deg)")
    ax.legend(); ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig("fig5_estimated_vs_true.png")
    print(f"saved fig5_estimated_vs_true.png  (slope={slope:.2f}, error={mae:.1f}deg)")


def table1():
    # simultaneous estimation of vertical and horizontal slant
    print("\nTable I - 2-D estimation (vertical, horizontal)")
    print(f"{'true (v,h)':>12} | {'estimated (v,h)':>16} | {'err':>6}")
    print("-" * 42)
    errs = []
    for tv in (-20, 0, 20):
        for th in (-20, 0, 20):
            e = [bs.estimate_orientation_2d(
                    bs.synth_projected_image(80, theta_x=th, theta_y=tv, seed=s), max_lines=40)
                 for s in range(4)]
            ev = np.mean([o["theta_vertical"] for o in e])
            eh = np.mean([o["theta_horizontal"] for o in e])
            err = np.hypot(ev - tv, eh - th); errs.append(err)
            print(f"{(tv, th)!s:>12} | {f'({ev:+.0f}, {eh:+.0f})':>16} | {err:6.1f}")
    print("-" * 42)
    print(f"mean 2-D error: {np.mean(errs):.1f}deg")


if __name__ == "__main__":
    figure3(); figure5(); table1()
    print("\nNote: absolute precision is coarser than the paper's ~1.4deg "
          "because small, fast signals are used.")
