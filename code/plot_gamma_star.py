"""plot_gamma_star.py
Single-panel heatmap per density.
    x = beta   y = alpha   colour = gamma* (argmax score)
Score = success_rate - lambda * mfpt_capped / MAX_STEPS.
Writes figures/gamma_star_rho{10,20,30,40}.svg
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------- nature palette ----------
INK, CREAM = "#1f2a1f", "#fbfaf4"
FOREST, LEAF, MOSS, SAGE = "#1b4332", "#2d6a4f", "#74c69d", "#95d5b2"
RUST = "#b3261e"

GAMMA_CMAP = LinearSegmentedColormap.from_list(
    "nature_gamma", [CREAM, SAGE, MOSS, LEAF, FOREST]
)

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "axes.edgecolor":   FOREST, "axes.labelcolor": INK, "axes.titlecolor": FOREST,
    "axes.labelsize": 12, "axes.titlesize": 13, "axes.titleweight": "regular",
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "text.color": INK, "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans 3", "Helvetica Neue", "Helvetica",
                        "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.grid": False,
    "savefig.facecolor": CREAM, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

DENSITIES   = ["P0100", "P0200", "P0300", "P0400"]
DENSITY_PCT = {"P0100": 10, "P0200": 20, "P0300": 30, "P0400": 40}
MAX_STEPS   = 200.0


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    A = np.unique(d["alpha"]); B = np.unique(d["beta"]); G = np.unique(d["gamma"])
    S_succ = np.full((len(A), len(B), len(G)), np.nan)
    S_mfpt = np.full_like(S_succ, np.nan)
    ia = np.searchsorted(A, d["alpha"]); ib = np.searchsorted(B, d["beta"])
    ig = np.searchsorted(G, d["gamma"])
    S_succ[ia, ib, ig] = d["success_rate"]
    S_mfpt[ia, ib, ig] = d["mfpt_capped"]
    return A, B, G, S_succ, S_mfpt


def gauss_1d(arr, sigma, axis):
    if sigma <= 0:
        return arr
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2); k /= k.sum()
    a = np.moveaxis(arr, axis, -1)
    n = a.shape[-1]
    p = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(r, r)], mode="edge")
    out = sum(k[i] * p[..., i:i + n] for i in range(len(k)))
    return np.moveaxis(out, -1, axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_fine")
    ap.add_argument("--out",  default="figures")
    ap.add_argument("--lambda-mfpt", type=float, default=0.2)
    ap.add_argument("--smooth-gamma", type=float, default=1.5)
    ap.add_argument("--smooth-ab",    type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for dens in DENSITIES:
        csv = os.path.join(args.root, dens, "grid_sweep.csv")
        if not os.path.exists(csv):
            print(f"skip {dens}"); continue

        A, B, G, S_succ, S_mfpt = load(csv)
        raw = S_succ - args.lambda_mfpt * (S_mfpt / MAX_STEPS)

        sm = gauss_1d(raw, args.smooth_gamma, axis=2)
        sm = gauss_1d(gauss_1d(sm, args.smooth_ab, axis=0), args.smooth_ab, axis=1)

        # argmax + parabolic refine
        k = sm.argmax(axis=2)
        ia, ib = np.indices(k.shape)
        km = np.clip(k - 1, 0, None)
        kp = np.clip(k + 1, None, len(G) - 1)
        y0, y1, y2 = sm[ia, ib, km], sm[ia, ib, k], sm[ia, ib, kp]
        den = y0 - 2 * y1 + y2
        inner = (k > 0) & (k < len(G) - 1) & (den < -1e-12)
        off = np.where(inner, 0.5 * (y0 - y2) / np.where(inner, den, 1.0), 0.0)
        gstar = np.clip(G[k] + np.clip(off, -1, 1) * (G[1] - G[0]), 0, 1)
        smax  = sm.max(axis=2)

        i, j = np.unravel_index(np.argmax(smax), smax.shape)
        a_best, b_best = A[i], B[j]
        g_best, s_best = gstar[i, j], smax[i, j]

        row = sm[i, j, :]
        near = G[row >= row.max() - 0.02]
        g_lo, g_hi = (near.min(), near.max()) if near.size else (np.nan, np.nan)
        pct = DENSITY_PCT[dens]

        # ---------- figure ----------
        fig, ax = plt.subplots(figsize=(7.2, 6.0))

        pcm = ax.pcolormesh(B, A, gstar, cmap=GAMMA_CMAP,
                            vmin=0, vmax=1, shading="nearest",
                            rasterized=True)

        # best marker + label
        ax.plot(b_best, a_best, marker="X", color=RUST,
                markersize=15, markeredgecolor=CREAM, markeredgewidth=1.4,
                zorder=10)
        ax.annotate("best", xy=(b_best, a_best),
                    xytext=(b_best + 1.2, a_best - 1.3),
                    color=RUST, fontsize=11, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=RUST, lw=1.0))

        ax.set_xlim(B[0], B[-1]); ax.set_ylim(A[0], A[-1])
        ax.set_xlabel(r"$\beta$  (goal bias)")
        ax.set_ylabel(r"$\alpha$  (wall penalty)")
        ax.set_title(f"Optimal static $\\gamma^*$ — {pct}\\% walls", pad=10)
        ax.set_aspect("equal", adjustable="box")

        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046,
                            label=r"$\gamma^*$ (argmax success)")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.tick_params(labelsize=9)

        # annotation box in lower-left, same layout as the reference image
        box = (f"best combination at {pct}\\% walls\n"
               f"$\\alpha^* = {a_best:.2f}$\n"
               f"$\\beta^* = {b_best:.2f}$\n"
               f"$\\gamma^* = {g_best:.2f}$\n"
               f"$S_{{\\max}} = {s_best:.3f}$\n"
               f"$\\gamma \\in [{g_lo:.2f}, {g_hi:.2f}]$ within 2\\%")
        ax.text(0.03, 0.03, box, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.45",
                          facecolor=CREAM, edgecolor=INK, linewidth=0.6),
                zorder=11)

        svg = os.path.join(args.out, f"gamma_star_rho{pct}.svg")
        png = os.path.join(args.out, f"gamma_star_rho{pct}.png")
        fig.savefig(svg, format="svg")
        fig.savefig(png, dpi=600)
        plt.close(fig)

        print(f"saved {svg}")
        print(f"   α*={a_best:.2f}  β*={b_best:.2f}  γ*={g_best:.2f}  "
              f"S={s_best:.3f}  γ-range 2%=[{g_lo:.2f},{g_hi:.2f}]")

    print("done")


if __name__ == "__main__":
    main()