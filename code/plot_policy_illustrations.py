"""plot_fokker_planck_modes.py
Trajectory probability cloud on an empty 16x16 grid.
Three panels:
  (a) pure geocentric, high beta     tight cigar along the goal diagonal
  (b) pure geocentric, low  beta     wider cigar, sub-optimal steps
  (c) pure egocentric                isotropic spread from the start
Writes figures/fp_modes.svg
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, CREAM = "#1f2a1f", "#fbfaf4"
FOREST, LEAF, MOSS, SAGE = "#1b4332", "#2d6a4f", "#74c69d", "#95d5b2"
FP_CMAP = LinearSegmentedColormap.from_list("fp", [CREAM, SAGE, MOSS, LEAF, FOREST])

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "axes.edgecolor": FOREST, "axes.labelcolor": INK, "axes.titlecolor": FOREST,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": [ "Helvetica Neue", "Helvetica", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "savefig.facecolor": CREAM, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

GRID = 16
GOAL = np.array([GRID - 1, GRID - 1])
ACTS = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]], dtype=np.int32)


def accumulate(beta, T, n, seed, mode):
    """Run n agents for T steps, accumulate occupancy of every cell at every step.
    Agents stop contributing after reaching the goal."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 2), dtype=np.int32)
    arrived = np.zeros(n, dtype=bool)
    H = np.zeros((GRID, GRID), dtype=np.float64)

    for _ in range(T):
        active = ~arrived
        np.add.at(H, (pos[active, 1], pos[active, 0]), 1.0)

        if mode == "geo":
            dx = GOAL[0] - pos[:, 0]
            dy = GOAL[1] - pos[:, 1]
            mag = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-9)
            ux, uy = dx / mag, dy / mag
            sc = beta * (ACTS[None, :, 0] * ux[:, None]
                         + ACTS[None, :, 1] * uy[:, None])
            mx = sc.max(axis=1, keepdims=True)
            ex = np.exp(sc - mx)
            p = ex / ex.sum(axis=1, keepdims=True)
            cum = np.cumsum(p, axis=1)
            u = rng.random((n, 1))
            a = (u > cum).sum(axis=1)
        else:
            a = rng.integers(0, 4, n)

        new_pos = pos + ACTS[a]
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, GRID - 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, GRID - 1)
        pos = new_pos
        arrived |= np.all(pos == GOAL, axis=1)

    return H


def draw(ax, H, title):
    Hn = H / H.max() if H.max() > 0 else H
    im = ax.imshow(Hn, origin="upper", cmap=FP_CMAP, vmin=0, vmax=1,
                   interpolation="nearest", extent=[0, GRID, GRID, 0])
    for i in range(GRID + 1):
        ax.plot([0, GRID], [i, i], color=INK, lw=0.15, alpha=0.22)
        ax.plot([i, i], [0, GRID], color=INK, lw=0.15, alpha=0.22)
    ax.text(0.5, 0.5, "S", ha="center", va="center", color=CREAM,
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.15", fc=LEAF, ec=CREAM, lw=0.7))
    ax.text(GRID - 0.5, GRID - 0.5, "G", ha="center", va="center", color=CREAM,
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.15", fc=FOREST, ec=CREAM, lw=0.7))
    ax.set_title(title, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(True); ax.spines[s].set_color(FOREST)
    return im


def main():
    N = 100_000
    T = 500                      # same time budget for a fair comparison
    H_hi = accumulate(beta=3.0, T=T, n=N, seed=1, mode="geo")
    H_lo = accumulate(beta=0.6, T=T, n=N, seed=2, mode="geo")
    H_eg = accumulate(beta=0.0, T=T, n=N, seed=3, mode="ego")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    ims = [
        draw(axes[0], H_hi, r"(a) Pure geocentric, high $\beta=3.0$"),
        draw(axes[1], H_lo, r"(b) Pure geocentric, low $\beta=0.6$"),
        draw(axes[2], H_eg, r"(c) Pure egocentric"),
    ]
    fig.colorbar(ims[0], ax=axes, orientation="vertical", fraction=0.025,
                 pad=0.015, label="relative occupancy")
    fig.suptitle(f"Probability cloud on an empty $16\\times16$ grid,  $T = {T}$ steps",
                 fontsize=13, color=FOREST, y=1.03)
    fig.savefig("figures/fp_modes.svg", format="svg")
    fig.savefig("figures/fp_modes.png", dpi=600)
    plt.close(fig)
    print("saved figures/fp_modes.svg")


if __name__ == "__main__":
    main()