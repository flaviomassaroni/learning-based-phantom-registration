"""Genera le figure della presentazione a partire dalla mesh e dai log delle ablation."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
STL = os.path.join(HERE, "..", "phantom_v2", "ModelloPhantom", "Prototipo_Scocca Esterna.stl")

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.transparent": False,
})

TRE_TARGETS = np.array([
    [0.003488, -34.892820, 41.875931], [0.024687, -14.234268, 43.863817],
    [0.034754, 6.424284, 41.211098], [0.022443, 27.082836, 37.631691],
    [0.003493, 47.741388, 37.052715]])
LANDMARKS = np.array([[2.5, -65.0, 82.5], [50.0, -65.0, 77.5], [50.0, 25.0, 50.0], [42.5, 102.5, 27.5]])


def phantom_figures():
    mesh = trimesh.load(STL, force="mesh")
    rng = np.random.default_rng(0)
    pts, _ = trimesh.sample.sample_surface(mesh, 12000, seed=0)
    seed_pt = pts[np.argmax(pts[:, 2] - 0.3 * np.abs(pts[:, 1]))]
    patch = pts[np.linalg.norm(pts - seed_pt, axis=1) < 40]
    patch = patch[rng.choice(len(patch), min(512, len(patch)), replace=False)]
    sparse = pts[rng.choice(len(pts), 25, replace=False)]
    target = pts[rng.choice(len(pts), 1024, replace=False)]

    def view(ax):
        ax.computed_zorder = False
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect(np.ptp(pts, axis=0))
        ax.set_axis_off()

    # 1) contesto: mesh + landmark + target TRE
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*pts[::3].T, s=0.6, c="#c9c8c2", alpha=0.45, depthshade=False)
    ax.scatter(*LANDMARKS.T, s=90, c=ORANGE, edgecolors="white", linewidths=1.5, depthshade=False, label="4 landmark (baseline)")
    ax.scatter(*TRE_TARGETS.T, s=70, marker="D", c=BLUE, edgecolors="white", linewidths=1.5, depthshade=False, label="5 target TRE (midollo)")
    view(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    fig.savefig(os.path.join(OUT, "phantom_landmarks.png"))
    plt.close(fig)

    # 2) modalità di acquisizione
    fig = plt.figure(figsize=(13, 4.6))
    for i, (title, src, col) in enumerate([
        ("Target: digital twin (1024 pt FPS)", target, INK2),
        ("Sweep: patch connessa (512 pt)", patch, BLUE),
        ("Sparse: 20–50 punti sparsi", sparse, ORANGE)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(*pts[::4].T, s=0.4, c="#dddcd7", alpha=0.5, depthshade=False)
        ax.scatter(*src.T, s=3 if len(src) > 100 else 40, c=col,
                   edgecolors="white" if len(src) < 100 else "none", linewidths=0.8, depthshade=False)
        view(ax)
        ax.set_title(title, color=INK, fontsize=14)
    fig.savefig(os.path.join(OUT, "acquisition_modes.png"))
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.1f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=11, color=INK2)


def corr_weight_figure():
    w = ["0", "0.01", "0.03", "0.10"]
    sweep = [14.028, 14.061, 12.761, 13.433]
    sparse = [5.440, 4.061, 3.653, 3.605]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, data, title, best in [(axes[0], sweep, "Sweep (512 pt, r = 20 mm)", 2),
                                  (axes[1], sparse, "Sparse (25 pt)", 3)]:
        cols = [BLUE if i == best else "#a9c6ec" for i in range(4)]
        bars = ax.bar(w, data, color=cols, width=0.6, edgecolor="white", linewidth=2)
        label_bars(ax, bars)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("peso λ_corr della loss KL")
        ax.set_ylabel("TRE media [mm]")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(data) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_corr_weight.png"))
    plt.close(fig)


def acquisition_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    r = [20, 40, 60]
    sw_pose = [14.028, 11.868, 8.213]
    sw_kl = [12.761, 9.126, 6.055]
    n = [20, 25, 35, 50]
    sp_pose = [5.970, 5.440, 4.587, 3.704]
    sp_kl = [4.151, 3.605, 2.753, 1.784]
    for ax, x, a, b, title, xl, klabel in [
            (axes[0], r, sw_pose, sw_kl, "Sweep: raggio della patch", "raggio patch [mm]", "pose + KL (λ = 0.03)"),
            (axes[1], n, sp_pose, sp_kl, "Sparse: numero di punti", "punti acquisiti", "pose + KL (λ = 0.10)")]:
        ax.plot(x, a, "-o", color=BLUE, lw=2, ms=8, mec="white", mew=2, label="solo pose loss")
        ax.plot(x, b, "-o", color=ORANGE, lw=2, ms=8, mec="white", mew=2, label=klabel)
        ax.text(x[-1], a[-1], f"  {a[-1]:.1f}", va="center", color=INK2, fontsize=11)
        ax.text(x[-1], b[-1], f"  {b[-1]:.1f}", va="center", color=INK2, fontsize=11)
        ax.text(x[0], a[0], f"{a[0]:.1f}  ", va="center", ha="right", color=INK2, fontsize=11)
        ax.text(x[0], b[0], f"{b[0]:.1f}  ", va="center", ha="right", color=INK2, fontsize=11)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - (x[-1] - x[0]) * 0.15, x[-1] + (x[-1] - x[0]) * 0.15)
        ax.set_ylim(0, None)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylabel("TRE media [mm]")
        ax.legend(frameon=False, fontsize=11, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_acquisition.png"))
    plt.close(fig)


def final_figure():
    labels = ["Sweep r=60\nDCP + KL", "Sparse n=50\nDCP + KL", "Baseline\n4 landmark"]
    mean = [6.055, 1.784, 0.401]
    p95 = [12.659, 3.476, 0.795]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    b1 = ax.bar(x - 0.18, mean, 0.34, color=[BLUE, ORANGE, GRAY], edgecolor="white", linewidth=2, label="TRE media")
    b2 = ax.bar(x + 0.18, p95, 0.34, color=[BLUE, ORANGE, GRAY], alpha=0.45, edgecolor="white", linewidth=2, label="TRE p95")
    label_bars(ax, b1, "{:.2f}")
    label_bars(ax, b2, "{:.2f}")
    ax.set_xticks(x, labels)
    ax.set_ylabel("TRE [mm]")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 14.5)
    ax.text(0.99, 0.95, "barre piene = media · barre chiare = p95", transform=ax.transAxes,
            ha="right", va="top", color=INK2, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_comparison.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    corr_weight_figure()
    acquisition_figure()
    final_figure()
    phantom_figures()
    print("figure generate in", OUT)
