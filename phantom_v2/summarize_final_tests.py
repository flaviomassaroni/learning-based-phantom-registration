#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Riassume i log prodotti da run_final_tests.sh:
  - final_tests_summary.md: tabelle di test held-out, bacino e rumore;
  - final_tests_basin.png: TRE e success rate al crescere del disallineamento;
  - final_tests_failures.png: casi peggiori (GT vs posa predetta).
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = "checkpoints"
CONFIGS = [
    ("Sweep r=60, solo pose", "acq_sweep_r60_dcp_seed42"),
    ("Sweep r=60, pose + KL 0.03", "acq_sweep_r60_corr003_seed42"),
    ("Sparse n=50, solo pose", "acq_sparse_n50_dcp_seed42"),
    ("Sparse n=50, pose + KL 0.10", "acq_sparse_n50_corr010_seed42"),
]
KL_CONFIGS = [CONFIGS[1], CONFIGS[3]]
LEVELS = [("0.17", "10"), ("0.35", "20"), ("0.52", "30"), ("0.79", "50")]
NOISE = ["0.0", "0.3", "0.6"]


def parse(eval_name):
    """Estrae dall'ultimo blocco di eval.log le statistiche e i success rate."""
    path = os.path.join(CKPT, eval_name, "eval.log")

    if not os.path.isfile(path):
        return None

    text = open(path).read().split("Checkpoint:")[-1]
    out = {}

    for name, mean, median, p95, mx in re.findall(
        r"^(\w+): mean=([\d.]+), std=[\d.]+, median=([\d.]+), "
        r"p95=([\d.]+), max=([\d.]+)",
        text,
        re.M,
    ):
        out[name] = dict(
            mean=float(mean), median=float(median),
            p95=float(p95), max=float(mx),
        )

    for method, rate in re.findall(
        r"^(\w+)_success_rate \(TRE < [\d.]+ mm\): ([\d.]+)%", text, re.M
    ):
        out[f"{method}_success"] = float(rate)

    return out


def fmt(res, method):
    if res is None or f"{method}_tre_mean_mm" not in res:
        return "| n.d. | n.d. | n.d. | n.d. | n.d. |"

    rot = res[f"{method}_rotation_deg"]["mean"]
    tr = res[f"{method}_translation_mm"]["mean"]
    tre = res[f"{method}_tre_mean_mm"]
    succ = res.get(f"{method}_success", float("nan"))
    return (
        f"| {rot:.2f} | {tr:.2f} | {tre['mean']:.2f} | "
        f"{tre['p95']:.2f} | {succ:.1f}% |"
    )


def main():
    lines = ["# Test finali (seed 314159)", ""]

    lines += [
        "## Test held-out (2000 campioni, ±0.17 rad, ±10 mm, σ = 0.3 mm)",
        "",
        "| Configurazione | Rot. [°] | Trasl. [mm] | TRE media [mm] "
        "| TRE p95 [mm] | Success (TRE < 5 mm) |",
        "|---|---|---|---|---|---|",
    ]
    baseline = None

    for label, exp in CONFIGS:
        res = parse(f"test_{exp}")
        lines.append(f"| {label} " + fmt(res, "dcp"))

        if res is not None and baseline is None and "baseline_tre_mean_mm" in res:
            baseline = res

    lines.append("| Baseline 4 landmark " + fmt(baseline, "baseline"))

    lines += [
        "",
        "## Bacino di convergenza (1000 campioni per livello)",
        "",
        "| Configurazione | Disallineamento max per asse | TRE media [mm] "
        "| TRE p95 [mm] | Success DCP | Success baseline |",
        "|---|---|---|---|---|---|",
    ]
    basin = {}

    for label, exp in KL_CONFIGS:
        for rot, trans in LEVELS:
            res = parse(f"basin_{exp}_rot{rot}_trans{trans}")
            deg = np.degrees(float(rot))
            level = f"±{deg:.0f}°, ±{trans} mm"

            if res is None:
                lines.append(f"| {label} | {level} | n.d. | n.d. | n.d. | n.d. |")
                continue

            tre = res["dcp_tre_mean_mm"]
            basin.setdefault(label, []).append(
                (deg, tre["mean"], res.get("dcp_success", np.nan))
            )
            lines.append(
                f"| {label} | {level} | {tre['mean']:.2f} | {tre['p95']:.2f} | "
                f"{res.get('dcp_success', float('nan')):.1f}% | "
                f"{res.get('baseline_success', float('nan')):.1f}% |"
            )

    lines += [
        "",
        "## Robustezza al rumore (σ per coordinata)",
        "",
        "| Configurazione | σ [mm] | TRE media DCP [mm] | Success DCP "
        "| TRE media baseline [mm] |",
        "|---|---|---|---|---|",
    ]

    for label, exp in KL_CONFIGS:
        for sigma in NOISE:
            name = (
                f"test_{exp}" if sigma == "0.3"
                else f"noise_{exp}_sigma{sigma}"
            )
            res = parse(name)

            if res is None:
                lines.append(f"| {label} | {sigma} | n.d. | n.d. | n.d. |")
                continue

            lines.append(
                f"| {label} | {sigma} | {res['dcp_tre_mean_mm']['mean']:.2f} | "
                f"{res.get('dcp_success', float('nan')):.1f}% | "
                f"{res['baseline_tre_mean_mm']['mean']:.2f} |"
            )

    with open("final_tests_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nScritto final_tests_summary.md")

    if basin:
        plot_basin(basin)

    plot_failures()


def plot_basin(basin):
    colors = ["#2a78d6", "#eb6834"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    for (label, rows), color in zip(basin.items(), colors):
        deg, tre, succ = zip(*rows)
        axes[0].plot(deg, tre, "-o", color=color, lw=2, ms=8,
                     mec="white", mew=2, label=label)
        axes[1].plot(deg, succ, "-o", color=color, lw=2, ms=8,
                     mec="white", mew=2, label=label)

    axes[0].set_ylabel("TRE media [mm]")
    axes[1].set_ylabel("success rate (TRE < 5 mm) [%]")
    axes[1].set_ylim(0, 105)

    for ax in axes:
        ax.set_xlabel("rotazione max per asse [°] (traslazione 10–50 mm)")
        ax.grid(color="#e4e3df")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig("final_tests_basin.png", dpi=200)
    plt.close(fig)
    print("Scritto final_tests_basin.png")


def plot_failures():
    """Per i 3 casi peggiori di ogni modello: target, source con GT e con posa predetta."""
    panels = []

    for label, exp in KL_CONFIGS:
        path = os.path.join(CKPT, f"test_{exp}", "worst_cases.npz")

        if os.path.isfile(path):
            data = np.load(path)

            for k in range(3):
                if f"case{k}_src_mm" in data:
                    panels.append((label, k, data))

    if not panels:
        return

    fig = plt.figure(figsize=(4.2 * 3, 4.2 * ((len(panels) + 2) // 3)))

    for n, (label, k, d) in enumerate(panels):
        src = d[f"case{k}_src_mm"]
        tgt = d[f"case{k}_tgt_mm"]
        gt = src @ d[f"case{k}_R_gt"].T + d[f"case{k}_t_gt_mm"]
        pred = src @ d[f"case{k}_R_pred"].T + d[f"case{k}_t_pred_mm"]

        ax = fig.add_subplot((len(panels) + 2) // 3, 3, n + 1, projection="3d")
        ax.computed_zorder = False
        ax.scatter(*tgt.T, s=1, c="#c9c8c2")
        ax.scatter(*gt.T, s=6, c="#1baf7a", label="posa vera")
        ax.scatter(*pred.T, s=6, c="#e34948", label="posa DCP")
        ax.set_title(
            f"{label}\nidx {int(d[f'case{k}_index'])}, "
            f"TRE {float(d[f'case{k}_tre_mm']):.1f} mm",
            fontsize=10,
        )
        ax.set_axis_off()

        if n == 0:
            ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig("final_tests_failures.png", dpi=200)
    plt.close(fig)
    print("Scritto final_tests_failures.png")


if __name__ == "__main__":
    main()
