"""Build the paper's headline figures from noise_sigma sweep CSVs.

Produces:
  - notebooks/validation/output_cross_protein/headline.png     (2x2 primary)
  - notebooks/validation/output_{gb1,trpb}/walsh_spectrum_at_best_sigma.png
  - notebooks/validation/output_cross_protein/ssmula_coverage.png

Run: uv run python scripts/build_noise_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mutascape.validation.figures import save_fig

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CROSS = REPO_ROOT / "notebooks/validation/output_cross_protein"
OUT_GB1 = REPO_ROOT / "notebooks/validation/output_gb1"
OUT_TRPB = REPO_ROOT / "notebooks/validation/output_trpb"

HEADLINE_METRICS = ("walsh_L1", "fitness_W1")
SSMULA_COLS = (
    "ssmula_frac_active",
    "ssmula_frac_optima",
    "ssmula_kurtosis",
    "ssmula_kde_peaks",
    "n_optima",
    "accessible_frac",
)
SSMULA_LABELS = {
    "ssmula_frac_active": "frac_active",
    "ssmula_frac_optima": "frac_local_optima",
    "ssmula_kurtosis": "kurtosis",
    "ssmula_kde_peaks": "kde_peak_count",
    "n_optima": "n_local_optima",
    "accessible_frac": "accessible_fraction",
}


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    real = df[df["sweep"] == "REAL"].iloc[0]
    synth = df[df["sweep"] != "REAL"].copy()
    synth["noise_sigma"] = synth["noise_sigma"].astype(float)
    return synth.sort_values("noise_sigma"), real


def _best_sigma(synth: pd.DataFrame) -> float:
    score = synth["walsh_L1"] + synth["fitness_W1"]
    return float(synth.loc[score.idxmin(), "noise_sigma"])


def fig_headline(gb1: pd.DataFrame, trpb: pd.DataFrame) -> plt.Figure:
    gb1_synth, _ = _split(gb1)
    trpb_synth, _ = _split(trpb)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    palette = sns.color_palette()
    for ri, (name, synth) in enumerate([("GB1", gb1_synth), ("TrpB", trpb_synth)]):
        best = _best_sigma(synth)
        for ci, metric in enumerate(HEADLINE_METRICS):
            ax = axes[ri][ci]
            ax.plot(synth["noise_sigma"], synth[metric], "o-",
                    color=palette[ri], linewidth=2)
            ax.axvline(best, color="grey", linestyle=":", linewidth=1,
                       label=f"best={best:g}")
            best_val = synth.loc[synth["noise_sigma"] == best, metric].values[0]
            ax.scatter([best], [best_val], color="black", zorder=5, s=40)
            ax.set_xlabel("noise_sigma")
            ax.set_ylabel(metric)
            ax.set_title(f"{name}: {metric}")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("Validation metric sensitivity to noise", y=1.02)
    fig.tight_layout()
    return fig


def fig_walsh_spectrum_at_best(df: pd.DataFrame, protein: str) -> plt.Figure:
    synth, real = _split(df)
    best = _best_sigma(synth)
    best_row = synth[synth["noise_sigma"] == best].iloc[0]
    orders = [1, 2, 3, 4]
    real_vals = [real[f"walsh_{k}"] for k in orders]
    synth_vals = [best_row[f"walsh_{k}"] for k in orders]
    x = np.arange(len(orders))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = sns.color_palette()
    ax.bar(x - w / 2, real_vals, w, label=f"real {protein}",
           color=palette[0])
    ax.bar(x + w / 2, synth_vals, w,
           label=f"synth at noise_sigma={best:g}", color=palette[1])
    ax.set_xticks(x)
    ax.set_xticklabels([f"order {k}" for k in orders])
    ax.set_ylabel("Walsh variance fraction")
    ax.set_title(f"{protein} Walsh spectrum at best noise_sigma={best:g}")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_ssmula_coverage(gb1: pd.DataFrame, trpb: pd.DataFrame) -> plt.Figure:
    gb1_synth, gb1_real = _split(gb1)
    trpb_synth, trpb_real = _split(trpb)
    n = len(SSMULA_COLS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.3 * rows), squeeze=False)
    palette = sns.color_palette()
    for i, col in enumerate(SSMULA_COLS):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.plot(gb1_synth["noise_sigma"], gb1_synth[col], "o-",
                color=palette[0], linewidth=2, label="GB1 synth")
        ax.axhline(gb1_real[col], color=palette[0], linestyle="--",
                   linewidth=1, alpha=0.7, label="GB1 real")
        ax.plot(trpb_synth["noise_sigma"], trpb_synth[col], "s-",
                color=palette[1], linewidth=2, label="TrpB synth")
        ax.axhline(trpb_real[col], color=palette[1], linestyle="--",
                   linewidth=1, alpha=0.7, label="TrpB real")
        ax.set_xlabel("noise_sigma")
        ax.set_ylabel(SSMULA_LABELS[col])
        ax.set_title(SSMULA_LABELS[col])
        ax.legend(fontsize=7, ncol=2)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(
        "SSMuLA attribute trajectories vs noise_sigma (real as dashed horizontals)",
        y=1.00,
    )
    fig.tight_layout()
    return fig


def main():
    gb1 = pd.read_csv(OUT_GB1 / "gb1_noise_sweep.csv")
    trpb = pd.read_csv(OUT_TRPB / "trpb_noise_sweep.csv")

    save_fig(fig_headline(gb1, trpb), "headline", OUT_CROSS)
    save_fig(fig_walsh_spectrum_at_best(gb1, "GB1"),
             "walsh_spectrum_at_best_sigma", OUT_GB1)
    save_fig(fig_walsh_spectrum_at_best(trpb, "TrpB"),
             "walsh_spectrum_at_best_sigma", OUT_TRPB)
    save_fig(fig_ssmula_coverage(gb1, trpb), "ssmula_coverage", OUT_CROSS)

    for stale in ("cross_protein_nk.png", "cross_protein_nk.pdf"):
        p = OUT_CROSS / stale
        if p.exists():
            p.unlink()
            print(f"removed {p}")
    print("all figures written.")


if __name__ == "__main__":
    main()
