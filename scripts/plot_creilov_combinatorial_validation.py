"""Plot the Chen 2023 CreiLOV combinatorial validation: MSA-coupling scale vs real."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "notebooks/validation/output_creilov_combinatorial"
SUMMARY_CSV = OUT / "creilov_combinatorial_summary.csv"


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    configs = [
        ("v1_no_couplings", "No pairwise couplings", "#d62728"),
        ("v2_scale_0.10", "MSA couplings, scale = 0.10", "#ff7f0e"),
        ("v2_scale_0.30", "MSA couplings, scale = 0.30", "#bcbd22"),
        ("v2_scale_0.50", "MSA couplings, scale = 0.50", "#17becf"),
        ("v2_scale_1.00", "MSA couplings, scale = 1.00", "#2ca02c"),
    ]
    depths = sorted(df["depth"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), tight_layout=True)

    # Pearson correlation by depth
    ax = axes[0]
    for name, label, color in configs:
        sub = df[df["config"] == name].sort_values("depth")
        ax.plot(sub["depth"], sub["pearson_to_real"], "o-", color=color, label=label, alpha=0.85)
    ax.set_xlabel("Mutation depth")
    ax.set_ylabel("Pearson correlation (Mutascape pred vs real)")
    ax.set_title("CreiLOV combinatorial -- depth-resolved prediction")
    ax.set_xticks(depths)
    ax.set_ylim(-0.1, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Additive correlation: real vs Mutascape, compared to the additive baseline.
    ax = axes[1]
    real_add = df[df["config"] == "v1_no_couplings"].sort_values("depth")
    ax.plot(
        real_add["depth"],
        real_add["real_additive_correlation"],
        "k-",
        lw=2.2,
        marker="s",
        label="real ↔ additive (target)",
    )
    for name, label, color in configs:
        sub = df[df["config"] == name].sort_values("depth")
        ax.plot(
            sub["depth"],
            sub["synth_additive_correlation"],
            "--",
            marker="o",
            color=color,
            label=f"{label} ↔ additive",
            alpha=0.85,
        )
    ax.set_xlabel("Mutation depth")
    ax.set_ylabel("Correlation to Mutascape additive baseline")
    ax.set_title("CreiLOV combinatorial -- over-additivity diagnostic")
    ax.set_xticks(depths)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.savefig(OUT / "creilov_combinatorial_validation.pdf")
    fig.savefig(OUT / "creilov_combinatorial_validation.png", dpi=160)
    print(f"wrote {OUT}/creilov_combinatorial_validation.pdf")


if __name__ == "__main__":
    main()
