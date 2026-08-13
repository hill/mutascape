"""Plot the fitness-matched Sarkisyan validation results.

Produces a depth-resolved additive-correlation and sign-epistasis figure
showing v1 vs v2-at-various-scales against real Sarkisyan multi-mutant
data. This is the headline whole-protein validation figure for the
mutascape paper / thesis combinatorial chapter.

Reads:
  notebooks/validation/output_gfp_v2/gfp_v2_validation_matched_summary.csv
Writes:
  notebooks/validation/output_gfp_v2/gfp_v2_matched_validation.pdf / .png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "notebooks/validation/output_gfp_v2"
SUMMARY_CSV = OUT / "gfp_v2_validation_matched_summary.csv"


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

    real = df.drop_duplicates(subset="depth")[
        ["depth", "real_additive_correlation", "real_sign_epistasis"]
    ]
    real = real.sort_values("depth")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), tight_layout=True)

    # Additive correlation
    ax = axes[0]
    ax.plot(
        real["depth"],
        real["real_additive_correlation"],
        "k-",
        lw=2.2,
        marker="s",
        label="Sarkisyan (real)",
    )
    for name, label, color in configs:
        sub = df[df["config"] == name].sort_values("depth")
        ax.plot(
            sub["depth"],
            sub["matched_additive_correlation"],
            "--",
            marker="o",
            color=color,
            label=label,
            alpha=0.85,
        )
    ax.set_xlabel("Mutation depth")
    ax.set_ylabel("Additive correlation (matched)")
    ax.set_title("GFP whole-protein, fitness-matched")
    ax.set_xticks(depths)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Sign epistasis
    ax = axes[1]
    ax.plot(
        real["depth"],
        real["real_sign_epistasis"],
        "k-",
        lw=2.2,
        marker="s",
        label="Sarkisyan (real)",
    )
    for name, label, color in configs:
        sub = df[df["config"] == name].sort_values("depth")
        ax.plot(
            sub["depth"],
            sub["matched_sign_epistasis"],
            "--",
            marker="o",
            color=color,
            label=label,
            alpha=0.85,
        )
    ax.set_xlabel("Mutation depth")
    ax.set_ylabel("Sign-epistasis fraction (matched)")
    ax.set_title("GFP whole-protein, fitness-matched")
    ax.set_xticks(depths)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.savefig(OUT / "gfp_msa_matched_validation.pdf")
    fig.savefig(OUT / "gfp_msa_matched_validation.png", dpi=160)
    print(f"wrote {OUT}/gfp_msa_matched_validation.pdf")
    print(f"wrote {OUT}/gfp_msa_matched_validation.png")


if __name__ == "__main__":
    main()
