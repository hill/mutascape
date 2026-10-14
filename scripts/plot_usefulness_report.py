"""Plot both early and late budgets from the controlled benchmark experiment."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.build_quantile_manuscript import BLUE, GREEN, STYLE

ROOT = Path("results/controlled_benchmarks_20260930")


def main(root=ROOT, budgets=(32, 96)):
    geometry = pd.read_csv(root / "geometry.csv")
    contrasts = pd.read_csv(root / "paired_contrasts.csv")
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for protein, colour in zip(("CreiLOV", "eqFP611"), (BLUE, GREEN)):
        spectrum = geometry.loc[geometry.protein == protein].groupby("strength").pair_variance
        means, lower, upper = spectrum.mean(), spectrum.min(), spectrum.max()
        axes[0].plot(means.index, 100 * means, "o-", color=colour, label=protein)
        axes[0].fill_between(means.index, 100 * lower, 100 * upper, color=colour, alpha=0.15)
        for ax, budget in zip(axes[1:], budgets):
            values = contrasts.loc[(contrasts.protein == protein) & (contrasts.budget == budget)]
            ax.plot(values.strength, 100 * values.difference, "o-", color=colour)
            ax.fill_between(
                values.strength, 100 * values.low, 100 * values.high, color=colour, alpha=0.15
            )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Pair strength")
        ax.set_xticks([0, 0.3, 1, 3])
    for ax in axes[1:]:
        ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
        ax.set_ylim(-4.6, 1.1)
        ax.set_ylabel("Pairwise minus additive regret\n(percentage points)")
    axes[0].set_ylabel("Variance from pair interactions (%)")
    axes[0].set_title("(a) Controlled interaction variance")
    axes[1].set_title(f"(b) {budgets[0]} evaluations")
    axes[2].set_title(f"(c) {budgets[1]} evaluations")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(root / "controlled_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "controlled_comparison.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--budgets", type=int, nargs=2, default=(32, 96))
    args = parser.parse_args()
    main(args.root, args.budgets)
