"""Plot the fixed pilot assessment without selecting a favourable draw."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("results/creilov_pilot_readout_burden_confirmation")
    metrics = pd.read_csv(root / "metrics.csv")
    samples = [pd.read_csv(root / f"samples_{seed}.csv") for seed in (20260920, 20260921, 20260922)]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colours = {"original": "#a1a1aa", "pilot": "#447a9c"}
    for i, (model, label) in enumerate((("original", "Default"), ("pilot", "48-variant pilot"))):
        for offset, classifier, marker in ((-0.1, "logistic", "o"), (0.1, "histogram", "x")):
            values = metrics.loc[
                metrics.arm.str.startswith(model) & (metrics.classifier == classifier), "auc"
            ]
            axes[0].scatter(
                np.full(len(values), i + offset),
                values,
                color=colours[model],
                marker=marker,
                s=28,
                alpha=0.7,
                label=classifier if i == 1 else None,
            )
    axes[0].axhline(0.5, color="0.6", linestyle="--", linewidth=0.8)
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Default", "48-variant pilot"],
        ylim=(0.45, 1.025),
        ylabel="Held-out C2ST AUC",
        title="Calibration reduces discrimination",
    )
    axes[0].legend(frameon=False, loc="lower left")
    real = np.concatenate([frame.measured for frame in samples])
    original = np.concatenate(
        [frame.filter(like="original_").to_numpy().ravel() for frame in samples]
    )
    pilot = np.concatenate([frame.filter(like="pilot_").to_numpy().ravel() for frame in samples])
    edges = np.linspace(
        min(real.min(), original.min(), pilot.min()),
        max(real.max(), original.max(), pilot.max()),
        55,
    )
    for values, label, colour in (
        (real, "Measured", "#27272a"),
        (original, "Default", colours["original"]),
        (pilot, "48-variant pilot", colours["pilot"]),
    ):
        axes[1].hist(
            values,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=colour,
            label=label,
        )
    axes[1].set(
        xlabel="Log10 fluorescence relative to WT",
        ylabel="Density",
        title="Raw fitness values overlap more",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(root / "diagnosis.png", dpi=180)
    figure.savefig(root / "diagnosis.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
