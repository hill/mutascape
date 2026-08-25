"""Export aggregate completion figures for the Mutascape manuscript."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mutascape.benchmarks.figures import (
    plot_calibration_curve,
    plot_posterior_rank_histogram,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _plot_predictive_comparison(metrics: pd.DataFrame):
    selected = metrics[metrics["metric"].isin(["spearman", "ndcg", "mae", "rmse"])]
    selected = selected[selected["model"].isin(["posterior", "ridge_additive"])]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for metric, axis in zip(["spearman", "ndcg", "mae", "rmse"], axes.flat):
        subset = selected[selected["metric"] == metric]
        sns.boxplot(
            subset,
            x="model",
            y="value",
            hue="model",
            order=["ridge_additive", "posterior"],
            palette=["#4c78a8", "#e45756"],
            legend=False,
            ax=axis,
        )
        sns.stripplot(
            subset,
            x="model",
            y="value",
            order=["ridge_additive", "posterior"],
            color="0.2",
            alpha=0.35,
            size=2.5,
            ax=axis,
        )
        axis.set_title(metric.upper())
        axis.set_xlabel("")
        axis.set_xticks([0, 1], ["Additive", "Posterior"])
        axis.set_ylabel("Landscape score")
    figure.tight_layout()
    return figure


def _plot_distribution_tests(metrics: pd.DataFrame):
    selected = metrics[
        (metrics["model"] == "posterior") & metrics["metric"].isin(["c2st_auc", "mmd_squared"])
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for metric, label, axis in [
        ("c2st_auc", "C2ST AUC", axes[0]),
        ("mmd_squared", "MMD squared", axes[1]),
    ]:
        values = selected.loc[selected["metric"] == metric, "value"]
        sns.boxplot(y=values, color="#72b7b2", width=0.35, ax=axis)
        sns.stripplot(y=values, color="0.2", alpha=0.5, size=3, ax=axis)
        axis.set_ylabel(label)
        axis.set_xlabel("30 held-out landscapes")
    axes[0].axhline(0.5, color="0.4", linestyle=":", label="Chance")
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="0.4", linestyle=":")
    figure.tight_layout()
    return figure


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.benchmark / "metrics.csv")
    calibration = pd.read_csv(args.benchmark / "calibration.csv")
    predictions = pd.read_parquet(args.benchmark / "predictions.parquet")

    figures = {
        "predictive_comparison.pdf": _plot_predictive_comparison(metrics),
        "calibration.pdf": plot_calibration_curve(calibration),
        "distribution_tests.pdf": _plot_distribution_tests(metrics),
        "posterior_ranks.pdf": plot_posterior_rank_histogram(predictions),
    }
    for filename, figure in figures.items():
        figure.savefig(args.output / filename, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
