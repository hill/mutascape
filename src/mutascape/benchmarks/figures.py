"""Publication figures for posterior landscape evaluation."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_metric_fingerprint(
    scores: pd.DataFrame,
    *,
    value: str = "value",
    aggregate: str = "median",
) -> Figure:
    """Plot a model-by-metric matrix across benchmark landscapes."""
    required = {"model", "metric", value}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"scores are missing columns: {', '.join(missing)}")
    table = scores.pivot_table(index="metric", columns="model", values=value, aggfunc=aggregate)
    figure, axis = plt.subplots(figsize=(1.5 + 1.4 * len(table.columns), 1.5 + 0.45 * len(table)))
    sns.heatmap(table, cmap="vlag_r", center=0.0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Landscape fingerprint")
    axis.set_xlabel("Model")
    axis.set_ylabel("Metric")
    figure.tight_layout()
    return figure


def plot_calibration_curve(calibration: pd.DataFrame) -> Figure:
    """Compare nominal and measured posterior interval coverage."""
    required = {"nominal", "coverage"}
    missing = sorted(required - set(calibration.columns))
    if missing:
        raise ValueError(f"calibration is missing columns: {', '.join(missing)}")
    figure, axis = plt.subplots(figsize=(5.0, 4.3))
    if "model" in calibration:
        sns.lineplot(
            calibration,
            x="nominal",
            y="coverage",
            hue="model",
            marker="o",
            ax=axis,
        )
    else:
        axis.plot(calibration["nominal"], calibration["coverage"], marker="o")
    axis.plot([0, 1], [0, 1], color="0.4", linestyle=":", label="Ideal")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Nominal interval", ylabel="Observed coverage")
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    return figure


def plot_completion_predictions(predictions: pd.DataFrame) -> Figure:
    """Show held-out fitness predictions and errors by mutation depth."""
    required = {"fitness", "prediction", "depth"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions are missing columns: {', '.join(missing)}")

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    scatter = axes[0].scatter(
        predictions["fitness"],
        predictions["prediction"],
        c=predictions["depth"],
        cmap="viridis",
        alpha=0.55,
        s=14,
    )
    low = float(min(predictions["fitness"].min(), predictions["prediction"].min()))
    high = float(max(predictions["fitness"].max(), predictions["prediction"].max()))
    axes[0].plot([low, high], [low, high], color="0.4", linestyle=":")
    axes[0].set(xlabel="Measured fitness", ylabel="Posterior mean")
    figure.colorbar(scatter, ax=axes[0], label="Mutation depth")

    errors = predictions.assign(
        absolute_error=lambda frame: abs(frame["prediction"] - frame["fitness"])
    )
    sns.boxplot(errors, x="depth", y="absolute_error", color="#6baed6", ax=axes[1])
    axes[1].set(xlabel="Mutation depth", ylabel="Absolute error")
    figure.tight_layout()
    return figure


def plot_posterior_rank_histogram(predictions: pd.DataFrame, *, bins: int = 10) -> Figure:
    """Plot posterior ranks, which should approach a uniform distribution."""
    if "posterior_rank" not in predictions:
        raise ValueError("predictions are missing the posterior_rank column")
    figure, axis = plt.subplots(figsize=(5.0, 3.8))
    axis.hist(predictions["posterior_rank"], bins=bins, range=(0, 1), color="#756bb1")
    expected = len(predictions) / bins
    axis.axhline(expected, color="0.4", linestyle=":")
    axis.set(xlabel="Posterior rank", ylabel="Variants", xlim=(0, 1))
    figure.tight_layout()
    return figure


def plot_optimizer_rank_transfer(results: pd.DataFrame) -> Figure:
    """Plot optimiser ranks on real and synthetic landscape evaluations."""
    required = {"landscape", "oracle", "optimiser", "score"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"results are missing columns: {', '.join(missing)}")
    ranked = results.copy()
    ranked["rank"] = ranked.groupby(["landscape", "oracle"])["score"].rank(
        ascending=False, method="average"
    )
    real = ranked[ranked["oracle"] == "real"][["landscape", "optimiser", "rank"]]
    real = real.rename(columns={"rank": "real_rank"})
    synthetic = ranked[ranked["oracle"] == "synthetic"]
    paired = synthetic.merge(real, on=["landscape", "optimiser"], validate="many_to_one")
    table = paired.pivot_table(
        index="optimiser", columns="landscape", values="rank", aggfunc="mean"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    sns.heatmap(table, cmap="viridis_r", annot=True, fmt=".1f", ax=axes[0])
    axes[0].set_title("Synthetic rank")
    axes[0].set(xlabel="Landscape", ylabel="Optimiser")
    axes[1].scatter(paired["real_rank"], paired["rank"], alpha=0.6)
    maximum = float(max(paired["real_rank"].max(), paired["rank"].max()))
    axes[1].plot([1, maximum], [1, maximum], color="0.4", linestyle=":")
    axes[1].set(xlabel="Real rank", ylabel="Synthetic rank")
    figure.tight_layout()
    return figure
