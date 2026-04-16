"""Publication figures for the validation study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

REAL_STYLE = dict(color="black", linewidth=2, markersize=6)
SYNTH_COLOR = sns.color_palette()[0]


def save_fig(fig: Figure, name: str, output_dir: str | Path = ".") -> None:
    """Save figure as both PDF and PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


def _plot_real_line(ax, depths, values, label="Real GFP"):
    ax.plot(depths, values, "o-", **REAL_STYLE, label=label, zorder=10)


def _get_profile(metrics, key, depths):
    return [metrics[key].get(d, np.nan) for d in depths]


def fig_additive_decay(
    real_metrics: dict,
    synth_metrics_by_alpha: dict[float, dict],
    best_alpha: float,
    depths: list[int] | None = None,
) -> Figure:
    """Additive correlation decay: real vs synthetic."""
    if depths is None:
        depths = list(range(2, 8))

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = sns.color_palette()

    _plot_real_line(ax, depths, _get_profile(real_metrics, "additive_correlation", depths))

    for i, (alpha, synth_m) in enumerate(sorted(synth_metrics_by_alpha.items())):
        synth_y = _get_profile(synth_m, "additive_correlation", depths)
        is_best = abs(alpha - best_alpha) < 0.01
        ax.plot(
            depths,
            synth_y,
            "s--" if is_best else "--",
            color=palette[i % len(palette)],
            alpha=1.0 if is_best else 0.3,
            linewidth=2 if is_best else 1,
            label=f"\u03b1={alpha:.1f}" + (" (best)" if is_best else ""),
        )

    ax.set_xlabel("Mutation depth")
    ax.set_ylabel("Pearson r (additive vs actual)")
    ax.set_title("Additive correlation decay")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def fig_sign_epistasis(
    real_metrics: dict,
    synth_metrics_by_alpha: dict[float, dict],
    best_alpha: float,
    depths: list[int] | None = None,
) -> Figure:
    """Sign epistasis fraction: real vs synthetic."""
    if depths is None:
        depths = list(range(2, 8))

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = sns.color_palette()

    _plot_real_line(ax, depths, _get_profile(real_metrics, "sign_epistasis", depths))

    for i, (alpha, synth_m) in enumerate(sorted(synth_metrics_by_alpha.items())):
        synth_y = _get_profile(synth_m, "sign_epistasis", depths)
        is_best = abs(alpha - best_alpha) < 0.01
        ax.plot(
            depths,
            synth_y,
            "s--" if is_best else "--",
            color=palette[i % len(palette)],
            alpha=1.0 if is_best else 0.3,
            linewidth=2 if is_best else 1,
            label=f"\u03b1={alpha:.1f}" + (" (best)" if is_best else ""),
        )

    ax.set_xlabel("Mutation depth")
    ax.set_ylabel("Fraction with sign epistasis")
    ax.set_title("Sign epistasis by depth")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def _hist_pair(ax, real_dev, synth_dev):
    """Plot overlaid histograms for real vs synthetic deviations."""
    if len(real_dev) > 0:
        ax.hist(
            real_dev,
            bins=40,
            alpha=0.5,
            color="black",
            density=True,
            label="Real",
        )
    if len(synth_dev) > 0:
        ax.hist(
            synth_dev,
            bins=40,
            alpha=0.5,
            color=SYNTH_COLOR,
            density=True,
            label="Synthetic",
        )


def fig_deviation_distributions(
    real_metrics: dict,
    synth_metrics: dict,
    depths: list[int] | None = None,
) -> Figure:
    """Side-by-side histograms of epistatic deviation distributions."""
    if depths is None:
        depths = [2, 4, 6]

    fig, axes = plt.subplots(1, len(depths), figsize=(4 * len(depths), 3.5))
    if len(depths) == 1:
        axes = [axes]

    for ax, d in zip(axes, depths):
        real_dev = real_metrics["deviation_distributions"].get(d, np.array([]))
        synth_dev = synth_metrics["deviation_distributions"].get(d, np.array([]))
        _hist_pair(ax, real_dev, synth_dev)
        ax.set_xlabel("Epistatic deviation")
        ax.set_ylabel("Density")
        ax.set_title(f"Depth {d}")
        ax.legend(fontsize=8)

    fig.suptitle("Epistatic deviation distributions", y=1.02)
    fig.tight_layout()
    return fig


def fig_alpha_calibration(sweep_df) -> Figure:
    """Composite validation score vs alpha."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        sweep_df["alpha"],
        sweep_df["composite_score"],
        "o-",
        color=SYNTH_COLOR,
        linewidth=2,
    )

    best_idx = sweep_df["composite_score"].idxmin()
    best_alpha = float(sweep_df.loc[best_idx, "alpha"])
    best_score = float(sweep_df.loc[best_idx, "composite_score"])
    ax.scatter(
        [best_alpha],
        [best_score],
        color="red",
        s=100,
        zorder=10,
        label=f"Best: \u03b1={best_alpha:.2f}",
    )

    ax.set_xlabel("Alpha (epistatic strength)")
    ax.set_ylabel("Composite validation score (lower = better)")
    ax.set_title("Alpha calibration")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_validation_summary(
    real_metrics: dict,
    synth_metrics: dict,
    best_alpha: float,
    depths: list[int] | None = None,
) -> Figure:
    """2x3 summary figure for the paper."""
    if depths is None:
        depths = list(range(2, 8))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    palette = sns.color_palette()
    synth_label = f"mutascape (\u03b1={best_alpha:.2f})"

    # 1a: Additive correlation
    ax = axes[0, 0]
    real_y = _get_profile(real_metrics, "additive_correlation", depths)
    synth_y = _get_profile(synth_metrics, "additive_correlation", depths)
    ax.plot(depths, real_y, "o-", **REAL_STYLE, label="Real GFP")
    ax.plot(depths, synth_y, "s--", color=palette[0], lw=2, label=synth_label)
    ax.set_ylabel("Pearson r")
    ax.set_title("Additive correlation")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # 1b: Sign epistasis
    ax = axes[0, 1]
    real_y = _get_profile(real_metrics, "sign_epistasis", depths)
    synth_y = _get_profile(synth_metrics, "sign_epistasis", depths)
    ax.plot(depths, real_y, "o-", **REAL_STYLE, label="Real GFP")
    ax.plot(depths, synth_y, "s--", color=palette[0], lw=2, label="mutascape")
    ax.set_ylabel("Fraction")
    ax.set_title("Sign epistasis")
    ax.set_ylim(0, 1.05)

    # 1c: MAE
    ax = axes[0, 2]
    real_y = [real_metrics["deviation_stats"].get(d, {}).get("mae", np.nan) for d in depths]
    synth_y = [synth_metrics["deviation_stats"].get(d, {}).get("mae", np.nan) for d in depths]
    ax.plot(depths, real_y, "o-", **REAL_STYLE, label="Real GFP")
    ax.plot(depths, synth_y, "s--", color=palette[0], lw=2, label="mutascape")
    ax.set_ylabel("MAE")
    ax.set_title("Epistatic deviation magnitude")

    # Row 2: deviation distributions at depths 2, 4, 6
    for i, d in enumerate([2, 4, 6]):
        ax = axes[1, i]
        real_dev = real_metrics["deviation_distributions"].get(d, np.array([]))
        synth_dev = synth_metrics["deviation_distributions"].get(d, np.array([]))
        _hist_pair(ax, real_dev, synth_dev)
        ax.set_xlabel("Epistatic deviation")
        ax.set_ylabel("Density")
        ax.set_title(f"Depth {d}")
        if i == 0:
            ax.legend(fontsize=8)

    for ax in axes[0, :]:
        ax.set_xlabel("Mutation depth")

    title = f"mutascape vs GFP combinatorial data (\u03b1={best_alpha:.2f})"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
