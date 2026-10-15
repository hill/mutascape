"""Publication figures for the validation study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from mutascape.nonlinearity import GlobalEpistasisSigmoid

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


def _get_profile(metrics, key, depths):
    return [metrics[key].get(d, np.nan) for d in depths]


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
    protein_name: str = "real",
    dist_depths: list[int] | None = None,
    synth_label: str | None = None,
) -> Figure:
    """2x3 summary figure for the paper."""
    if depths is None:
        depths = list(range(2, 8))
    if dist_depths is None:
        dist_depths = [d for d in (2, 4, 6) if d in depths] or depths[:3]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    palette = sns.color_palette()
    real_label = f"Real {protein_name}"
    if synth_label is None:
        synth_label = f"mutascape (\u03b1={best_alpha:.2f})"

    # 1a: Additive correlation
    ax = axes[0, 0]
    real_y = _get_profile(real_metrics, "additive_correlation", depths)
    synth_y = _get_profile(synth_metrics, "additive_correlation", depths)
    ax.plot(depths, real_y, "o-", **REAL_STYLE, label=real_label)
    ax.plot(depths, synth_y, "s--", color=palette[0], lw=2, label=synth_label)
    ax.set_ylabel("Pearson r")
    ax.set_title("Additive correlation")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # 1b: Sign epistasis
    ax = axes[0, 1]
    real_y = _get_profile(real_metrics, "sign_epistasis", depths)
    synth_y = _get_profile(synth_metrics, "sign_epistasis", depths)
    ax.plot(depths, real_y, "o-", **REAL_STYLE, label=real_label)
    ax.plot(depths, synth_y, "s--", color=palette[0], lw=2, label="mutascape")
    ax.set_ylabel("Fraction")
    ax.set_title("Sign epistasis")
    ax.set_ylim(0, 1.05)

    # 1c: MAE
    ax = axes[0, 2]
    real_y = [real_metrics["deviation_stats"].get(d, {}).get("mae", np.nan) for d in depths]
    synth_y = [synth_metrics["deviation_stats"].get(d, {}).get("mae", np.nan) for d in depths]
    ax.plot(depths, real_y, "o-", **REAL_STYLE, label=real_label)
    ax.plot(depths, synth_y, "s--", color=palette[0], lw=2, label="mutascape")
    ax.set_ylabel("MAE")
    ax.set_title("Epistatic deviation magnitude")

    # Row 2: deviation distributions
    n_dist_cols = axes.shape[1]
    for i in range(n_dist_cols):
        ax = axes[1, i]
        if i < len(dist_depths):
            d = dist_depths[i]
            real_dev = real_metrics["deviation_distributions"].get(d, np.array([]))
            synth_dev = synth_metrics["deviation_distributions"].get(d, np.array([]))
            _hist_pair(ax, real_dev, synth_dev)
            ax.set_xlabel("Epistatic deviation")
            ax.set_ylabel("Density")
            ax.set_title(f"Depth {d}")
            if i == 0:
                ax.legend(fontsize=8)
        else:
            ax.axis("off")

    for ax in axes[0, :]:
        ax.set_xlabel("Mutation depth")

    title = f"Mutascape vs {protein_name} real data (\u03b1={best_alpha:.2f})"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def fig_sigmoid_curves(
    sigmoids_by_label: dict[str, GlobalEpistasisSigmoid],
    observed_singles: np.ndarray,
    z_range: tuple[float, float] | None = None,
) -> Figure:
    """Overlay fitted g(z) curves with the observed single-site fitness rug.

    `z_range` is the pre-image axis (stability-equivalent units). If None,
    inferred from each sigmoid's scale so the saturation knees are visible.
    The rug along the left edge shows the distribution of observed single-
    site fitness effects, so the reader can see where the fit is actually
    constrained versus where it's extrapolating.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = sns.color_palette()

    if z_range is None:
        z_span = 4.0 * max(g.scale for g in sigmoids_by_label.values())
        z_range = (-z_span, z_span)
    zs = np.linspace(z_range[0], z_range[1], 400)

    for i, (label, g) in enumerate(sigmoids_by_label.items()):
        color = palette[i % len(palette)]
        ax.plot(zs, g.forward(zs), linewidth=2, color=color, label=label)
        ax.axhline(g.floor, color=color, linestyle=":", alpha=0.3)
        ax.axhline(g.ceiling, color=color, linestyle=":", alpha=0.3)

    rug_width = 0.02 * (z_range[1] - z_range[0])
    for y in np.asarray(observed_singles).ravel():
        ax.plot(
            [z_range[0], z_range[0] + rug_width], [y, y], color="black", alpha=0.35, linewidth=0.5
        )

    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0.0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"pre-image $z = g^{-1}(f)$")
    ax.set_ylabel("fitness $f$")
    ax.set_title("Fitted global-epistasis functions")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def fig_walsh_spectrum(
    real_walsh: dict[int, float],
    synth_walsh_by_label: dict[str, dict[int, float]],
) -> Figure:
    """Grouped bar chart of Walsh variance fractions: real vs configs."""
    orders = sorted(real_walsh.keys())
    configs = ["real"] + list(synth_walsh_by_label.keys())
    n = len(configs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = ["black"] + list(sns.color_palette()[: n - 1])
    width = 0.8 / n
    x = np.arange(len(orders))

    for i, label in enumerate(configs):
        values = (
            [real_walsh[o] for o in orders]
            if label == "real"
            else [synth_walsh_by_label[label][o] for o in orders]
        )
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, values, width, color=palette[i], label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"order {o}" for o in orders])
    ax.set_ylabel("variance fraction")
    ax.set_title("Walsh-Hadamard variance decomposition")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_fitness_distribution(
    real_values: np.ndarray,
    synth_values_by_label: dict[str, np.ndarray],
    bins: int = 60,
) -> Figure:
    """Overlay density histograms of fitness: real vs synthetic configs."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = sns.color_palette()

    real_finite = np.asarray(real_values)[np.isfinite(real_values)].ravel()
    lo = float(real_finite.min())
    hi = float(real_finite.max())
    for values in synth_values_by_label.values():
        arr = np.asarray(values)
        arr = arr[np.isfinite(arr)].ravel()
        if arr.size:
            lo = min(lo, float(arr.min()))
            hi = max(hi, float(arr.max()))
    edges = np.linspace(lo, hi, bins + 1).tolist()

    ax.hist(real_finite, bins=edges, density=True, color="black", alpha=0.4, label="real")
    for i, (label, values) in enumerate(synth_values_by_label.items()):
        arr = np.asarray(values)
        arr = arr[np.isfinite(arr)].ravel()
        ax.hist(
            arr, bins=edges, density=True, color=palette[i % len(palette)], alpha=0.35, label=label
        )

    ax.set_xlabel("fitness")
    ax.set_ylabel("density")
    ax.set_title("Fitness distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_tensor_scatter(
    real_tensor: np.ndarray,
    synth_tensor: np.ndarray,
    label: str = "mutascape",
    gridsize: int = 60,
) -> Figure:
    """Hexbin of real vs synthetic fitness on shared tensor positions.

    Only positions where both tensors are finite are plotted. A y=x reference
    line and the Pearson r on the overlap set are included in the title.
    """
    if real_tensor.shape != synth_tensor.shape:
        raise ValueError(
            f"tensor shape mismatch: real={real_tensor.shape}, synth={synth_tensor.shape}"
        )
    mask = np.isfinite(real_tensor) & np.isfinite(synth_tensor)
    r = real_tensor[mask].ravel()
    s = synth_tensor[mask].ravel()

    r_pearson = (
        float("nan")
        if r.size < 2 or np.std(r) == 0 or np.std(s) == 0
        else float(np.corrcoef(r, s)[0, 1])
    )

    fig, ax = plt.subplots(figsize=(5.5, 5))
    hb = ax.hexbin(r, s, gridsize=gridsize, cmap="viridis", mincnt=1, bins="log")

    lo = float(min(r.min(), s.min()))
    hi = float(max(r.max(), s.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y = x")

    ax.set_xlabel("real fitness")
    ax.set_ylabel(f"{label} fitness")
    ax.set_title(f"Per-variant fitness (Pearson r = {r_pearson:.3f}, n = {r.size})")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r"$\log_{10}$ count")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig
