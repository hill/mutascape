"""Plotting utilities for mutascape fitness landscapes."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from mutascape.landscape import Landscape


def plot_epistasis_scatter(
    landscape: Landscape,
    depths: list[int] = [1, 2, 4],
    n_samples: int = 500,
    seed: int | None = None,
) -> Figure:
    """Scatter plot of additive prediction vs actual fitness at each mutation depth.

    Points on the diagonal mean no epistasis; deviation shows epistatic effects.
    """
    fig, axes = plt.subplots(1, len(depths), figsize=(4.5 * len(depths), 4))
    if len(depths) == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        variants, fitness = landscape._sample(n=n_samples, n_mutations=depth)
        additive = np.array([landscape.additive_fitness(v) for v in variants])

        ax.scatter(additive, fitness, alpha=0.3, s=10, color=sns.color_palette()[0])

        lims = [
            min(additive.min(), fitness.min()) - 0.1,
            max(additive.max(), fitness.max()) + 0.1,
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
        ax.set_xlabel("Additive prediction")
        ax.set_ylabel("Actual fitness")
        ax.set_title(f"{depth}-mutants")

    fig.suptitle(f"Additive vs actual fitness (alpha={landscape.alpha})", y=1.02)
    fig.tight_layout()
    return fig


def plot_fitness_distribution(
    landscape: Landscape,
    depths: list[int] = [1, 2, 3, 4],
    n_samples: int = 500,
) -> Figure:
    """Overlaid fitness distributions at different mutation depths."""
    fig, ax = plt.subplots(figsize=(7, 4))

    palette = sns.color_palette(n_colors=len(depths))
    for depth, color in zip(depths, palette):
        _, fitness = landscape._sample(n=n_samples, n_mutations=depth)
        ax.hist(fitness, bins=30, alpha=0.45, color=color, label=f"{depth}-mutants", density=True)

    ax.set_xlabel("Fitness")
    ax.set_ylabel("Density")
    ax.set_title(f"Fitness distribution by mutation depth (alpha={landscape.alpha})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_alpha_sweep(
    landscape: Landscape,
    alphas: list[float] | None = None,
    n_samples: int = 2000,
    n_mutations: int = 3,
) -> tuple[Figure, dict]:
    """Show how landscape properties change with epistatic strength.

    Returns the figure and a dict of {alpha: stats} for further analysis.
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]

    from mutascape.landscape import Landscape as Ls

    sweep: dict[float, dict[str, float]] = {}
    for alpha in alphas:
        ls = Ls(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=alpha,
            seed=landscape.seed,
            wt_sequence=landscape.wt_sequence,
        )
        sweep[alpha] = ls.stats(n_samples=n_samples, n_mutations=n_mutations)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    corrs = [sweep[a]["additive_correlation"] for a in alphas]
    sign_epi = [sweep[a]["fraction_sign_epistasis"] for a in alphas]

    color1 = sns.color_palette()[0]
    color2 = sns.color_palette()[1]

    ax1.plot(alphas, corrs, "o-", color=color1, label="Additive correlation")
    ax1.set_xlabel("Alpha (epistatic strength)")
    ax1.set_ylabel("Additive correlation", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(alphas, sign_epi, "s-", color=color2, label="Sign epistasis fraction")
    ax2.set_ylabel("Fraction sign epistasis", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(f"Landscape ruggedness vs alpha ({n_mutations}-mutants)")
    fig.tight_layout()
    return fig, sweep


def plot_contact_map(
    landscape: Landscape,
    max_labels: int = 30,
) -> Figure:
    """Heatmap of the structural contact map."""
    cm = landscape.contact_map
    n = cm.n_positions

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot as sparse scatter for cleaner look at scale
    rows, cols = np.where(cm.matrix)
    dot_size = max(1, 80 / (n / 20))
    ax.scatter(
        cols,
        rows,
        s=dot_size,
        marker="s",
        color=sns.color_palette()[0],
        alpha=0.7,
        linewidths=0,
    )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")

    # Only show a subset of tick labels to avoid overlap
    positions = list(cm.positions)
    if n > max_labels:
        step = max(1, n // max_labels)
        tick_indices = list(range(0, n, step))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([positions[i] for i in tick_indices], fontsize=8, rotation=45)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([positions[i] for i in tick_indices], fontsize=8)
    else:
        ax.set_xticks(range(n))
        ax.set_xticklabels(positions, fontsize=8, rotation=45)
        ax.set_yticks(range(n))
        ax.set_yticklabels(positions, fontsize=8)

    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    ax.set_title(
        f"Contact map (threshold={cm.threshold}\u00c5, {cm.n_contacts} contacts)",
        fontsize=11,
    )
    fig.tight_layout()
    plt.close(fig)
    return fig
