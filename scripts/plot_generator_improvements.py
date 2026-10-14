"""Show candidate outcomes, replication and the local Walsh sampling design."""

from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.plot_external_generators import style
from scripts.replicate_generator_improvements import OUT as REPLICATION
from scripts.run_generator_improvements import OUT

FIGURES = OUT / "figures"
LABELS = {
    "mutascape": "Mutascape, current default",
    "mutascape_additive": "Mutascape additive",
    "mutascape_linear": "Mutascape linear + pairs",
    "mutascape_pool": "Mutascape pooling",
    "wide_sigmoid": "Wider sigmoid limits",
    "effect_pairs": "Effect-dependent pair variance",
    "uniform_matched": "Matched uniform pair variance",
    "diminishing4": "Negative beneficial pairs, horizon 4",
    "diminishing8": "Negative beneficial pairs, horizon 8",
    "power_pool15": "Power pooling, exponent 1.5",
    "msa_slip4": "DMS + SLIP interactions, horizon 4",
    "msa_slip8": "DMS + SLIP interactions, horizon 8",
    "msa_regression": "DMS + Potts regression",
    "slip4": "Native SLIP, horizon 4",
    "nk1": "NK, K = 1",
    "rmf": "Rough Mount Fuji",
    "fixed": "Fixed method from other proteins",
}


def save(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def outcomes():
    first, second = pd.read_csv(OUT / "intervals.csv"), pd.read_csv(REPLICATION / "intervals.csv")
    order = [
        "mutascape",
        "mutascape_additive",
        "mutascape_linear",
        "mutascape_pool",
        "effect_pairs",
        "uniform_matched",
        "slip4",
        "nk1",
        "rmf",
        "fixed",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, table, title, colour in zip(
        axes,
        [first, second],
        ["Initial experiment", "Fresh-seed replication"],
        ["#007dc3", "#28a315"],
    ):
        sub = table.query("protein=='All' and budget==96 and metric=='selected_regret'")
        for i, model in enumerate(order):
            row = sub.loc[sub.model == model]
            if row.empty:
                ax.text(0.003, i, "Added for replication", va="center", fontsize=9, color="#777777")
                continue
            row = row.iloc[0]
            ax.plot([row.low, row.high], [i, i], color=colour, lw=1.1)
            ax.scatter(row.point, i, color=colour, s=40, zorder=4)
        ax.set_title(title)
        ax.set_yticks(range(len(order)), [LABELS[m] for m in order])
        ax.set_xlabel("Measured regret after method selection\nLower is better")
        ax.set_xlim(-0.002, 0.105)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.15)
    axes[0].invert_yaxis()
    fig.text(
        0.13,
        -0.005,
        "GB1, CreiLOV and eqFP611 receive equal weight. 96 evaluations. "
        "Lines show 95% bootstrap intervals.",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "selection_replication")


def candidates():
    table = pd.read_csv(OUT / "intervals.csv")
    order = [
        "mutascape",
        "wide_sigmoid",
        "effect_pairs",
        "diminishing4",
        "diminishing8",
        "power_pool15",
        "msa_slip4",
        "msa_slip8",
        "msa_regression",
        "slip4",
        "nk1",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, metric, title in zip(
        axes,
        ["selected_regret", "tau"],
        ["Method-selection outcome", "Agreement in the full method ranking"],
    ):
        sub = table.query("protein=='All' and budget==96 and metric==@metric")
        for i, model in enumerate(order):
            row = sub.loc[sub.model == model].iloc[0]
            colour = "#8565a7" if model.startswith("msa") or model == "slip4" else "#007dc3"
            ax.plot([row.low, row.high], [i, i], color=colour, lw=1)
            ax.scatter(row.point, i, color=colour, s=35)
        ax.set_yticks(range(len(order)), [LABELS[m] for m in order])
        ax.set_title(title)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.15)
    axes[0].set_xlabel("Measured regret after selection\nLower is better")
    axes[1].set_xlabel("Kendall's tau\nHigher is better")
    axes[0].invert_yaxis()
    fig.text(
        0.10,
        -0.005,
        "Exploratory candidate screen, before replication. 96 evaluations. "
        "Lines show 95% bootstrap intervals.",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "candidate_results")


def subcube():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(xlim=(-0.7, 7.1), ylim=(-0.8, 2.7))
    ax.axis("off")
    ax.text(
        -0.5,
        2.45,
        "One local Walsh transform uses all eight combinations of three substitutions",
        fontsize=17,
    )
    states = list(product([0, 1], repeat=3))
    coordinates = {s: np.array([1.4 * s[0] + 0.6 * s[1], 1.3 * s[2] + 0.45 * s[1]]) for s in states}
    for a, b in combinations(states, 2):
        if sum(x != y for x, y in zip(a, b)) == 1:
            points = np.stack([coordinates[a], coordinates[b]])
            ax.plot(points[:, 0], points[:, 1], color="#8db4c9", lw=1.5, zorder=1)
    for state, xy in coordinates.items():
        label = "".join(letter for letter, present in zip("ABC", state) if present) or "None"
        offset = (0, -0.22) if state[1] == 0 else (0, 0.19)
        ax.scatter(*xy, s=40, color="#007dc3", zorder=2)
        ax.text(xy[0] + offset[0], xy[1] + offset[1], label, ha="center", va="center", fontsize=12)
    ax.text(
        0.85,
        -0.6,
        "Each point has a measured fitness.\nThe rest of the sequence stays fixed.",
        ha="center",
        fontsize=11,
    )
    for x in (2.5, 4.5):
        ax.add_patch(
            FancyArrowPatch(
                (x, 0.95), (x + 0.45, 0.95), arrowstyle="-|>", mutation_scale=15, color="#555555"
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (3.05, 0.3),
            1.3,
            1.3,
            boxstyle="round,pad=0.08",
            facecolor="#007dc313",
            edgecolor="#007dc3",
        )
    )
    ax.text(3.7, 0.95, "Walsh\ntransform", ha="center", va="center", fontsize=14)
    ax.text(5.05, 1.65, "Separate the local fitness pattern", fontsize=12)
    ax.text(5.05, 1.15, "Single-site terms\nA, B, C", fontsize=12, color="#007dc3")
    ax.text(5.05, 0.5, "Pair terms\nAB, AC, BC", fontsize=12, color="#28a315")
    ax.text(5.05, -0.15, "Three-site term\nABC", fontsize=12, color="#8565a7")
    ax.text(
        3.7,
        -0.55,
        "The transform also gives the mean.\nThe variance spectrum excludes it.",
        ha="center",
        fontsize=10,
    )
    save(fig, "local_walsh_explanation")


if __name__ == "__main__":
    style()
    FIGURES.mkdir(exist_ok=True)
    outcomes()
    candidates()
    subcube()
