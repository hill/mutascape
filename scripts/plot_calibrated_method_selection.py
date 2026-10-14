"""Plot experimental outcomes after choosing methods from calibrated landscapes."""

import matplotlib.pyplot as plt
import pandas as pd

from scripts.calibrated_method_selection import EVALUATION, METHODS, OUT

LABELS = {
    "calibrated": "Combination-calibrated Mutascape",
    "singles": "Single-site Mutascape + same observations",
    "additive": "Mutascape additive + same observations",
    "fixed": "Fixed recommendation + same observations",
    "random_choice": "Uniform random method choice",
    "random_search": "Random search + same observations",
    "fixed_direct": "Fixed recommendation, all budget used in search",
    "random_direct": "Random search, all budget used in search",
}
COLOURS = {
    "calibrated": "#26a51b",
    "singles": "#078acb",
    "additive": "#9871ae",
    "fixed": "#333333",
    "random_choice": "#b07832",
    "random_search": "#999999",
    "fixed_direct": "#333333",
    "random_direct": "#999999",
}


def save(fig, stem: str) -> None:
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(folder / f"{stem}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def point(ax, row, y: float, label: str) -> None:
    ax.plot([row.low, row.high], [y, y], color=COLOURS[label], lw=1.3)
    ax.plot([row.low, row.low], [y - 0.06, y + 0.06], color=COLOURS[label], lw=1)
    ax.plot([row.high, row.high], [y - 0.06, y + 0.06], color=COLOURS[label], lw=1)
    ax.scatter(row.point, y, color=COLOURS[label], s=35, zorder=3)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "savefig.dpi": 180,
        }
    )
    data = pd.read_csv(OUT / "intervals.csv")
    data = data.query("method_group=='all' and metric=='selected_regret'")
    order = ["calibrated", "singles", "additive", "fixed", "random_choice", "random_search"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, budget in zip(axes, [48, 96]):
        sub = data.query(
            "protein=='All' and budget==@budget and uncertainty=='protein_and_calibration'"
        ).set_index("selector")
        for y, label in enumerate(order):
            point(ax, sub.loc[label], y, label)
        ax.set_yticks(range(len(order)), [LABELS[label] for label in order])
        ax.set_xlabel("Experimental regret after method selection\nLower is better")
        ax.set_title(f"96 calibration + {budget} search evaluations")
        ax.set_xlim(left=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.12)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Does a synthetic recommendation improve experimental search?\n"
        "Four proteins receive equal weight. Lines show 95% hierarchical bootstrap intervals.",
        fontsize=12,
    )
    save(fig, "method_selection_summary")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, layout="constrained")
    order = ["calibrated", "singles", "fixed", "fixed_direct", "random_direct"]
    short = [
        "Combination-calibrated",
        "Single-site + calibration observations",
        "Fixed method + calibration observations",
        "Fixed method, 192 search evaluations",
        "Random search, 192 evaluations",
    ]
    for ax, protein in zip(axes.flat, EVALUATION):
        sub = data.query("protein==@protein and budget==96").set_index("selector")
        for y, label in enumerate(order):
            point(ax, sub.loc[label], y, label)
        ax.set_title(protein)
        ax.set_yticks(range(len(order)), short)
        ax.set_xlim(left=0)
        ax.set_xlabel("Experimental regret, lower is better")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.12)
    axes[0, 0].invert_yaxis()
    fig.suptitle(
        "The same total cost of 192 multi-mutant measurements\n"
        "Calibration uses 96 measurements, followed by 96 further search evaluations.",
        fontsize=12,
    )
    save(fig, "calibration_cost_comparison")

    freq = pd.read_csv(OUT / "selection_frequencies.csv")
    freq = freq.query("method_group=='all' and budget==96 and selector=='calibrated'")
    matrix = (
        freq.groupby(["protein", "method"])
        .weight.mean()
        .unstack()
        .reindex(index=EVALUATION, columns=list(METHODS))
    )
    names = [
        "Random",
        "Greedy",
        "Annealing",
        "Genetic",
        "Ridge greedy",
        "Ridge UCB",
        "GP EI",
        "GP UCB",
        "Forest greedy",
        "Forest Thompson",
        "MLP UCB",
        "AdaLead",
    ]
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(12), names, rotation=40, ha="right")
    ax.set_yticks(range(4), EVALUATION)
    for i in range(4):
        for j in range(12):
            value = matrix.iloc[i, j]
            if value > 0:
                ax.text(
                    j,
                    i,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                    fontsize=9,
                )
    fig.colorbar(im, ax=ax, label="Fraction of recommendations")
    ax.set_title(
        "Recommendations across ten independent calibration samples\n"
        "96 further evaluations. Tied recommendations share weight."
    )
    save(fig, "recommendation_stability")

    headroom = pd.read_csv(OUT / "headroom.csv")
    counts = headroom.groupby("protein").zero_headroom.sum().reindex(EVALUATION)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    bars = ax.bar(EVALUATION, counts, color="#cae3c1", edgecolor="#26a51b", linewidth=1)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.15, f"{count}/10", ha="center")
    ax.set_ylim(0, 10.8)
    ax.set_ylabel("Calibration samples")
    ax.set_title(
        "The calibrated generator often predicts no further improvement\n"
        "Experimental genotypes outperform the starting observations in all 40 cases."
    )
    save(fig, "no_predicted_improvement")


if __name__ == "__main__":
    main()
