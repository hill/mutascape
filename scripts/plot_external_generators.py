"""Build comparison figures and a diagram from the saved experiment outputs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from scripts.analyse_external_generators import LABELS
from scripts.compare_external_generators import OUT, PROTEINS

FIGURES = OUT / "figures"
COLOURS = dict(
    measured="#222222",
    mutascape="#007dc3",
    mutascape_additive="#28a315",
    mutascape_zero_pair="#77b947",
    slip4="#d44444",
    slip8="#dd9090",
    slip_raw="#7b1818",
    nk1="#8565a7",
    nk0="#ad99c4",
    nk3="#644b7f",
    rmf="#e49c32",
    hoc="#a4a4a4",
    mutascape_pool="#008c80",
    mutascape_linear="#5799bb",
    ensemble="#003e70",
    fixed="#777777",
    random_choice="#aaaaaa",
)


def style():
    plt.rcParams.update(
        {
            "font.family": "STIX Two Text",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    FIGURES.mkdir(exist_ok=True)


def save(fig, name):
    for extension in ("svg", "png"):
        fig.savefig(FIGURES / f"{name}.{extension}", bbox_inches="tight", dpi=190)
    plt.close(fig)


def diagram():
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set(xlim=(-0.12, 12.1), ylim=(-0.05, 7.3))
    ax.axis("off")

    def box(x, y, w, h, label, colour):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.09",
                edgecolor=colour,
                facecolor=colour + "13",
                linewidth=1.2,
            )
        )
        ax.text(
            x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=12, linespacing=1.45
        )

    def arrow(x, y, u, v):
        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (u, v),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#666666",
                linewidth=1.0,
                connectionstyle="arc3,rad=0",
            )
        )

    ax.text(0, 7.05, "What each generator learns, and how we test its usefulness", fontsize=19)
    box(0.05, 5.6, 3.1, 0.85, "WT + measured single-site fitness", COLOURS["mutascape"])
    box(0.05, 4.3, 3.1, 0.85, "Protein sequence alignment", COLOURS["slip4"])
    box(0.05, 3, 3.1, 0.85, "Alphabet + interaction settings", COLOURS["nk1"])
    box(
        3.7,
        5.6,
        3.25,
        0.85,
        "Mutascape configurations\nPreserve measured single-site values",
        COLOURS["mutascape"],
    )
    box(3.7, 4.3, 3.25, 0.85, "SLIP\nFit Potts model, then tune difficulty", COLOURS["slip4"])
    box(
        3.7,
        3,
        3.25,
        0.85,
        "NK / Rough Mount Fuji\nGenerate fitness from random terms",
        COLOURS["nk1"],
    )
    for y in (6.02, 4.72, 3.42):
        arrow(3.18, y, 3.57, y)
        arrow(7.03, y, 7.7, y)
    box(
        7.8,
        3,
        4.0,
        3.45,
        "Simulated campaigns\n\nSame genotype search space\nSame 12 search methods\n"
        "48 or 96 new evaluations\n30 campaign repeats",
        "#555555",
    )
    box(
        0.05,
        0.6,
        3.65,
        1.25,
        "Measured combinatorial fitness\nKept out of generator fitting\n"
        "Used for reference campaigns",
        "#222222",
    )
    box(
        4.2,
        0.6,
        3.65,
        1.25,
        "Select a search method\nUse its performance on\nthe synthetic landscapes",
        COLOURS["mutascape"],
    )
    box(
        8.35,
        0.6,
        3.45,
        1.25,
        "Test that choice\nHow well does it perform\non measured fitness?",
        COLOURS["mutascape_additive"],
    )
    arrow(9.8, 2.9, 6.05, 1.98)
    arrow(7.92, 1.22, 8.2, 1.22)
    arrow(1.88, 0.46, 1.88, 0.15)
    ax.plot([1.88, 10.05], [0.15, 0.15], color="#666666", lw=1)
    arrow(10.05, 0.15, 10.05, 0.47)
    save(fig, "comparison_design")


def summary_plot(intervals, budget):
    order = [
        "mutascape",
        "mutascape_additive",
        "mutascape_zero_pair",
        "slip4",
        "nk1",
        "rmf",
        "mutascape_pool",
        "mutascape_linear",
        "ensemble",
        "fixed",
        "random_choice",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.1), sharey=True, gridspec_kw={"wspace": 0.25})
    sub = intervals.query("protein=='All' and budget==@budget")
    for ax, metric, title in zip(
        axes,
        ["selected_regret", "tau"],
        ["(a) Measured regret after method selection", "(b) Agreement in method rankings"],
    ):
        for i, m in enumerate(order):
            row = sub.query("model==@m and metric==@metric").iloc[0]
            if not np.isfinite(row.point):
                ax.text(0.025, i, "Not defined", color="#777777", va="center")
                continue
            ax.plot([row.low, row.high], [i, i], color=COLOURS[m], lw=1.4)
            ax.plot([row.low, row.high], [i, i], linestyle="", marker="|", color=COLOURS[m], ms=6)
            ax.scatter(row.point, i, color=COLOURS[m], s=40, zorder=3)
        ax.set_title(title, pad=16)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(range(len(order)), [LABELS[m] for m in order])
        ax.axvline(0, color="#aaaaaa", lw=0.6, zorder=0)
        ax.grid(axis="x", alpha=0.15)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Normalised regret on measured fitness\nLower is better")
    axes[1].set_xlabel("Kendall's tau\nHigher is better")
    axes[1].set_xlim(-0.4, 1.05)
    fig.text(
        0.29,
        -0.015,
        f"{budget} additional evaluations. Proteins receive equal weight. "
        "Bars show 95% bootstrap intervals.",
        fontsize=10,
    )
    save(fig, f"search_comparison_{budget}")


def protein_plot(intervals):
    models = ["mutascape", "mutascape_additive", "slip4", "nk1", "rmf", "ensemble", "fixed"]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5), sharey=True)
    for ax, protein in zip(axes, PROTEINS):
        for i, m in enumerate(models):
            row = intervals.query(
                "protein==@protein and budget==96 and model==@m and metric=='selected_regret'"
            ).iloc[0]
            ax.plot([row.low, row.high], [i, i], color=COLOURS[m], lw=1.3)
            ax.scatter(row.point, i, color=COLOURS[m], s=35, zorder=3)
        ax.set_yticks(range(len(models)), [LABELS[m] for m in models])
        ax.set_title(protein)
        ax.set_xlabel("Measured regret after selection")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.15)
        ax.set_xlim(left=0)
    axes[0].invert_yaxis()
    fig.text(
        0.23,
        -0.02,
        "96 additional evaluations. Lower is better. "
        "Bars show 95% bootstrap intervals within each protein.",
        fontsize=10,
    )
    fig.tight_layout(w_pad=2)
    save(fig, "selection_by_protein")


def ranks_plot():
    data = pd.read_csv(OUT / "method_means.csv")
    models = ["measured", "mutascape", "mutascape_additive", "slip4", "nk1", "rmf"]
    names = {
        "random": "Random",
        "greedy": "Greedy",
        "annealing": "Annealing",
        "genetic": "Genetic",
        "ridge_greedy": "Pairwise ridge",
        "ridge_ucb": "Ridge UCB",
        "gp_ei": "GP EI",
        "gp_ucb": "GP UCB",
        "forest_greedy": "Random forest",
        "forest_thompson": "Forest Thompson",
        "mlp_ucb": "MLP UCB",
        "adalead": "AdaLead",
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    for ax, protein in zip(axes, PROTEINS):
        sub = data.query("protein==@protein and budget==96")
        # Rank within each search space, then average over its spaces.
        sub = sub.copy()
        sub["regret"] = sub["regret"].round(12)
        sub["rank"] = sub.groupby(["domain", "model"]).regret.rank(method="average")
        matrix = sub.groupby(["model", "method"])["rank"].mean().unstack().reindex(models)
        order = matrix.loc["measured"].sort_values().index
        matrix = matrix[order]
        im = ax.imshow(matrix.to_numpy(), cmap="Blues_r", vmin=1, vmax=12, aspect="auto")
        ax.set_yticks(range(len(models)), [LABELS[m] for m in models])
        ax.set_xticks(range(len(order)), [names[m] for m in order], rotation=20, ha="right")
        ax.set_title(protein, loc="left")
        for i in range(len(models)):
            for j in range(len(order)):
                val = matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if val < 5 else "#222222",
                )
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.55, label="Method rank, 1 is best", pad=0.02)
    save(fig, "method_rankings")


def properties_plot():
    properties = pd.read_csv(OUT / "properties.csv")
    models = ["measured", "mutascape", "mutascape_additive", "slip4", "nk1", "rmf"]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8))
    for col, protein in enumerate(PROTEINS):
        sub = properties.query("protein==@protein")
        folder = OUT / protein / "0"
        known = np.load(folder / "design.npz")["known"]
        real = np.load(folder / "measured_0.npy")
        keep = np.setdiff1d(np.arange(len(real)), known)
        arrays = {
            m: np.stack(
                [
                    np.load(folder / f"{m}_{s}.npy")[keep]
                    for s in range(5 if m in ("mutascape", "mutascape_pool") else 1)
                ]
            )
            for m in ("measured", "mutascape", "mutascape_additive", "mutascape_pool")
        }
        x = np.linspace(
            min(a.min() for a in arrays.values()), max(a.max() for a in arrays.values()), 600
        )
        for m, values in arrays.items():
            cdf = np.stack([np.searchsorted(np.sort(v), x, side="right") / len(v) for v in values])
            axes[0, col].plot(x, cdf.mean(axis=0), color=COLOURS[m], lw=1.4, label=LABELS[m])
            axes[0, col].fill_between(
                x, cdf.min(axis=0), cdf.max(axis=0), color=COLOURS[m], alpha=0.15
            )
        axes[0, col].set_title(protein)
        axes[0, col].set_xlabel("Fitness on the measured assay scale")
        axes[0, col].set_ylim(0, 1.02)
        for i, m in enumerate(models):
            rows = sub.query("model==@m")
            matrix = rows[["walsh1", "walsh2", "walsh3"]].to_numpy()
            mean = matrix.mean(axis=0)
            xloc = np.arange(3) + (i - 2.5) * 0.12
            axes[1, col].bar(
                xloc, mean, width=0.115, color=COLOURS[m] + "40", edgecolor=COLOURS[m], linewidth=1
            )
            axes[1, col].errorbar(
                xloc,
                mean,
                yerr=np.array([mean - matrix.min(axis=0), matrix.max(axis=0) - mean]),
                fmt="none",
                ecolor=COLOURS[m],
                capsize=2,
                lw=0.8,
            )
        axes[1, col].set_xticks([0, 1, 2], ["1", "2", "3"])
        axes[1, col].set_xlabel("Interaction order")
        axes[1, col].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Fraction of multi-mutant genotypes")
    axes[1, 0].set_ylabel("Fraction of local fitness variance")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    handles = [
        Patch(facecolor=COLOURS[m] + "40", edgecolor=COLOURS[m], label=LABELS[m]) for m in models
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.52, -0.03)
    )
    fig.tight_layout(h_pad=3, rect=(0, 0.055, 1, 1))
    save(fig, "fitness_and_walsh")


def selection_diagram():
    data = pd.read_csv(OUT / "per_search_space.csv")
    examples = ["mutascape", "mutascape_additive", "ensemble"]
    names = {"forest_greedy": "Random forest", "gp_ucb": "GP UCB"}
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set(xlim=(-0.1, 12.1), ylim=(-0.2, 5.6))
    ax.axis("off")
    ax.text(0, 5.4, "What a benchmark recommendation means in practice", fontsize=19)
    ax.text(
        0, 4.93, "eqFP611, 96 additional evaluations, 30 measured campaign repeats", fontsize=12
    )
    for x, text in [
        (0, "Synthetic benchmark"),
        (4.5, "Its recommended search method"),
        (9, "Measured outcome"),
    ]:
        ax.text(x, 4.15, text, fontsize=12, color="#555555")
    for i, model in enumerate(examples):
        row = data.query("protein=='eqFP611' and budget==96 and model==@model").iloc[0]
        choices = row.choices.split("|")
        method = names[choices[0]] if len(choices) == 1 else f"Tie across {len(choices)} methods"
        y = 3.4 - i * 1.25
        colour = COLOURS[model]
        label = LABELS[model].replace(" configuration average", "\nconfiguration average")
        for x, w, text in [
            (0, 3.4, label),
            (4.5, 3.6, method),
            (9, 2.9, f"{100 * row.remaining_regret:.2f}% of available\nimprovement missed"),
        ]:
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - 0.4),
                    w,
                    0.8,
                    boxstyle="round,pad=.07",
                    edgecolor=colour,
                    facecolor=colour + "13",
                    linewidth=1.1,
                )
            )
            ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=13)
        for x, u in [(3.5, 4.35), (8.2, 8.85)]:
            ax.add_patch(
                FancyArrowPatch((x, y), (u, y), arrowstyle="-|>", mutation_scale=14, color=colour)
            )
    ax.text(
        0,
        -0.03,
        "Lower is better. Tied recommendations receive equal weight. "
        "This panel illustrates one protein.",
        fontsize=11,
    )
    save(fig, "method_selection_example")


def main():
    style()
    diagram()
    intervals = pd.read_csv(OUT / "intervals.csv")
    for budget in (48, 96):
        summary_plot(intervals, budget)
    protein_plot(intervals)
    ranks_plot()
    properties_plot()
    selection_diagram()


if __name__ == "__main__":
    main()
