"""Create manuscript figures from frozen search-transfer outputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARMS = [
    "dms_pairwise_mean",
    "potts_low_mean",
    "rmf_high_mean",
    "additive",
]
LABELS = ["Mutascape", "Potts", "RMF", "Additive"]
COLOURS = ["#e8792e", "#3d6ddd", "#55b685", "#808791"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula", type=Path, required=True)
    parser.add_argument("--combingym", type=Path, required=True)
    parser.add_argument("--graphfla", type=Path, required=True)
    parser.add_argument(
        "--ssmula-budget",
        action="append",
        type=_parse_budget_run,
        default=[],
        help="Add one BUDGET=PATH SSMuLA run to the budget figure",
    )
    parser.add_argument(
        "--combingym-budget",
        action="append",
        type=_parse_budget_run,
        default=[],
        help="Add one BUDGET=PATH CombinGym run to the budget figure",
    )
    parser.add_argument(
        "--graphfla-budget",
        action="append",
        type=_parse_budget_run,
        default=[],
        help="Add one BUDGET=PATH GraphFLA run to the budget figure",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_budget_run(value: str) -> tuple[int, Path]:
    try:
        budget_text, path_text = value.split("=", maxsplit=1)
        return int(budget_text), Path(path_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected BUDGET=PATH") from error


def _metrics(root: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    proteins = pd.read_csv(root / "scores.csv")[["dataset", "protein"]].drop_duplicates()
    transfer = pd.read_csv(root / "transfer.csv")
    pairwise = pd.read_csv(root / "pairwise_transfer.csv")
    adjusted = pd.read_csv(root / "reliability_adjusted_transfer.csv")
    tau = (
        transfer.merge(proteins, on="dataset")
        .groupby(["arm", "protein"])["kendall_tau"]
        .median()
        .groupby("arm")
        .median()
        .reindex(ARMS)
    )
    protein_pairs = (
        pairwise.merge(proteins, on="dataset")
        .groupby(["arm", "protein"])[["correct_pairs", "scored_pairs"]]
        .sum()
    )
    protein_pairs = protein_pairs.loc[protein_pairs["scored_pairs"] > 0]
    winners = (
        (protein_pairs["correct_pairs"] / protein_pairs["scored_pairs"])
        .groupby("arm")
        .mean()
        .reindex(ARMS)
    )
    reliability = (
        adjusted.merge(proteins, on="dataset")
        .groupby(["arm", "protein"])["meets_reliability_floor"]
        .mean()
        .groupby("arm")
        .mean()
        .reindex(ARMS)
    )
    return tau, winners, reliability


def plot_summary(ssmula: Path, combingym: Path, graphfla: Path, output: Path) -> None:
    panels = [("SSMuLA", ssmula), ("CombinGym", combingym), ("GraphFLA", graphfla)]
    style = {
        "font.family": "STIX Two Text",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "text.color": "#161616",
        "axes.labelcolor": "#161616",
        "axes.edgecolor": "#161616",
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
        y = np.arange(len(ARMS))
        width = 0.22
        colours = [
            ("#167db5", "#c1deeb"),
            ("#d84a45", "#f3cdca"),
            ("#737373", "#e6e6e6"),
        ]
        metrics = ["Kendall correlation", "Pair agreement", "Repeatability"]
        for index, (axis, (title, root)) in enumerate(zip(axes, panels)):
            for offset, values, label, (edge, fill) in zip(
                (-width, 0, width), _metrics(root), metrics, colours
            ):
                axis.barh(
                    y + offset,
                    values,
                    width,
                    label=label,
                    facecolor=fill,
                    edgecolor=edge,
                    linewidth=0.65,
                    zorder=3,
                )
            axis.set_title(f"({chr(97 + index)}) {title}", loc="left", fontsize=10, pad=10)
            axis.set_yticks(y, LABELS)
            axis.invert_yaxis()
            axis.set_xlim(0, 1)
            axis.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            axis.set_xlabel("Score", labelpad=4)
            axis.tick_params(axis="y", length=0, pad=5)
            axis.tick_params(axis="x", length=3, width=0.6)
            axis.spines[["top", "right", "left"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            fontsize=9,
            handlelength=1.5,
            columnspacing=1.8,
        )
        figure.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.76, wspace=0.65)
        for extension in ("pdf", "svg", "png"):
            figure.savefig(output / f"search_transfer.{extension}", bbox_inches="tight", dpi=240)
        plt.close(figure)


def plot_landscapes(ssmula: Path, output: Path) -> None:
    transfer = pd.read_csv(ssmula / "transfer.csv")
    pivot = transfer.loc[transfer["arm"].isin(ARMS)].pivot(
        index="dataset", columns="arm", values="kendall_tau"
    )
    pivot = pivot.sort_values("dms_pairwise_mean")
    figure, axis = plt.subplots(figsize=(3.45, 4.2))
    y = np.arange(len(pivot))
    for arm, label, colour in zip(ARMS, LABELS, COLOURS):
        axis.scatter(
            pivot[arm],
            y,
            s=25 if arm == "dms_pairwise_mean" else 15,
            label=label,
            color=colour,
            zorder=3 if arm == "dms_pairwise_mean" else 2,
        )
    axis.axvline(0, color="#667085", linewidth=0.8)
    axis.set_yticks(y, pivot.index)
    axis.set_xlim(-1, 1)
    axis.set_xlabel("Kendall correlation")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    figure.tight_layout()
    figure.savefig(output / "landscape_transfer.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_budget_curve(
    ssmula: list[tuple[int, Path]],
    combingym: list[tuple[int, Path]],
    graphfla: list[tuple[int, Path]],
    output: Path,
) -> None:
    panels = [
        ("SSMuLA", sorted(ssmula)),
        ("CombinGym", sorted(combingym)),
        ("GraphFLA", sorted(graphfla)),
    ]
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.45 * len(panels), 2.75),
        squeeze=False,
        sharey=True,
    )
    for axis, (title, runs) in zip(axes[0], panels):
        budgets = [budget for budget, _ in runs]
        tables = []
        for _, root in runs:
            proteins = pd.read_csv(root / "scores.csv")[["dataset", "protein"]].drop_duplicates()
            tables.append(pd.read_csv(root / "transfer.csv").merge(proteins, on="dataset"))
        for arm, label, colour in zip(ARMS[:3], LABELS[:3], COLOURS[:3]):
            distributions = [
                table.loc[table["arm"] == arm].groupby("protein")["kendall_tau"].median().dropna()
                for table in tables
            ]
            medians = [values.median() for values in distributions]
            lower = [values.quantile(0.25) for values in distributions]
            upper = [values.quantile(0.75) for values in distributions]
            axis.plot(budgets, medians, marker="o", label=label, color=colour, linewidth=2)
            axis.fill_between(budgets, lower, upper, color=colour, alpha=0.12)
        if title == "GraphFLA":
            axis.axvspan(350, 420, color="#667085", alpha=0.12)
            axis.annotate(
                "4/7 nearly saturated",
                (384, 0.97),
                xytext=(140, 0.85),
                fontsize=7,
                arrowprops={"arrowstyle": "->", "lw": 0.7},
            )
        axis.axhline(0, color="#667085", linewidth=0.8)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, labels=[str(budget) for budget in budgets])
        axis.set_ylim(-0.25, 1)
        axis.set_xlabel("Query budget")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Kendall correlation")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left", fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "budget_transfer.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_summary(args.ssmula, args.combingym, args.graphfla, args.output)
    plot_landscapes(args.ssmula, args.output)
    if args.ssmula_budget:
        plot_budget_curve(
            args.ssmula_budget,
            args.combingym_budget,
            args.graphfla_budget,
            args.output,
        )


if __name__ == "__main__":
    main()
