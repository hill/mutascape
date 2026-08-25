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
    "posterior_mean",
    "additive",
]
LABELS = ["Mutascape", "Potts", "RMF", "Empirical prior", "Additive"]
COLOURS = ["#e8792e", "#3d6ddd", "#55b685", "#7d65b5", "#808791"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path)
    parser.add_argument(
        "--development-budget",
        action="append",
        type=_parse_budget_run,
        default=[],
        help="Add one BUDGET=PATH run to the budget-sensitivity figure",
    )
    parser.add_argument(
        "--external-budget",
        action="append",
        type=_parse_budget_run,
        default=[],
        help="Add one BUDGET=PATH run to the external budget-sensitivity figure",
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
    transfer = pd.read_csv(root / "transfer.csv")
    pairwise = pd.read_csv(root / "pairwise_transfer.csv")
    adjusted = pd.read_csv(root / "reliability_adjusted_transfer.csv")
    tau = transfer.groupby("arm")["kendall_tau"].median().reindex(ARMS)
    pair_totals = pairwise.groupby("arm")[["correct_pairs", "scored_pairs"]].sum()
    winners = (pair_totals["correct_pairs"] / pair_totals["scored_pairs"]).reindex(ARMS)
    reliability = adjusted.groupby("arm")["meets_reliability_floor"].mean().reindex(ARMS)
    return tau, winners, reliability


def plot_summary(development: Path, external: Path | None, output: Path) -> None:
    panels = [("Development", development)]
    if external is not None and (external / "transfer.csv").exists():
        panels.append(("External protein panel", external))
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.45 * len(panels), 2.75),
        squeeze=False,
    )
    y = np.arange(len(ARMS))
    width = 0.25
    for axis, (title, root) in zip(axes[0], panels):
        tau, winners, reliability = _metrics(root)
        axis.barh(y - width, tau, width, label="Rank transfer", color="#e8792e")
        axis.barh(y, winners, width, label="Pairwise winners", color="#3d6ddd")
        axis.barh(
            y + width,
            reliability,
            width,
            label="Reliability pass",
            color="#55b685",
        )
        if len(panels) > 1:
            axis.set_title(title, loc="left", fontweight="bold")
        axis.set_yticks(y, LABELS)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xlabel("Score")
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=7,
        handlelength=1.2,
        columnspacing=0.8,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output / "search_transfer.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_landscapes(development: Path, output: Path) -> None:
    transfer = pd.read_csv(development / "transfer.csv")
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
    axis.set_xlabel("Kendall rank transfer")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    figure.tight_layout()
    figure.savefig(output / "landscape_transfer.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_budget_curve(
    development: list[tuple[int, Path]],
    external: list[tuple[int, Path]],
    output: Path,
) -> None:
    panels = [("Development", sorted(development))]
    if external:
        panels.append(("External protein panel", sorted(external)))
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.45 * len(panels), 2.75),
        squeeze=False,
        sharey=True,
    )
    for axis, (title, runs) in zip(axes[0], panels):
        budgets = [budget for budget, _ in runs]
        tables = [pd.read_csv(root / "transfer.csv") for _, root in runs]
        for arm, label, colour in zip(ARMS[:3], LABELS[:3], COLOURS[:3]):
            distributions = [
                table.loc[table["arm"] == arm, "kendall_tau"].dropna() for table in tables
            ]
            medians = [values.median() for values in distributions]
            lower = [values.quantile(0.25) for values in distributions]
            upper = [values.quantile(0.75) for values in distributions]
            axis.plot(budgets, medians, marker="o", label=label, color=colour, linewidth=2)
            axis.fill_between(budgets, lower, upper, color=colour, alpha=0.12)
        axis.axhline(0, color="#667085", linewidth=0.8)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, labels=[str(budget) for budget in budgets])
        axis.set_ylim(-0.25, 1)
        axis.set_xlabel("Query budget")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Kendall rank transfer")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left", fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "budget_transfer.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_summary(args.development, args.external, args.output)
    plot_landscapes(args.development, args.output)
    if args.development_budget:
        plot_budget_curve(args.development_budget, args.external_budget, args.output)


if __name__ == "__main__":
    main()
