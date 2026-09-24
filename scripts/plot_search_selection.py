"""Plot measured regret differences after synthetic method selection."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from scripts.search_selection_validation import OUT


def main():
    table = pd.read_csv(OUT / "contrasts.csv").query(
        "subset == 'all' and metric == 'selected_regret'"
    )
    plt.rcParams.update(
        {
            "font.family": "STIX Two Text",
            "font.size": 11,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 5.6), sharey=True)
    names = ["Pab1", "Ube4b", "New-protein mean", "GB1", "TrpB4", "T7", "TEV", "Repeat-panel mean"]
    y_positions = [9, 8, 7, 5, 4, 3, 2, 1]
    colours = {"fixed": "#137fc1", "additive": "#32a627"}
    for axis, budget in zip(axes, (48, 96)):
        axis.axvline(0, color="#888888", linewidth=0.7, linestyle=(0, (4, 4)))
        axis.axhline(6, color="#dddddd", linewidth=0.6)
        for name, y in zip(names, y_positions):
            for control, shift in (("fixed", 0.13), ("additive", -0.13)):
                if name.endswith("mean"):
                    panel = "new" if name.startswith("New") else "repeat"
                    row = table.loc[
                        (table.scope == "panel")
                        & (table.panel == panel)
                        & (table.budget == budget)
                        & (table.control == control)
                    ].iloc[0]
                else:
                    row = table.loc[
                        (table.scope == "protein")
                        & (table.protein == name)
                        & (table.budget == budget)
                        & (table.control == control)
                    ].iloc[0]
                axis.hlines(
                    y + shift,
                    row.low * 100,
                    row.high * 100,
                    color=colours[control],
                    linewidth=1.2,
                    alpha=0.7,
                )
                axis.plot(
                    row.difference * 100,
                    y + shift,
                    "D" if name.endswith("mean") else "o",
                    color=colours[control],
                    markersize=4.4,
                )
        axis.set_title(f"{budget} combination queries", loc="left", pad=14, fontsize=13)
        axis.set_yticks(y_positions, names)
        axis.tick_params(axis="y", length=0, pad=8)
        axis.set_ylim(0.3, 9.7)
        axis.set_xlabel("Difference in measured regret\n(percentage points of fitness range)")
    low = min(axis.get_xlim()[0] for axis in axes)
    high = max(axis.get_xlim()[1] for axis in axes)
    for axis in axes:
        axis.set_xlim(low, high)
    figure.legend(
        [Line2D([], [], color=colours[c], marker="o", linewidth=1) for c in colours],
        ["Mutascape minus fixed selector", "Mutascape minus additive selector"],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.98),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.58,
        0.03,
        "Below zero favours Mutascape. Lines show 95% bootstrap intervals.",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.19, right=0.99, top=0.86, bottom=0.17, wspace=0.13)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / f"selection_regret.{extension}", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
