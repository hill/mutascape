"""Build manuscript assets from the corrected connected-library efficacy study."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.benchmark_efficacy import OUT, PROTEINS, verify
from scripts.build_quantile_manuscript import BLUE, GREEN, PALE_BLUE, PALE_GREEN, STYLE
from scripts.build_singles_manuscript import save, write_table

LABELS = {
    "mutascape": "Mutascape",
    "zero_pair": "Zero-pair sigmoid",
    "additive": "Additive",
    "linear": "Linear",
    "pool": "Pooling",
    "ensemble": "Ensemble",
    "fixed": "Fixed choice",
}


def build(manuscript: Path):
    verify()
    data = pd.read_csv(OUT / "per_protein.csv")
    intervals = pd.read_csv(OUT / "intervals.csv")
    contrasts = pd.read_csv(OUT / "contrasts.csv")
    means = data.groupby(["budget", "model"])[
        ["mspd", "spearman", "tau", "selected_regret", "remaining_regret"]
    ].mean()
    tables, figures = manuscript / "tables", manuscript / "figures"
    models = list(LABELS)
    summary = means.loc[96]
    best_rank = max(float(f"{summary.loc[m, 'spearman']:.3f}") for m in models)
    best_regret = min(float(f"{summary.loc[m, 'selected_regret']:.4f}") for m in models)
    records = []
    for model in models:
        rank = float(f"{summary.loc[model, 'spearman']:.3f}")
        regret = float(f"{summary.loc[model, 'selected_regret']:.4f}")
        records.append(
            [
                LABELS[model],
                f"*{rank:.3f}*" if rank == best_rank else f"{rank:.3f}",
                f"*{regret:.4f}*" if regret == best_regret else f"{regret:.4f}",
            ]
        )
    write_table(
        tables / "search_summary.typ",
        ["Benchmark", "Spearman correlation", "Selected-method regret"],
        records,
        ["1.2fr", "1fr", "1fr"],
    )
    restricted = pd.read_csv(OUT / "ml_only_sensitivity.csv")
    restricted = restricted.loc[restricted.metric == "spearman"].groupby(["budget", "model"])
    restricted_means = restricted.point.mean()
    write_table(
        tables / "ml_sensitivity.typ",
        ["Benchmark", "48 evaluations", "96 evaluations"],
        [[LABELS[m], *[f"{restricted_means.loc[(b, m)]:.3f}" for b in (48, 96)]] for m in models],
        ["1.2fr", "1fr", "1fr"],
    )
    records = []
    for budget in (48, 96):
        for name in (*PROTEINS, "Mean"):
            group = (
                means.loc[budget]
                if name == "Mean"
                else data.loc[(data.budget == budget) & (data.protein == name)].set_index("model")
            )
            values = [float(f"{group.loc[m, 'spearman']:.3f}") for m in models]
            best = max(values)
            cells = [f"*{v:.3f}*" if v == best else f"{v:.3f}" for v in values]
            records.append([f"{name}, {budget}", *cells])
    write_table(
        tables / "singles_generators.typ",
        ["Protein, evaluations", *[LABELS[m] for m in models]],
        records,
        ["1.3fr", *["1fr"] * len(models)],
    )
    records = []
    group = means.loc[96]
    columns = ("mspd", "selected_regret", "remaining_regret")
    best = {c: min(float(f"{group.loc[m, c]:.4f}") for m in models) for c in columns}
    for model in models:
        cells = []
        for metric in columns:
            value = float(f"{group.loc[model, metric]:.4f}")
            cells.append(f"*{value:.4f}*" if value == best[metric] else f"{value:.4f}")
        records.append([LABELS[model], *cells])
    write_table(
        tables / "benchmark_efficacy.typ",
        ["Benchmark", "MSPD", "Selected-method regret", "Fraction of improvement missed"],
        records,
        ["1.2fr", "0.7fr", "1fr", "1.1fr"],
    )
    headroom = pd.read_csv(OUT / "headroom.csv")
    records = []
    for protein in PROTEINS:
        group = headroom.loc[headroom.protein == protein].groupby("model").mean(numeric_only=True)
        records.append(
            [
                protein,
                int(group.loc["measured", "known"]),
                int(group.loc["measured", "improving_candidates"]),
                f"{100 * group.loc['measured', 'improving_fraction']:.1f}%",
                f"{100 * group.loc['mutascape', 'improving_fraction']:.1f}%",
            ]
        )
    write_table(
        tables / "search_domains.typ",
        [
            "Protein",
            "Known variants",
            "Measured improving",
            "Measured fraction",
            "Mutascape fraction",
        ],
        records,
    )
    data.to_csv(figures / "singles_search_comparisons.csv", index=False)
    intervals.to_csv(figures / "singles_search_intervals.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.55))
        designs = [
            ("mutascape", -0.23, BLUE, PALE_BLUE),
            ("ensemble", 0, GREEN, PALE_GREEN),
            ("fixed", 0.23, "0.25", "0.85"),
        ]
        for ax, metric, title in zip(
            axes,
            ("spearman", "selected_regret"),
            ("(a) Method-rank correlation", "(b) Measured regret after selection"),
        ):
            for model, offset, colour, fill in designs:
                selected = (
                    intervals.loc[
                        (intervals.budget == 96)
                        & (intervals.model == model)
                        & (intervals.metric == metric)
                    ]
                    .set_index("protein")
                    .loc[list(PROTEINS)]
                )
                for index, row in enumerate(selected.itertuples()):
                    y = index + offset
                    if np.isfinite(row.point):
                        ax.barh(
                            y,
                            row.point,
                            height=0.19,
                            color=fill,
                            edgecolor=colour,
                            linewidth=0.85,
                            label=LABELS[model] if index == 0 else None,
                        )
                        ax.plot([row.low, row.high], [y, y], color=colour, linewidth=0.8)
                        ax.plot(
                            [row.low, row.low], [y - 0.055, y + 0.055], color=colour, linewidth=0.8
                        )
                        ax.plot(
                            [row.high, row.high],
                            [y - 0.055, y + 0.055],
                            color=colour,
                            linewidth=0.8,
                        )
            ax.set_yticks(range(len(PROTEINS)), PROTEINS)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.axvline(0, color="0.6", linewidth=0.65)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axes[0].set_xlabel("Spearman correlation")
        axes[0].set_xlim(-1.03, 1.03)
        axes[1].set_xlabel("Normalised regret, lower is better")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        save(fig, figures, "search_transfer")
    facts = {}
    for budget in (48, 96):
        for model in models:
            for metric in columns + ("spearman", "tau"):
                facts[f"{model}_{metric}_{budget}"] = f"{means.loc[(budget, model), metric]:.4f}"
    for control in ("mutascape", "fixed"):
        row = contrasts.loc[
            (contrasts.budget == 96)
            & (contrasts.target == "ensemble")
            & (contrasts.control == control)
            & (contrasts.metric == "selected_regret")
        ].iloc[0]
        for metric in ("difference", "low", "high"):
            facts[f"ensemble_minus_{control}_{metric}"] = f"{row[metric]:.4f}"
    (tables / "efficacy_facts.json").write_text(json.dumps(facts, indent=2) + "\n")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[2] / "mutascape_manu_v1")
