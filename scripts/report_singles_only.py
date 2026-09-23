"""Freeze compact evidence and plot the singles-only research comparisons."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from mutascape.benchmarks.validity import pairwise_winner_transfer, reliable_method_pairs
from scripts.investigate_singles_only import (
    CONFIRMATION,
    DEVELOPMENT,
    OUT,
    SEEDS,
    digest,
    final_models,
)

DESTINATION = Path("src/mutascape/data/validation/singles-only-research-v1.json")
FIGURE = Path("docs/figures/singles_only_confirmation")


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def transfer(scores):
    measured = (
        scores.loc[scores.model == "measured"]
        .groupby(["protein", "budget", "method"])
        .regret.mean()
        .unstack()
    )
    rows = []
    for (name, budget), group in scores.groupby(["protein", "budget"]):
        if name not in CONFIRMATION:
            continue
        means = (
            group.groupby(["model", "method", "trial"])
            .regret.mean()
            .reset_index()
            .rename(columns={"model": "arm"})
            .assign(dataset=name)
        )
        pairs = pairwise_winner_transfer(means, seed=2026092300, n_bootstrap=2000)
        ranks = means.pivot_table(index="method", columns="arm", values="regret")
        prior = measured.loc[[(p, budget) for p in DEVELOPMENT]].mean()
        eligible = reliable_method_pairs(means, seed=2026092300, n_bootstrap=2000)
        for item in pairs.to_dict("records"):
            model = item.pop("arm")
            item.pop("dataset")
            rows.append(
                dict(
                    protein=name,
                    budget=budget,
                    model=model,
                    kendall_tau=kendalltau(ranks.measured, ranks[model]).statistic,
                    **item,
                )
            )
        correct = sum(
            np.sign(prior[r.first] - prior[r.second]) == r.sign for r in eligible.itertuples()
        )
        rows.append(
            dict(
                protein=name,
                budget=budget,
                model="fixed_ranking",
                kendall_tau=kendalltau(ranks.measured, prior.reindex(ranks.index)).statistic,
                scored_pairs=len(eligible),
                correct_pairs=correct,
                winner_accuracy=correct / len(eligible) if len(eligible) else np.nan,
            )
        )
    return pd.DataFrame(rows)


def contrasts(summary, protocol):
    rows = []
    selected = summary.loc[summary.protein.isin(CONFIRMATION)]
    for metric in ("kendall_tau", "winner_accuracy"):
        table = selected.groupby(["protein", "model"])[metric].mean().unstack()
        for reference in ("additive", "main_singles", "fixed_ranking"):
            for model in ("linear", "sigmoid"):
                differences = (table[model] - table[reference]).reindex(CONFIRMATION).to_numpy()
                # Four proteins permit all ordered cluster-bootstrap resamples.
                indices = np.indices((4, 4, 4, 4)).reshape(4, -1).T
                low, high = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
                rows.append(
                    dict(
                        protocol=protocol,
                        metric=metric,
                        model=model,
                        reference=reference,
                        difference=differences.mean(),
                        lower=low,
                        upper=high,
                        proteins=4,
                    )
                )
    return pd.DataFrame(rows)


def plot(fidelity, cold, warm):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "savefig.dpi": 190,
            "svg.fonttype": "none",
            "svg.hashsalt": "singles-only-research-v1",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    labels = {
        "additive": "Additive",
        "main_singles": "Main singles model",
        "sigmoid": "Simple sigmoid",
    }
    colours = {"additive": "#777777", "main_singles": "#0878be", "sigmoid": "#279b18"}
    x = np.arange(4)
    for ax, metric, title in zip(
        axes[0],
        ("w1", "local_walsh_l1", "auc_sequence"),
        ("Distribution distance", "Local Walsh distance", "C2ST with sequence features"),
    ):
        for j, (model, label) in enumerate(labels.items()):
            data = fidelity.loc[(fidelity.model == model) & fidelity.protein.isin(CONFIRMATION)]
            grouped = data.groupby("protein")[metric]
            median = grouped.median().reindex(CONFIRMATION).to_numpy()
            low = grouped.min().reindex(CONFIRMATION).to_numpy()
            high = grouped.max().reindex(CONFIRMATION).to_numpy()
            ax.errorbar(
                x + (j - 1) * 0.13,
                median,
                yerr=np.array([median - low, high - median]),
                fmt="o",
                markersize=4,
                capsize=2,
                color=colours[model],
                label=label,
            )
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(x, CONFIRMATION)
    axes[0, 0].set_ylabel("Wasserstein / measured SD")
    axes[0, 1].set_ylabel("L1 distance")
    axes[0, 2].set_ylabel("AUC")
    axes[0, 2].set_ylim(0.48, 1.02)
    axes[0, 2].axhline(0.5, color="black", ls=":", lw=0.7)
    for ax, table, title in [
        (axes[1, 0], cold.loc[cold.budget == 48], "Method ranks, 48 queries"),
        (axes[1, 1], warm.loc[warm.budget == 48], "Method ranks with DMS, 48 queries"),
        (axes[1, 2], warm.loc[warm.budget == 96], "Method ranks with DMS, 96 queries"),
    ]:
        for j, (model, label) in enumerate(labels.items()):
            values = table.loc[table.model == model].set_index("protein").kendall_tau
            ax.plot(
                x + (j - 1) * 0.13, values.reindex(CONFIRMATION), "o", color=colours[model], ms=4
            )
        fixed = table.loc[table.model == "fixed_ranking"].set_index("protein").kendall_tau
        ax.plot(x, fixed.reindex(CONFIRMATION), "_", color="black", ms=13, label="Fixed ranking")
        ax.set_xticks(x, CONFIRMATION)
        ax.set_ylabel("Kendall tau")
        ax.set_ylim(-0.75, 1.04)
        ax.set_title(title, loc="left", fontsize=11)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fixed_handle, fixed_name = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles + fixed_handle, names + fixed_name, loc="upper center", ncol=4, frameon=False
    )
    fig.text(
        0.06,
        0.025,
        "Top panels show medians and ranges across three landscape samples. "
        "Search panels rank mean regret across ten search trials.\n"
        "DMS tests start with WT and singles. "
        "Four confirmation proteins did not select model settings.",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.89, bottom=0.14, hspace=0.36, wspace=0.30)
    FIGURE.parent.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        path = FIGURE.with_suffix(f".{suffix}")
        fig.savefig(path, metadata={"Date": None} if suffix == "svg" else None)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def main():
    expected = [(name, "measured", 0) for name in (*DEVELOPMENT, *CONFIRMATION)]
    expected += [
        (name, label, seed)
        for name in (*DEVELOPMENT, *CONFIRMATION)
        for label, config in final_models(name).items()
        for seed in (SEEDS[:1] if config["strength"] == 0 else SEEDS)
    ]
    expected += [(n, "main_singles", seed) for n in (*DEVELOPMENT, *CONFIRMATION) for seed in SEEDS]
    for name, model, seed in expected:
        table = pd.read_csv(OUT / name / f"search_{model}_{seed}.csv")
        assert len(table) == 240 and np.isfinite(table.regret).all()
    warm = pd.concat([pd.read_csv(p) for p in OUT.glob("*/warm_*.csv")], ignore_index=True)
    assert len(warm) == 49 * 240
    assert (warm.spent <= warm.budget).all()
    warm_summary = transfer(warm)
    warm_summary.to_csv(OUT / "warm_summary.csv", index=False)
    cold = pd.read_csv(OUT / "search_summary.csv")
    assert len(cold.loc[cold.model == "main_singles"]) == 18
    fidelity = pd.read_csv(OUT / "fidelity.csv")
    fidelity.loc[fidelity.model == "structure_only", "w1"] = np.nan
    intervals = pd.concat([contrasts(cold, "random_start"), contrasts(warm_summary, "dms_start")])
    intervals.to_csv(OUT / "search_contrasts.csv", index=False)
    medians = fidelity.groupby(["protein", "model"]).median(numeric_only=True).reset_index()
    overview = []
    for label, table in [("random_start", cold), ("dms_start", warm_summary)]:
        grouped = table.loc[table.protein.isin(CONFIRMATION)].groupby(["model", "budget"])
        summary = grouped[["kendall_tau", "winner_accuracy"]].mean()
        summary["proteins_with_rank"] = grouped.kendall_tau.count()
        summary["proteins_with_eligible_pairs"] = grouped.winner_accuracy.count()
        summary = summary.reset_index()
        overview += records(summary.assign(protocol=label))
    result = dict(
        status="Research comparison, no change to the public default",
        protocol=json.loads((OUT / "protocol.json").read_text()),
        selected=json.loads((OUT / "selected.json").read_text()),
        inventory=records(pd.read_csv(OUT / "inventory.csv")),
        sources={
            n: json.loads((OUT / n / "metadata.json").read_text())
            for n in (*DEVELOPMENT, *CONFIRMATION)
        },
        main_priors={p.stem: json.loads(p.read_text()) for p in OUT.glob("prior_*.json")},
        fidelity=records(fidelity),
        fidelity_medians=records(medians),
        search=records(cold),
        dms_search=records(warm_summary),
        confirmation_means=overview,
        paired_protein_bootstrap=records(intervals),
        search_runs=len(expected) * 240,
        dms_search_runs=len(warm),
        search_budget_shortfalls=records(
            warm.assign(shortfall=warm.spent < warm.budget)
            .groupby(["protein", "model", "method"])
            .shortfall.sum()
            .loc[lambda x: x > 0]
            .reset_index()
        ),
        final_code_sha256={
            str(p): digest(p)
            for p in [
                Path("src/mutascape/model.py"),
                *Path("scripts").glob("*singles*.py"),
                Path("src/mutascape/completion.py"),
                Path("src/mutascape/nonlinearity.py"),
                Path("src/mutascape/benchmarks/methods.py"),
                Path("src/mutascape/benchmarks/evaluation.py"),
                Path("src/mutascape/benchmarks/library.py"),
            ]
        },
    )
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(fidelity, cold, warm_summary)
    print(pd.DataFrame(overview).round(3).to_string(index=False))
    print(intervals.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
