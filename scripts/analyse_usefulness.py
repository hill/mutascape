"""Summarise uncertainty and plot the fixed usefulness experiment."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.analyse_search_selection import choice_weights
from scripts.audit_search_rankings import batch_tau
from scripts.build_quantile_manuscript import BLUE, GREEN, STYLE
from scripts.test_usefulness import OUT, PROTEINS, SEARCH_MODELS, SETTINGS, verify

BOOTSTRAPS = 5000
MODELS = (*SEARCH_MODELS, "scenario_ensemble", "fixed_other_proteins")


def crossed_medians(values, calibration_indices, seed_indices):
    """Resample shared landscape seeds across all selected calibration samples."""
    return np.median(values[calibration_indices[:, :, None], seed_indices[:, None, :]], axis=(1, 2))


def uncertainty():
    table = pd.read_csv(OUT / "search_trials.csv")
    methods = sorted(table.method.unique())
    rng = np.random.default_rng(2026093091)
    measured = {}
    synthetic = {}
    for protein in PROTEINS:
        subset = table.loc[table.protein == protein]
        real = (
            subset.loc[subset.model == "measured"]
            .pivot(index=["sample", "trial"], columns="method", values="regret")
            .reindex(columns=methods)
            .to_numpy()
        )
        assert real.shape == (30, 12)
        measured[protein] = real[rng.integers(30, size=(BOOTSTRAPS, 30))].mean(axis=1)
        samples = rng.integers(3, size=(BOOTSTRAPS, 3))
        repeats = rng.integers(10, size=(BOOTSTRAPS, 3, 10))
        synthetic[protein] = {}
        for model in SEARCH_MODELS:
            values = (
                subset.loc[subset.model == model]
                .pivot(index=["sample", "trial"], columns="method", values="regret")
                .reindex(columns=methods)
                .sort_index()
                .to_numpy()
                .reshape(3, 10, 12)
            )
            synthetic[protein][model] = values[samples[:, :, None], repeats].mean(axis=(1, 2))
        synthetic[protein]["scenario_ensemble"] = np.mean(
            [synthetic[protein][model] for model in SETTINGS], axis=0
        )
    for protein in PROTEINS:
        others = np.array([rankdata(measured[p], axis=-1) for p in PROTEINS if p != protein])
        choices = rng.integers(4, size=(BOOTSTRAPS, 4))
        synthetic[protein]["fixed_other_proteins"] = others[
            choices, np.arange(BOOTSTRAPS)[:, None]
        ].mean(axis=1)
    point = pd.read_csv(OUT / "search_summary.csv").set_index(["protein", "model"])
    boot = {}
    intervals = []
    for model in MODELS:
        selected, tau = [], []
        for protein in PROTEINS:
            prediction, actual = synthetic[protein][model], measured[protein]
            performance = (choice_weights(prediction) * actual).sum(axis=1)
            correlation = batch_tau(prediction, actual)
            selected.append(performance)
            tau.append(correlation)
            for metric, values in (("selected_regret", performance), ("tau", correlation)):
                finite = values[np.isfinite(values)]
                low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
                intervals.append(
                    dict(
                        protein=protein,
                        model=model,
                        metric=metric,
                        point=point.loc[(protein, model), metric],
                        low=low,
                        high=high,
                    )
                )
        boot[model] = dict(selected_regret=np.array(selected), tau=np.array(tau))
    pd.DataFrame(intervals).to_csv(OUT / "search_intervals.csv", index=False)
    proteins = rng.integers(len(PROTEINS), size=(BOOTSTRAPS, len(PROTEINS)))
    comparisons = [
        ("conditioned_192", "quantile_192"),
        ("conditioned_192", "default"),
        ("conditioned_192", "mean_192"),
        ("conditioned_192", "fixed_other_proteins"),
        ("scenario_ensemble", "default"),
        ("scenario_ensemble", "additive"),
        ("scenario_ensemble", "fixed_other_proteins"),
    ]
    contrasts = []
    for target, control in comparisons:
        for metric in ("selected_regret", "tau"):
            values = boot[target][metric] - boot[control][metric]
            selected = values[proteins, np.arange(BOOTSTRAPS)[:, None]]
            counts = np.isfinite(selected).sum(axis=1)
            means = np.divide(
                np.nansum(selected, axis=1),
                counts,
                out=np.full(BOOTSTRAPS, np.nan),
                where=counts > 0,
            )
            differences = (
                point.xs(target, level="model")[metric] - point.xs(control, level="model")[metric]
            )
            low, high = np.nanquantile(means, [0.025, 0.975])
            contrasts.append(
                dict(
                    target=target,
                    control=control,
                    metric=metric,
                    difference=differences.mean(),
                    low=low,
                    high=high,
                    defined_proteins=int(differences.notna().sum()),
                    defined_bootstraps=int(np.isfinite(means).sum()),
                )
            )
    pd.DataFrame(contrasts).to_csv(OUT / "search_contrasts.csv", index=False)
    fidelity = pd.read_csv(OUT / "fidelity.csv")
    fidelity_boot = {}
    for protein in PROTEINS:
        sample_indices = rng.integers(3, size=(BOOTSTRAPS, 3))
        seed_indices = rng.integers(3, size=(BOOTSTRAPS, 3))
        fidelity_boot[protein] = {}
        for model in ("quantile_192", "conditioned_192"):
            subset = fidelity.loc[(fidelity.protein == protein) & (fidelity.model == model)]
            values = subset.pivot(index="pilot", columns="sample", values="auc").to_numpy()
            fidelity_boot[protein][model] = crossed_medians(values, sample_indices, seed_indices)
    differences = np.array(
        [fidelity_boot[p]["conditioned_192"] - fidelity_boot[p]["quantile_192"] for p in PROTEINS]
    )
    means = differences[proteins, np.arange(BOOTSTRAPS)[:, None]].mean(axis=1)
    median = fidelity.groupby(["protein", "model"]).auc.median().unstack()
    delta = median.conditioned_192 - median.quantile_192
    (OUT / "fidelity_contrast.json").write_text(
        json.dumps(
            dict(
                target="conditioned_192",
                control="quantile_192",
                metric="auc",
                difference=delta.mean(),
                low=np.quantile(means, 0.025),
                high=np.quantile(means, 0.975),
            ),
            indent=2,
        )
        + "\n"
    )
    fallback = table.groupby(["protein", "model", "method"]).agg(
        campaigns=("spent", "size"),
        fallback_campaigns=("fallback_queries", lambda x: (x > 0).sum()),
        mean_fallback_queries=("fallback_queries", "mean"),
    )
    fallback.to_csv(OUT / "search_fallback.csv")


def plot():
    plt.rcParams.update(STYLE)
    fidelity = pd.read_csv(OUT / "fidelity.csv")
    intervals = pd.read_csv(OUT / "search_intervals.csv")
    intervals = intervals.loc[intervals.metric == "selected_regret"].set_index(["protein", "model"])
    labels = {
        "default": "Singles-only Mutascape",
        "quantile_192": "Combination-calibrated, 192 measurements",
        "conditioned_192": "Conditioned pairs, 192 measurements",
    }
    colours = {"default": BLUE, "quantile_192": "#a26b22", "conditioned_192": GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"wspace": 0.34})
    for index, model in enumerate(labels):
        location = np.arange(len(PROTEINS)) + (index - 1) * 0.19
        subset = (
            fidelity.loc[fidelity.model == model]
            .groupby("protein")
            .auc.agg(["median", "min", "max"])
            .loc[list(PROTEINS)]
        )
        axes[0].errorbar(
            subset["median"],
            location,
            xerr=np.array([subset["median"] - subset["min"], subset["max"] - subset["median"]]),
            fmt="o",
            color=colours[model],
            capsize=3,
            ms=4,
            label=labels[model],
        )
        subset = intervals.xs(model, level="model").loc[list(PROTEINS)]
        axes[1].hlines(location, subset.low, subset.high, color=colours[model], linewidth=1)
        axes[1].plot(subset.point, location, "o", color=colours[model], ms=4)
    for ax in axes:
        ax.set_yticks(np.arange(len(PROTEINS)), PROTEINS)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axvline(0.5, color="0.6", linestyle="--", linewidth=0.8)
    axes[0].set_xlim(0.47, 1.03)
    axes[0].set_xlabel("C2ST AUC, lower is better")
    axes[0].set_title("(a) Held-out fitness agreement")
    axes[1].set_xlabel("Measured regret after method selection")
    axes[1].set_title("(b) Search-method selection")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", frameon=False, ncol=1)
    fig.subplots_adjust(bottom=0.29)
    fig.savefig(OUT / "usefulness_results.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "usefulness_results.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    verify()
    uncertainty()
    plot()
