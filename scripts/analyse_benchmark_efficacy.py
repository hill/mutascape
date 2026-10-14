"""Summarise absolute performance, ranks and method selection for the efficacy study."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mutascape.benchmarks.methods import METHODS
from scripts.analyse_search_selection import choice_weights
from scripts.audit_search_rankings import batch_tau
from scripts.benchmark_efficacy import BUDGETS, ENSEMBLE, OUT, PROTEINS, SETTINGS, verify
from scripts.search_selection_validation import digest, save_json

BOOTSTRAPS = 5000
MODEL_NAMES = (*SETTINGS, "ensemble", "fixed")


def spearman(first, second):
    a = rankdata(np.round(first, 12), axis=-1)
    b = rankdata(np.round(second, 12), axis=-1)
    a -= a.mean(axis=-1, keepdims=True)
    b -= b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((a**2).sum(axis=-1) * (b**2).sum(axis=-1))
    return np.divide(
        (a * b).sum(axis=-1),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )


def metrics(prediction, actual, remaining):
    weights = choice_weights(prediction)
    return {
        "mspd": ((prediction - actual) ** 2).mean(axis=-1),
        "spearman": spearman(prediction, actual),
        "tau": batch_tau(prediction, actual),
        "selected_regret": (weights * actual).sum(axis=-1),
        "remaining_regret": (weights * remaining).sum(axis=-1),
    }


def limits(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)


def main():
    verify()
    files = sorted(OUT.glob("*/trials_*.csv"))
    table = pd.concat([pd.read_csv(p) for p in files])
    expected = len(PROTEINS) * 6 * 5 * 6 * 2 * len(METHODS)
    assert len(table) == expected == 17280
    assert (table.spent == table.budget).all()
    assert not table.duplicated(["protein", "model", "sample", "trial", "budget", "method"]).any()
    table.to_csv(OUT / "trials.csv", index=False)
    table.groupby(["protein", "budget", "model", "method"]).regret.mean().to_csv(
        OUT / "method_means.csv"
    )
    rng = np.random.default_rng(2026093093)
    methods = list(METHODS)
    points, intervals, contrasts, centres, ml_only = [], [], [], [], []
    for budget in BUDGETS:
        data, boot, remaining, remaining_boot = {}, {}, {}, {}
        for protein in PROTEINS:
            sub = table.loc[(table.protein == protein) & (table.budget == budget)]
            data[protein], boot[protein] = {}, {}
            measured_indices = rng.integers(30, size=(BOOTSTRAPS, 30))
            samples = rng.integers(5, size=(BOOTSTRAPS, 5))
            repeats = rng.integers(6, size=(BOOTSTRAPS, 5, 6))
            for model in ("measured", *SETTINGS):
                frame = sub.loc[sub.model == model]
                values = (
                    frame.pivot(index=["sample", "trial"], columns="method", values="regret")
                    .reindex(columns=methods)
                    .sort_index()
                    .to_numpy()
                )
                data[protein][model] = values.mean(axis=0)
                if model == "measured":
                    boot[protein][model] = values[measured_indices].mean(axis=1)
                    headroom_values = (
                        frame.pivot(
                            index=["sample", "trial"], columns="method", values="remaining_regret"
                        )
                        .reindex(columns=methods)
                        .sort_index()
                        .to_numpy()
                    )
                    remaining[protein] = headroom_values.mean(axis=0)
                    remaining_boot[protein] = headroom_values[measured_indices].mean(axis=1)
                else:
                    values = values.reshape(5, 6, len(methods))
                    boot[protein][model] = values[samples[:, :, None], repeats].mean(axis=(1, 2))
            data[protein]["ensemble"] = np.mean([data[protein][m] for m in ENSEMBLE], axis=0)
            boot[protein]["ensemble"] = np.mean([boot[protein][m] for m in ENSEMBLE], axis=0)
        for protein in PROTEINS:
            others = [p for p in PROTEINS if p != protein]
            data[protein]["fixed"] = np.mean([data[p]["measured"] for p in others], axis=0)
            choice = rng.integers(3, size=(BOOTSTRAPS, 3))
            stacked = np.stack([boot[p]["measured"] for p in others])
            boot[protein]["fixed"] = stacked[choice, np.arange(BOOTSTRAPS)[:, None]].mean(axis=1)
        boot_metrics, point_metrics = {}, {}
        for protein in PROTEINS:
            for model in MODEL_NAMES:
                result = metrics(
                    data[protein][model], data[protein]["measured"], remaining[protein]
                )
                sampled = metrics(
                    boot[protein][model], boot[protein]["measured"], remaining_boot[protein]
                )
                selected = choice_weights(data[protein][model])
                points.append(
                    dict(
                        protein=protein,
                        budget=budget,
                        model=model,
                        **result,
                        choices="|".join(np.array(methods)[selected > 0]),
                    )
                )
                boot_metrics[protein, model] = sampled
                point_metrics[protein, model] = result
                keep = np.array(
                    [m not in {"random", "greedy", "annealing", "genetic"} for m in methods]
                )
                restricted = metrics(
                    data[protein][model][keep],
                    data[protein]["measured"][keep],
                    remaining[protein][keep],
                )
                restricted_boot = metrics(
                    boot[protein][model][:, keep],
                    boot[protein]["measured"][:, keep],
                    remaining_boot[protein][:, keep],
                )
                for metric, point in restricted.items():
                    low, high = limits(restricted_boot[metric])
                    ml_only.append(
                        dict(
                            protein=protein,
                            budget=budget,
                            model=model,
                            metric=metric,
                            point=float(point),
                            low=low,
                            high=high,
                            defined_bootstraps=int(np.isfinite(restricted_boot[metric]).sum()),
                        )
                    )
                for metric, point in result.items():
                    low, high = limits(sampled[metric])
                    intervals.append(
                        dict(
                            protein=protein,
                            budget=budget,
                            model=model,
                            metric=metric,
                            point=float(point),
                            low=low,
                            high=high,
                            defined_bootstraps=int(np.isfinite(sampled[metric]).sum()),
                        )
                    )
            for model in ("measured", *SETTINGS):
                scores = -data[protein][model]
                centred = scores - scores.mean()
                for method, score in zip(methods, centred):
                    centres.append(
                        dict(
                            protein=protein,
                            budget=budget,
                            model=model,
                            method=method,
                            centred_score=score,
                        )
                    )
        protein_indices = rng.integers(4, size=(BOOTSTRAPS, 4))
        comparisons = (("ensemble", "mutascape"), ("ensemble", "fixed"), ("mutascape", "fixed"))
        for target, control in comparisons:
            for metric in ("selected_regret", "remaining_regret", "spearman", "tau", "mspd"):
                values = np.stack(
                    [
                        boot_metrics[p, target][metric] - boot_metrics[p, control][metric]
                        for p in PROTEINS
                    ]
                )
                selected = values[protein_indices, np.arange(BOOTSTRAPS)[:, None]]
                count = np.isfinite(selected).sum(axis=1)
                mean = np.divide(
                    np.nansum(selected, axis=1),
                    count,
                    out=np.full(BOOTSTRAPS, np.nan),
                    where=count > 0,
                )
                point = np.asarray(
                    [
                        point_metrics[p, target][metric] - point_metrics[p, control][metric]
                        for p in PROTEINS
                    ]
                )
                low, high = limits(mean)
                contrasts.append(
                    dict(
                        budget=budget,
                        target=target,
                        control=control,
                        metric=metric,
                        difference=float(np.nanmean(point)),
                        low=low,
                        high=high,
                        defined_proteins=int(np.isfinite(point).sum()),
                    )
                )
        for protein in PROTEINS:
            for control in ("mutascape", "fixed"):
                delta = (
                    boot_metrics[protein, "ensemble"]["selected_regret"]
                    - boot_metrics[protein, control]["selected_regret"]
                )
                low, high = limits(delta)
                intervals.append(
                    dict(
                        protein=protein,
                        budget=budget,
                        model=f"ensemble_minus_{control}",
                        metric="selected_regret",
                        point=float(
                            point_metrics[protein, "ensemble"]["selected_regret"]
                            - point_metrics[protein, control]["selected_regret"]
                        ),
                        low=low,
                        high=high,
                        defined_bootstraps=BOOTSTRAPS,
                    )
                )
    pd.DataFrame(points).to_csv(OUT / "per_protein.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "intervals.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(centres).to_csv(OUT / "centred_method_performance.csv", index=False)
    pd.DataFrame(ml_only).to_csv(OUT / "ml_only_sensitivity.csv", index=False)
    headroom = pd.concat(
        [pd.read_csv(OUT / p / "headroom.csv").assign(protein=p) for p in PROTEINS]
    )
    headroom.to_csv(OUT / "headroom.csv", index=False)
    save_json(
        OUT / "audit.json",
        dict(
            campaigns=len(table),
            warnings=int(table.warnings.sum()),
            fallback_campaigns=int((table.fallback_queries > 0).sum()),
            fallback_queries=int(table.fallback_queries.sum()),
            source_sha256={str(p): digest(p) for p in [*files, __file__]},
            manifest_sha256=digest(OUT / "manifest.json"),
        ),
    )
    print(
        pd.DataFrame(points)
        .groupby(["budget", "model"])[
            ["mspd", "spearman", "tau", "selected_regret", "remaining_regret"]
        ]
        .mean()
        .round(5)
        .to_string()
    )
    print(
        pd.DataFrame(contrasts).query('metric == "selected_regret"').round(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
