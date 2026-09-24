"""Measure transfer of method choices to measured protein landscapes."""

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd

from mutascape.benchmarks.methods import METHODS
from scripts.audit_search_rankings import BASELINES, batch_tau
from scripts.search_selection_validation import OUT, digest

METHOD_NAMES = list(METHODS)
MODELS = ["mutascape", "additive", "zero_pair", "fixed"]
BOOTSTRAPS = 5000


def choice_weights(prediction):
    """Give equal weight to method means tied at twelve decimal places."""
    prediction = np.round(np.asarray(prediction), 12)
    selected = prediction == prediction.min(axis=-1, keepdims=True)
    return selected / selected.sum(axis=-1, keepdims=True)


def selected_performance(prediction, measured):
    return (choice_weights(prediction) * measured).sum(axis=-1)


def weighted_disagreement(prediction, measured):
    """Weight each ordering error by its measured mean-regret difference."""
    first, second = np.triu_indices(np.shape(prediction)[-1], 1)
    difference = np.round(measured, 12)[..., first] - np.round(measured, 12)[..., second]
    predicted = np.round(prediction, 12)[..., first] - np.round(prediction, 12)[..., second]
    weight = np.abs(difference)
    cost = np.where(predicted == 0, 0.5, np.sign(predicted) != np.sign(difference))
    total = weight.sum(axis=-1)
    return np.divide(
        (weight * cost).sum(axis=-1),
        total,
        out=np.full(np.shape(total), np.nan),
        where=total > 0,
    )


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)


def finite_mean(values, axis=-1):
    values = np.asarray(values)
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(
        np.nansum(values, axis=axis),
        count,
        out=np.full(np.shape(count), np.nan),
        where=count > 0,
    )


def trial_array(table, protein, budget, model, metric="regret"):
    selected = table.loc[
        (table.protein == protein) & (table.budget == budget) & (table.model == model)
    ]
    pivot = selected.pivot(index=["sample", "trial"], columns="method", values=metric)
    return pivot.reindex(columns=METHOD_NAMES).sort_index().to_numpy()


def measured_bootstrap(values, indices):
    return values[indices].mean(axis=1)


def synthetic_bootstrap(values, rng, replicates=BOOTSTRAPS):
    """Resample landscape samples, then search repeats within each selection."""
    samples, repeats, methods = values.shape
    chosen = rng.integers(samples, size=(replicates, samples))
    trials = rng.integers(repeats, size=(replicates, samples, repeats))
    return values[chosen[:, :, None], trials].mean(axis=(1, 2)).reshape(replicates, methods)


def validate_runs(table, inventory):
    assert not table.duplicated(["protein", "budget", "model", "sample", "trial", "method"]).any()
    assert np.isfinite(table.regret).all()
    assert table.regret.between(-1e-12, 1 + 1e-12).all()
    assert (table.spent <= table.budget).all()
    expected = 0
    for record in inventory:
        if not record["budgets"]:
            continue
        models = ["measured"] if record["panel"] == "development" else ["measured", *MODELS[:3]]
        for budget in record["budgets"]:
            for model in models:
                values = trial_array(table, record["protein"], budget, model)
                repeats = 50 if model == "measured" else 100
                assert values.shape == (repeats, len(METHOD_NAMES)), (record, budget, model)
                assert np.isfinite(values).all()
                expected += repeats * len(METHOD_NAMES)
    assert len(table) == expected


def analyse():
    inventory = json.loads((OUT / "inventory.json").read_text())
    paths = sorted(OUT.glob("*/trials/*.csv"))
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    validate_runs(table, inventory)
    table.groupby(["protein", "budget", "model", "method"])[
        ["regret", "remaining_regret", "spent"]
    ].mean().to_csv(OUT / "method_means.csv")
    table.loc[table.spent < table.budget].to_csv(OUT / "incomplete_budgets.csv", index=False)
    rng = np.random.default_rng(2026092451)
    per_protein, contrasts, pair_rows, reliability, summaries, selections = [], [], [], [], [], []
    bootstrap_outputs = {}
    development = [r["protein"] for r in inventory if r["panel"] == "development"]
    targets = [r for r in inventory if r["panel"] != "development" and r["budgets"]]
    for budget in (48, 96):
        development_values = np.stack(
            [trial_array(table, name, budget, "measured") for name in development]
        )
        fixed = development_values.mean(axis=(0, 1))
        dev_indices = rng.integers(len(development), size=(BOOTSTRAPS, len(development)))
        dev_trials = rng.integers(50, size=(BOOTSTRAPS, len(development), 50))
        fixed_boot = development_values[dev_indices[:, :, None], dev_trials].mean(axis=(1, 2))
        local_boot = {}
        for record in targets:
            name = record["protein"]
            measured = trial_array(table, name, budget, "measured")
            measured_mean = measured.mean(axis=0)
            indices = rng.integers(50, size=(BOOTSTRAPS, 50))
            measured_boot = measured_bootstrap(measured, indices)
            remaining = trial_array(table, name, budget, "measured", "remaining_regret")
            remaining_mean = remaining.mean(axis=0)
            remaining_boot = measured_bootstrap(remaining, indices)
            predictions = {"fixed": fixed}
            predictions_boot = {"fixed": fixed_boot}
            # Controls share repeat indices. Mutascape also has sample uncertainty.
            control_indices = rng.integers(100, size=(BOOTSTRAPS, 100))
            for model in MODELS[:3]:
                values = trial_array(table, name, budget, model)
                predictions[model] = values.mean(axis=0)
                predictions_boot[model] = (
                    synthetic_bootstrap(values.reshape(5, 20, len(METHOD_NAMES)), rng)
                    if model == "mutascape"
                    else values[control_indices].mean(axis=1)
                )
            for subset in ("all", "ml_only"):
                keep = np.array(
                    [True if subset == "all" else m not in BASELINES for m in METHOD_NAMES]
                )
                real, real_boot = measured_mean[keep], measured_boot[:, keep]
                headroom, headroom_boot = remaining_mean[keep], remaining_boot[:, keep]
                for model in MODELS:
                    prediction = predictions[model][keep]
                    boot_prediction = predictions_boot[model][:, keep]
                    point = dict(
                        selected_regret=float(selected_performance(prediction, real)),
                        remaining_regret=float(selected_performance(prediction, headroom)),
                        tau=float(batch_tau(prediction, real)),
                        weighted_disagreement=float(weighted_disagreement(prediction, real)),
                    )
                    boot = dict(
                        selected_regret=selected_performance(boot_prediction, real_boot),
                        remaining_regret=selected_performance(boot_prediction, headroom_boot),
                        tau=batch_tau(boot_prediction, real_boot),
                        weighted_disagreement=weighted_disagreement(boot_prediction, real_boot),
                    )
                    row = dict(
                        protein=name,
                        panel=record["panel"],
                        budget=budget,
                        subset=subset,
                        model=model,
                        **point,
                        hindsight_regret=real.min(),
                        random_regret=measured_mean[0],
                    )
                    for metric in point:
                        row[f"{metric}_low"], row[f"{metric}_high"] = interval(boot[metric])
                        local_boot[name, subset, model, metric] = boot[metric]
                    names = np.array(METHOD_NAMES)[keep]
                    weights = choice_weights(prediction)
                    row["choices"] = "|".join(names[weights > 0])
                    per_protein.append(row)
                    frequencies = choice_weights(boot_prediction).mean(axis=0)
                    for method, weight, frequency in zip(names, weights, frequencies):
                        selections.append(
                            dict(
                                protein=name,
                                budget=budget,
                                subset=subset,
                                model=model,
                                method=method,
                                weight=weight,
                                bootstrap_frequency=frequency,
                            )
                        )
                for control in ("fixed", "additive", "zero_pair"):
                    for metric in point:
                        target_boot = local_boot[name, subset, "mutascape", metric]
                        control_boot = local_boot[name, subset, control, metric]
                        low, high = interval(target_boot - control_boot)
                        target_row = next(
                            r for r in reversed(per_protein) if r["model"] == "mutascape"
                        )
                        control_row = next(
                            r for r in reversed(per_protein) if r["model"] == control
                        )
                        contrasts.append(
                            dict(
                                scope="protein",
                                protein=name,
                                panel=record["panel"],
                                budget=budget,
                                subset=subset,
                                metric=metric,
                                control=control,
                                proteins=1,
                                difference=target_row[metric] - control_row[metric],
                                low=low,
                                high=high,
                            )
                        )
            for first, second in combinations(range(len(METHOD_NAMES)), 2):
                delta = measured_mean[first] - measured_mean[second]
                low, high = interval(measured_boot[:, first] - measured_boot[:, second])
                pair_rows.append(
                    dict(
                        protein=name,
                        panel=record["panel"],
                        budget=budget,
                        first=METHOD_NAMES[first],
                        second=METHOD_NAMES[second],
                        difference=delta,
                        low=low,
                        high=high,
                        excludes_zero=bool(low > 0 or high < 0),
                        equivalent_001=bool(low >= -0.01 and high <= 0.01),
                        mutascape_difference=predictions["mutascape"][first]
                        - predictions["mutascape"][second],
                    )
                )
            orders = np.array([rng.permutation(50) for _ in range(1000)])
            halves = batch_tau(
                measured[orders[:, :25]].mean(axis=1), measured[orders[:, 25:]].mean(axis=1)
            )
            reliability.append(
                dict(
                    protein=name,
                    budget=budget,
                    split_half_median=float(np.nanmedian(halves)),
                    split_half_low=float(np.nanquantile(halves, 0.05)),
                    split_half_high=float(np.nanquantile(halves, 0.95)),
                    defined_splits=int(np.isfinite(halves).sum()),
                )
            )
        points = pd.DataFrame(per_protein).query("budget == @budget")
        for panel in ("new", "repeat"):
            names = [r["protein"] for r in targets if r["panel"] == panel]
            picks = rng.integers(len(names), size=(BOOTSTRAPS, len(names)))
            for subset in ("all", "ml_only"):
                summary_boot = {}
                for model in MODELS:
                    selected = points.query(
                        "panel == @panel and subset == @subset and model == @model"
                    )
                    for metric in (
                        "selected_regret",
                        "remaining_regret",
                        "tau",
                        "weighted_disagreement",
                    ):
                        values = np.stack(
                            [local_boot[n, subset, model, metric] for n in names], axis=1
                        )
                        resampled = values[np.arange(BOOTSTRAPS)[:, None], picks]
                        mean_boot = finite_mean(resampled)
                        summary_boot[model, metric] = mean_boot
                        low, high = interval(mean_boot)
                        summaries.append(
                            dict(
                                panel=panel,
                                budget=budget,
                                subset=subset,
                                model=model,
                                metric=metric,
                                estimate=selected[metric].mean(),
                                low=low,
                                high=high,
                                proteins=int(selected[metric].notna().sum()),
                            )
                        )
                        bootstrap_outputs[f"{panel}_{budget}_{subset}_{model}_{metric}"] = mean_boot
                for control in ("fixed", "additive", "zero_pair"):
                    for metric in (
                        "selected_regret",
                        "remaining_regret",
                        "tau",
                        "weighted_disagreement",
                    ):
                        low, high = interval(
                            summary_boot["mutascape", metric] - summary_boot[control, metric]
                        )
                        left = points.query(
                            "panel == @panel and subset == @subset and model == 'mutascape'"
                        )
                        right = points.query(
                            "panel == @panel and subset == @subset and model == @control"
                        )
                        contrasts.append(
                            dict(
                                scope="panel",
                                protein="",
                                panel=panel,
                                budget=budget,
                                subset=subset,
                                metric=metric,
                                control=control,
                                proteins=len(names),
                                difference=left[metric].mean() - right[metric].mean(),
                                low=low,
                                high=high,
                            )
                        )
    for filename, rows in [
        ("per_protein", per_protein),
        ("contrasts", contrasts),
        ("pair_differences", pair_rows),
        ("repeatability", reliability),
        ("summary", summaries),
        ("selections", selections),
    ]:
        pd.DataFrame(rows).to_csv(OUT / f"{filename}.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_summaries.npz", **bootstrap_outputs)
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(
            dict(
                run_manifest_sha256=digest(OUT / "manifest.json"),
                analysis_sha256=digest(__file__),
                bootstraps=BOOTSTRAPS,
                seed=2026092451,
                rows=len(table),
                trial_hashes={str(p): digest(p) for p in paths},
                synthetic_resampling=(
                    "Hierarchical Mutascape samples and repeats. Paired deterministic controls."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        pd.DataFrame(summaries)
        .query("subset == 'all' and metric == 'selected_regret'")
        .to_string(index=False)
    )
    print(
        pd.DataFrame(contrasts)
        .query("scope == 'panel' and subset == 'all' and metric == 'selected_regret'")
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    analyse()


if __name__ == "__main__":
    main()
