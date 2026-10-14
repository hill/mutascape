"""Compare candidates and repeat protein-excluded selection in the bootstrap."""

import json

import numpy as np
import pandas as pd

from scripts.analyse_external_generators import ENSEMBLE, limits, metrics
from scripts.analyse_search_selection import choice_weights
from scripts.compare_external_generators import METHODS, digest, save_json
from scripts.compare_external_generators import MODELS as CONTROLS
from scripts.generator_improvement_models import MODELS, STOCHASTIC
from scripts.run_generator_improvements import OUT, REFERENCE, verify

N_BOOT = 5000
DMS_CANDIDATES = ("mutascape_additive", "mutascape_linear", "mutascape_pool", *STOCHASTIC)


def excluded_protein_choice(protein, proteins, candidates, losses):
    """Choose a configuration using outcomes from other proteins only."""
    training = np.stack(
        [np.mean([losses[p, m] for p in proteins if p != protein], axis=0) for m in candidates],
        axis=-1,
    )
    return np.argmin(np.round(training, 12), axis=-1)


def main():
    manifest = verify()
    files = sorted(OUT.glob("*/*/trials_*.csv"))
    assert len(files) == len(manifest["inventory"]) * len(MODELS) * 5
    new = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert len(new) == len(files) * 6 * 2 * len(METHODS)
    assert (new.spent == new.budget).all()
    assert not new.duplicated(
        ["protein", "domain", "model", "block", "trial", "budget", "method"]
    ).any()
    new.to_csv(OUT / "trials.csv", index=False)
    table = pd.concat([pd.read_csv(REFERENCE / "trials.csv"), new], ignore_index=True)
    table.groupby(["protein", "domain", "budget", "model", "method"]).regret.mean().to_csv(
        OUT / "method_means.csv"
    )
    proteins = sorted(table.protein.unique())
    methods = list(METHODS)
    base_models = (*CONTROLS, *MODELS)
    all_models = (
        *base_models,
        "ensemble",
        "fixed",
        "random_choice",
        "transfer_dms",
        "transfer_all",
    )
    domains = {p: sorted(table.loc[table.protein == p, "domain"].unique()) for p in proteins}
    rng = np.random.default_rng(20261006101)
    intervals, comparisons, points, selected_configs = [], [], [], []
    for budget in [48, 96]:
        mean, boot, remaining, remboot, resampled_domains = {}, {}, {}, {}, {}
        for protein in proteins:
            n = len(domains[protein])
            resampled_domains[protein] = rng.integers(n, size=(N_BOOT, n))
            blocks = rng.integers(5, size=(N_BOOT, 5))
            repeats = rng.integers(6, size=(N_BOOT, 5, 6))
            synthetic_indices = (blocks[:, :, None] * 6 + repeats).reshape(N_BOOT, 30)
            measured_indices = rng.integers(30, size=(N_BOOT, 30))
            for domain in domains[protein]:
                sub = table.query("protein==@protein and domain==@domain and budget==@budget")
                for model in ("measured", *base_models):
                    data = sub.loc[sub.model == model]
                    values = (
                        data.pivot(index=["block", "trial"], columns="method", values="regret")
                        .reindex(columns=methods)
                        .sort_index()
                        .to_numpy()
                    )
                    assert values.shape == (30, len(methods))
                    indices = measured_indices if model == "measured" else synthetic_indices
                    key = protein, domain, model
                    mean[key], boot[key] = values.mean(axis=0), values[indices].mean(axis=1)
                    if model == "measured":
                        rem = (
                            data.pivot(
                                index=["block", "trial"],
                                columns="method",
                                values="remaining_regret",
                            )
                            .reindex(columns=methods)
                            .sort_index()
                            .to_numpy()
                        )
                        remaining[protein, domain] = rem.mean(axis=0)
                        remboot[protein, domain] = rem[indices].mean(axis=1)
                mean[protein, domain, "ensemble"] = np.mean(
                    [mean[protein, domain, m] for m in ENSEMBLE], axis=0
                )
                boot[protein, domain, "ensemble"] = np.mean(
                    [boot[protein, domain, m] for m in ENSEMBLE], axis=0
                )
                mean[protein, domain, "random_choice"] = np.zeros(len(methods))
                boot[protein, domain, "random_choice"] = np.zeros((N_BOOT, len(methods)))

        def protein_mean(array, protein, model):
            values = np.stack(
                [array[protein, d, model] for d in domains[protein]], axis=0 if array is mean else 1
            )
            if array is mean:
                return values.mean(axis=0)
            return values[np.arange(N_BOOT)[:, None], resampled_domains[protein]].mean(axis=1)

        for protein in proteins:
            for domain in domains[protein]:
                mean[protein, domain, "fixed"] = np.mean(
                    [protein_mean(mean, p, "measured") for p in proteins if p != protein], axis=0
                )
                boot[protein, domain, "fixed"] = np.mean(
                    [protein_mean(boot, p, "measured") for p in proteins if p != protein], axis=0
                )
        losses, lossboots = {}, {}
        for protein in proteins:
            for model in base_models:
                per_domain, per_boot = [], []
                for domain in domains[protein]:
                    per_domain.append(
                        (
                            choice_weights(mean[protein, domain, model])
                            * mean[protein, domain, "measured"]
                        ).sum()
                    )
                    per_boot.append(
                        (
                            choice_weights(boot[protein, domain, model])
                            * boot[protein, domain, "measured"]
                        ).sum(axis=-1)
                    )
                losses[protein, model] = np.mean(per_domain)
                samples = np.stack(per_boot, axis=1)
                lossboots[protein, model] = samples[
                    np.arange(N_BOOT)[:, None], resampled_domains[protein]
                ].mean(axis=1)
        for name, candidates in [
            ("transfer_dms", DMS_CANDIDATES),
            ("transfer_all", (*DMS_CANDIDATES, *MODELS[5:])),
        ]:
            for protein in proteins:
                selected = int(excluded_protein_choice(protein, proteins, candidates, losses))
                sampled = excluded_protein_choice(protein, proteins, candidates, lossboots)
                selected_configs.append(
                    dict(
                        protein=protein,
                        budget=budget,
                        selector=name,
                        chosen=candidates[selected],
                        bootstrap_frequency=json.dumps(
                            {m: float((sampled == i).mean()) for i, m in enumerate(candidates)}
                        ),
                    )
                )
                for domain in domains[protein]:
                    mean[protein, domain, name] = mean[protein, domain, candidates[selected]]
                    array = np.stack([boot[protein, domain, m] for m in candidates], axis=1)
                    boot[protein, domain, name] = array[np.arange(N_BOOT), sampled]
        point_results, boot_results = {}, {}
        for protein in proteins:
            for model in all_models:
                rows, samples = [], []
                for domain in domains[protein]:
                    pred = mean[protein, domain, model]
                    result = metrics(
                        pred, mean[protein, domain, "measured"], remaining[protein, domain]
                    )
                    sampled = metrics(
                        boot[protein, domain, model],
                        boot[protein, domain, "measured"],
                        remboot[protein, domain],
                    )
                    points.append(
                        dict(
                            protein=protein,
                            domain=domain,
                            model=model,
                            budget=budget,
                            choices="|".join(np.array(methods)[choice_weights(pred) > 0]),
                            **result,
                        )
                    )
                    rows.append(result)
                    samples.append(sampled)
                for metric in rows[0]:
                    value = np.mean([r[metric] for r in rows])
                    stack = np.stack([r[metric] for r in samples], axis=1)
                    values = stack[np.arange(N_BOOT)[:, None], resampled_domains[protein]].mean(
                        axis=1
                    )
                    point_results[protein, model, metric] = value
                    boot_results[protein, model, metric] = values
                    low, high = limits(values) if np.isfinite(value) else (np.nan, np.nan)
                    intervals.append(
                        dict(
                            protein=protein,
                            model=model,
                            budget=budget,
                            metric=metric,
                            point=value,
                            low=low,
                            high=high,
                        )
                    )
        pi = rng.integers(len(proteins), size=(N_BOOT, len(proteins)))
        for model in all_models:
            for metric in rows[0]:
                value = np.mean([point_results[p, model, metric] for p in proteins])
                stack = np.stack([boot_results[p, model, metric] for p in proteins], axis=1)
                values = stack[np.arange(N_BOOT)[:, None], pi].mean(axis=1)
                low, high = limits(values) if np.isfinite(value) else (np.nan, np.nan)
                intervals.append(
                    dict(
                        protein="All",
                        model=model,
                        budget=budget,
                        metric=metric,
                        point=value,
                        low=low,
                        high=high,
                    )
                )
        for target in (*MODELS, "transfer_dms", "transfer_all"):
            for control in ["mutascape", "mutascape_additive", "slip4", "nk1", "rmf", "fixed"]:
                for metric in ["selected_regret", "tau"]:
                    delta = np.stack(
                        [
                            boot_results[p, target, metric] - boot_results[p, control, metric]
                            for p in proteins
                        ],
                        axis=1,
                    )
                    values = delta[np.arange(N_BOOT)[:, None], pi].mean(axis=1)
                    low, high = limits(values)
                    difference = np.mean(
                        [
                            point_results[p, target, metric] - point_results[p, control, metric]
                            for p in proteins
                        ]
                    )
                    comparisons.append(
                        dict(
                            budget=budget,
                            target=target,
                            control=control,
                            metric=metric,
                            difference=difference,
                            low=low,
                            high=high,
                        )
                    )
    pd.DataFrame(intervals).to_csv(OUT / "intervals.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(points).to_csv(OUT / "per_search_space.csv", index=False)
    pd.DataFrame(selected_configs).to_csv(OUT / "selected_configurations.csv", index=False)
    save_json(
        OUT / "analysis_audit.json",
        dict(
            campaigns=len(new),
            reference_campaigns=len(table) - len(new),
            budget_errors=int((new.spent != new.budget).sum()),
            fallback_queries=int(new.fallback_queries.sum()),
            warnings=int(new.warnings.sum()),
            analysis_sha256=digest(__file__),
            trial_sha256={str(p): digest(p) for p in files},
        ),
    )
    print(
        pd.DataFrame(intervals)
        .query("protein=='All' and budget==96")
        .pivot(index="model", columns="metric", values="point")
        .round(5)
        .to_string()
    )


if __name__ == "__main__":
    main()
