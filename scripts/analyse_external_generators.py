"""Assess search-method selection on measured landscapes with paired uncertainty."""

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.analyse_search_selection import choice_weights
from scripts.audit_search_rankings import batch_tau
from scripts.compare_external_generators import (
    BUDGETS,
    METHODS,
    MODELS,
    OUT,
    PROTEINS,
    digest,
    save_json,
    verify,
)

LABELS = {
    "mutascape": "Mutascape",
    "mutascape_additive": "Mutascape additive",
    "mutascape_zero_pair": "Mutascape zero-pair sigmoid",
    "mutascape_linear": "Mutascape linear + pairs",
    "mutascape_pool": "Mutascape pooling",
    "ensemble": "Mutascape configuration average",
    "slip_raw": "SLIP untuned",
    "slip4": "SLIP, horizon 4",
    "slip8": "SLIP, horizon 8",
    "nk0": "NK, K = 0",
    "nk1": "NK, K = 1",
    "nk3": "NK, K = 3",
    "rmf": "Rough Mount Fuji",
    "hoc": "House-of-Cards",
    "fixed": "Fixed method from other proteins",
    "random_choice": "Random method choice",
    "measured": "Measured",
}
ENSEMBLE = ("mutascape", "mutascape_linear", "mutascape_pool")
N_BOOT = 5000


def spearman(x, y):
    a, b = rankdata(np.round(x, 12), axis=-1), rankdata(np.round(y, 12), axis=-1)
    a -= a.mean(axis=-1, keepdims=True)
    b -= b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.divide(
        (a * b).sum(axis=-1),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )


def metrics(prediction, measured, remaining):
    weights = choice_weights(prediction)
    selected = (weights * measured).sum(axis=-1)
    return dict(
        selected_regret=selected,
        selection_loss=selected - measured.min(axis=-1),
        remaining_regret=(weights * remaining).sum(axis=-1),
        tau=batch_tau(prediction, measured),
        spearman=spearman(prediction, measured),
        mspd=((prediction - measured) ** 2).mean(axis=-1),
    )


def limits(values):
    valid = np.asarray(values)[np.isfinite(values)]
    return np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)


def main():
    manifest = verify()
    files = sorted(p for p in OUT.glob("*/*/trials_*.csv") if "smoke" not in p.name)
    expected_files = len(manifest["inventory"]) * (len(MODELS) + 1) * 5
    assert len(files) == expected_files, (len(files), expected_files)
    table = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert len(table) == expected_files * 6 * len(BUDGETS) * len(METHODS)
    assert (table.spent == table.budget).all()
    assert not table.duplicated(
        ["protein", "domain", "model", "block", "trial", "budget", "method"]
    ).any()
    table.to_csv(OUT / "trials.csv", index=False)
    table.groupby(["protein", "domain", "budget", "model", "method"]).regret.mean().to_csv(
        OUT / "method_means.csv"
    )
    rng = np.random.default_rng(2026100699)
    points, interval_rows, contrast_rows, ml_rows = [], [], [], []
    analysis_models = (*MODELS, "ensemble", "fixed", "random_choice")
    methods = list(METHODS)
    ml_keep = np.array([m not in ("random", "greedy", "annealing", "genetic") for m in methods])
    for budget in BUDGETS:
        means, boots, remaining, remaining_boot, domain_indices = {}, {}, {}, {}, {}
        domains = {p: sorted(table.loc[table.protein == p, "domain"].unique()) for p in PROTEINS}
        for protein in PROTEINS:
            n_domains = len(domains[protein])
            domain_indices[protein] = rng.integers(n_domains, size=(N_BOOT, n_domains))
            landscape_indices = rng.integers(5, size=(N_BOOT, 5))
            repeat_indices = rng.integers(6, size=(N_BOOT, 5, 6))
            synthetic_indices = (landscape_indices[:, :, None] * 6 + repeat_indices).reshape(
                N_BOOT, 30
            )
            measured_indices = rng.integers(30, size=(N_BOOT, 30))
            for domain in domains[protein]:
                sub = table.loc[
                    (table.protein == protein) & (table.domain == domain) & (table.budget == budget)
                ]
                for model in ("measured", *MODELS):
                    data = sub.loc[sub.model == model]
                    values = (
                        data.pivot(index=["block", "trial"], columns="method", values="regret")
                        .reindex(columns=methods)
                        .sort_index()
                        .to_numpy()
                    )
                    assert values.shape == (30, len(methods))
                    key = protein, domain, model
                    means[key] = values.mean(axis=0)
                    indices = measured_indices if model == "measured" else synthetic_indices
                    boots[key] = values[indices].mean(axis=1)
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
                        remaining_boot[protein, domain] = rem[indices].mean(axis=1)
                means[protein, domain, "ensemble"] = np.mean(
                    [means[protein, domain, m] for m in ENSEMBLE], axis=0
                )
                boots[protein, domain, "ensemble"] = np.mean(
                    [boots[protein, domain, m] for m in ENSEMBLE], axis=0
                )
                means[protein, domain, "random_choice"] = np.zeros(len(methods))
                boots[protein, domain, "random_choice"] = np.zeros((N_BOOT, len(methods)))
        for protein in PROTEINS:
            other_means, other_boot = [], []
            for other in PROTEINS:
                if other == protein:
                    continue
                other_means.append(
                    np.mean([means[other, d, "measured"] for d in domains[other]], axis=0)
                )
                array = np.stack([boots[other, d, "measured"] for d in domains[other]], axis=1)
                other_boot.append(
                    array[np.arange(N_BOOT)[:, None], domain_indices[other]].mean(axis=1)
                )
            for domain in domains[protein]:
                means[protein, domain, "fixed"] = np.mean(other_means, axis=0)
                boots[protein, domain, "fixed"] = np.mean(other_boot, axis=0)
        point_by_protein, boot_by_protein = {}, {}
        for protein in PROTEINS:
            for model in analysis_models:
                domain_points, domain_boots, domain_ml = [], [], []
                for domain in domains[protein]:
                    pred, actual = means[protein, domain, model], means[protein, domain, "measured"]
                    result = metrics(pred, actual, remaining[protein, domain])
                    sampled = metrics(
                        boots[protein, domain, model],
                        boots[protein, domain, "measured"],
                        remaining_boot[protein, domain],
                    )
                    domain_points.append(result)
                    domain_boots.append(sampled)
                    weights = choice_weights(pred)
                    points.append(
                        dict(
                            protein=protein,
                            domain=domain,
                            budget=budget,
                            model=model,
                            **result,
                            choices="|".join(np.array(methods)[weights > 0]),
                        )
                    )
                    domain_ml.append(
                        metrics(pred[ml_keep], actual[ml_keep], remaining[protein, domain][ml_keep])
                    )
                for metric in domain_points[0]:
                    point = np.mean([d[metric] for d in domain_points])
                    samples = np.stack([d[metric] for d in domain_boots], axis=1)
                    values = samples[np.arange(N_BOOT)[:, None], domain_indices[protein]].mean(
                        axis=1
                    )
                    point_by_protein[protein, model, metric] = point
                    boot_by_protein[protein, model, metric] = values
                    low, high = limits(values) if np.isfinite(point) else (np.nan, np.nan)
                    interval_rows.append(
                        dict(
                            protein=protein,
                            budget=budget,
                            model=model,
                            metric=metric,
                            point=point,
                            low=low,
                            high=high,
                            defined_bootstraps=int(np.isfinite(values).sum()),
                        )
                    )
                    ml_rows.append(
                        dict(
                            protein=protein,
                            budget=budget,
                            model=model,
                            metric=metric,
                            point=np.mean([d[metric] for d in domain_ml]),
                        )
                    )
        protein_indices = rng.integers(len(PROTEINS), size=(N_BOOT, len(PROTEINS)))
        for model in analysis_models:
            for metric in domain_points[0]:
                point = np.mean([point_by_protein[p, model, metric] for p in PROTEINS])
                stacked = np.stack([boot_by_protein[p, model, metric] for p in PROTEINS], axis=1)
                values = stacked[np.arange(N_BOOT)[:, None], protein_indices].mean(axis=1)
                low, high = limits(values) if np.isfinite(point) else (np.nan, np.nan)
                interval_rows.append(
                    dict(
                        protein="All",
                        budget=budget,
                        model=model,
                        metric=metric,
                        point=point,
                        low=low,
                        high=high,
                        defined_bootstraps=int(np.isfinite(values).sum()),
                    )
                )
        comparisons = [
            (t, c)
            for t in ("mutascape", "ensemble", "mutascape_pool")
            for c in ("mutascape_additive", "slip4", "nk1", "rmf", "fixed", "random_choice")
        ]
        for target, control in comparisons:
            for metric in domain_points[0]:
                delta = np.stack(
                    [
                        boot_by_protein[p, target, metric] - boot_by_protein[p, control, metric]
                        for p in PROTEINS
                    ],
                    axis=1,
                )
                values = delta[np.arange(N_BOOT)[:, None], protein_indices].mean(axis=1)
                point = np.mean(
                    [
                        point_by_protein[p, target, metric] - point_by_protein[p, control, metric]
                        for p in PROTEINS
                    ]
                )
                low, high = limits(values) if np.isfinite(point) else (np.nan, np.nan)
                contrast_rows.append(
                    dict(
                        budget=budget,
                        target=target,
                        control=control,
                        metric=metric,
                        difference=point,
                        low=low,
                        high=high,
                        defined_bootstraps=int(np.isfinite(values).sum()),
                    )
                )
    pd.DataFrame(points).to_csv(OUT / "per_search_space.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(OUT / "intervals.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(ml_rows).to_csv(OUT / "ml_only.csv", index=False)
    audit = dict(
        campaigns=len(table),
        files=len(files),
        budget_errors=int((table.spent != table.budget).sum()),
        fallback_queries=int(table.fallback_queries.sum()),
        warnings=int(table.warnings.sum()),
        bootstrap_samples=N_BOOT,
        source_sha256=digest(__file__),
        manifest_sha256=digest(OUT / "manifest.json"),
        trial_hashes={str(p): digest(p) for p in files},
    )
    save_json(OUT / "analysis_audit.json", audit)
    summary = pd.DataFrame(interval_rows).query("protein=='All' and budget==96")
    print(summary.pivot(index="model", columns="metric", values="point").round(4).to_string())
    print(json.dumps({k: v for k, v in audit.items() if k != "trial_hashes"}, indent=2))


if __name__ == "__main__":
    main()
