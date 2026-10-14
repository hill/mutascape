"""Estimate paired uncertainty for the independent campaign replication."""

import numpy as np
import pandas as pd

from scripts.analyse_external_generators import limits, metrics
from scripts.analyse_search_selection import choice_weights
from scripts.compare_external_generators import METHODS, digest, save_json
from scripts.replicate_generator_improvements import MODELS, OUT, verify

N_BOOT = 5000
PROTEINS = ("GB1", "CreiLOV", "eqFP611")


def main():
    manifest = verify()
    files = sorted(OUT.glob("*/*/trials_*.csv"))
    assert len(files) == len(manifest["inventory"]) * (len(MODELS) + 1) * 10
    table = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert len(table) == len(files) * 6 * 2 * len(METHODS)
    assert (table.spent == table.budget).all()
    assert not table.duplicated(
        ["protein", "domain", "model", "block", "trial", "budget", "method"]
    ).any()
    table.to_csv(OUT / "trials.csv", index=False)
    table.groupby(["protein", "domain", "budget", "model", "method"]).regret.mean().to_csv(
        OUT / "method_means.csv"
    )
    domains = {p: sorted(table.loc[table.protein == p, "domain"].unique()) for p in PROTEINS}
    methods = list(METHODS)
    models = (*MODELS, "fixed", "random_choice")
    rng = np.random.default_rng(2026106000)
    intervals, points, contrasts, ml = [], [], [], []
    ml_keep = np.array([m not in ("random", "greedy", "annealing", "genetic") for m in methods])
    for budget in (48, 96):
        mean, boot, rem, remboot, di = {}, {}, {}, {}, {}
        for protein in PROTEINS:
            n = len(domains[protein])
            di[protein] = rng.integers(n, size=(N_BOOT, n))
            bi = rng.integers(10, size=(N_BOOT, 10))
            ri = rng.integers(6, size=(N_BOOT, 10, 6))
            si = (bi[:, :, None] * 6 + ri).reshape(N_BOOT, 60)
            mi = rng.integers(60, size=(N_BOOT, 60))
            for domain in domains[protein]:
                sub = table.query("protein==@protein and domain==@domain and budget==@budget")
                for model in ("measured", *MODELS):
                    data = sub.loc[sub.model == model]
                    values = (
                        data.pivot(index=["block", "trial"], columns="method", values="regret")
                        .reindex(columns=methods)
                        .sort_index()
                        .to_numpy()
                    )
                    assert values.shape == (60, len(methods))
                    indices = mi if model == "measured" else si
                    mean[protein, domain, model] = values.mean(axis=0)
                    boot[protein, domain, model] = values[indices].mean(axis=1)
                    if model == "measured":
                        r = (
                            data.pivot(
                                index=["block", "trial"],
                                columns="method",
                                values="remaining_regret",
                            )
                            .reindex(columns=methods)
                            .sort_index()
                            .to_numpy()
                        )
                        rem[protein, domain] = r.mean(axis=0)
                        remboot[protein, domain] = r[indices].mean(axis=1)
                mean[protein, domain, "random_choice"] = np.zeros(len(methods))
                boot[protein, domain, "random_choice"] = np.zeros((N_BOOT, len(methods)))
        for protein in PROTEINS:
            others, others_boot = [], []
            for other in PROTEINS:
                if other == protein:
                    continue
                others.append(np.mean([mean[other, d, "measured"] for d in domains[other]], axis=0))
                stack = np.stack([boot[other, d, "measured"] for d in domains[other]], axis=1)
                others_boot.append(stack[np.arange(N_BOOT)[:, None], di[other]].mean(axis=1))
            for domain in domains[protein]:
                mean[protein, domain, "fixed"] = np.mean(others, axis=0)
                boot[protein, domain, "fixed"] = np.mean(others_boot, axis=0)
        pm, pb = {}, {}
        for protein in PROTEINS:
            for model in models:
                results, sampled, ml_results = [], [], []
                for domain in domains[protein]:
                    pred, actual = mean[protein, domain, model], mean[protein, domain, "measured"]
                    result = metrics(pred, actual, rem[protein, domain])
                    sampled.append(
                        metrics(
                            boot[protein, domain, model],
                            boot[protein, domain, "measured"],
                            remboot[protein, domain],
                        )
                    )
                    results.append(result)
                    ml_results.append(
                        metrics(pred[ml_keep], actual[ml_keep], rem[protein, domain][ml_keep])
                    )
                    points.append(
                        dict(
                            protein=protein,
                            domain=domain,
                            budget=budget,
                            model=model,
                            choices="|".join(np.array(methods)[choice_weights(pred) > 0]),
                            **result,
                        )
                    )
                for metric in results[0]:
                    point = np.mean([r[metric] for r in results])
                    stack = np.stack([r[metric] for r in sampled], axis=1)
                    values = stack[np.arange(N_BOOT)[:, None], di[protein]].mean(axis=1)
                    pm[protein, model, metric], pb[protein, model, metric] = point, values
                    low, high = limits(values) if np.isfinite(point) else (np.nan, np.nan)
                    intervals.append(
                        dict(
                            protein=protein,
                            budget=budget,
                            model=model,
                            metric=metric,
                            point=point,
                            low=low,
                            high=high,
                        )
                    )
                    ml.append(
                        dict(
                            protein=protein,
                            budget=budget,
                            model=model,
                            metric=metric,
                            point=np.mean([r[metric] for r in ml_results]),
                        )
                    )
        pi = rng.integers(3, size=(N_BOOT, 3))
        for model in models:
            for metric in results[0]:
                point = np.mean([pm[p, model, metric] for p in PROTEINS])
                stack = np.stack([pb[p, model, metric] for p in PROTEINS], axis=1)
                values = stack[np.arange(N_BOOT)[:, None], pi].mean(axis=1)
                low, high = limits(values) if np.isfinite(point) else (np.nan, np.nan)
                intervals.append(
                    dict(
                        protein="All",
                        budget=budget,
                        model=model,
                        metric=metric,
                        point=point,
                        low=low,
                        high=high,
                    )
                )
        for target in ("effect_pairs", "uniform_matched"):
            for control in models:
                if target == control:
                    continue
                for metric in ("selected_regret", "tau"):
                    delta = np.stack(
                        [pb[p, target, metric] - pb[p, control, metric] for p in PROTEINS], axis=1
                    )
                    values = delta[np.arange(N_BOOT)[:, None], pi].mean(axis=1)
                    low, high = limits(values)
                    point = np.mean(
                        [pm[p, target, metric] - pm[p, control, metric] for p in PROTEINS]
                    )
                    contrasts.append(
                        dict(
                            budget=budget,
                            target=target,
                            control=control,
                            metric=metric,
                            difference=point,
                            low=low,
                            high=high,
                        )
                    )
    pd.DataFrame(intervals).to_csv(OUT / "intervals.csv", index=False)
    pd.DataFrame(points).to_csv(OUT / "per_search_space.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(ml).to_csv(OUT / "ml_only.csv", index=False)
    hiv = table.query("protein=='HIV-1' and model=='measured'")
    assert (hiv.regret == 0).all()
    save_json(
        OUT / "analysis_audit.json",
        dict(
            campaigns=len(table),
            proteins_in_primary_average=PROTEINS,
            excluded_from_ranking=dict(
                protein="HIV-1",
                reason="The starting observations already contain the measured maximum",
            ),
            budget_errors=int((table.spent != table.budget).sum()),
            warnings=int(table.warnings.sum()),
            fallback_queries=int(table.fallback_queries.sum()),
            source_sha256=digest(__file__),
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
