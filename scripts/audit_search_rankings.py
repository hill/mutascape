"""Audit saved search results without changing the generator or manuscript."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau

OUT = Path("results/search_ranking_audit_20260924")
OLD = Path("results/singles_only_restart_20260923")
HELD = Path("results/heldout_test_20260923")
PANEL = Path("results/heldout_panel_20260923")
DEVELOPMENT = ("DHFR", "eqFP611", "ParB-Noc", "ParD2", "CreiLOV")
CONFIRMATION = ("GB1", "TrpB4", "T7", "TEV")
BASELINES = {"random", "greedy", "annealing", "genetic"}
SEEDS = (2026092411, 2026092412, 2026092413)


def tau(left, right):
    """Keep numerical ties when comparing normalised mean regrets."""
    return float(kendalltau(np.round(left, 12), np.round(right, 12)).statistic)


def selected_regret(prediction, truth):
    winners = np.isclose(prediction, np.min(prediction), atol=1e-12, rtol=0)
    return float(np.mean(truth[winners])), int(winners.sum())


def batch_tau(left, right):
    first, second = np.triu_indices(left.shape[-1], 1)
    x = np.sign(np.round(left, 12)[..., first] - np.round(left, 12)[..., second])
    y = np.sign(np.round(right, 12)[..., first] - np.round(right, 12)[..., second])
    denominator = np.sqrt((x * x).sum(axis=-1) * (y * y).sum(axis=-1))
    return np.divide(
        (x * y).sum(axis=-1),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )


def reliability(measured):
    first = np.array(list(itertools.combinations(range(len(measured)), len(measured) // 2)))
    second = np.array([sorted(set(range(len(measured))) - set(row)) for row in first])
    values = batch_tau(measured[first].mean(axis=1), measured[second].mean(axis=1))
    return dict(
        split_half_median=float(np.nanmedian(values)),
        split_half_low=float(np.nanquantile(values, 0.05)),
        split_half_high=float(np.nanquantile(values, 0.95)),
        defined_splits=int(np.isfinite(values).sum()),
    )


def cluster_contrasts(table, index, metrics, target, controls):
    rows = []
    for metric in metrics:
        per = table.groupby([index, "model"])[metric].mean().unstack()
        for control in controls:
            complete = per[[target, control]].dropna()
            delta = (complete[target] - complete[control]).to_numpy()
            n = len(delta)
            if n == 0:
                continue
            if n <= 4:
                picks = np.array(list(itertools.product(range(n), repeat=n)))
            else:
                picks = np.random.default_rng(20260924).integers(n, size=(20000, n))
            lo, hi = np.quantile(delta[picks].mean(axis=1), [0.025, 0.975])
            rows.append(
                dict(
                    metric=metric,
                    control=control,
                    groups=n,
                    difference=delta.mean(),
                    low=lo,
                    high=hi,
                )
            )
    return pd.DataFrame(rows)


def old_audit():
    frames = []
    for path in sorted(OLD.glob("*/warm_*.csv")):
        frame = pd.read_csv(path)
        assert len(frame) == 240
        assert not frame.duplicated(["protein", "model", "seed", "budget", "method", "trial"]).any()
        assert np.isfinite(frame.regret).all()
        assert frame.regret.between(-1e-12, 1 + 1e-12).all()
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    scores.loc[scores.spent < scores.budget].groupby(["protein", "model", "method"]).size().to_csv(
        OUT / "unspent_old.csv"
    )
    means = scores.groupby(["protein", "budget", "model", "method"]).regret.mean()
    rows = []
    pairs = []
    repeatability = []
    for name, budget in itertools.product(CONFIRMATION, (48, 96)):
        local = means.loc[name, budget].unstack("model")
        fixed = pd.DataFrame({p: means.loc[p, budget, "measured"] for p in DEVELOPMENT}).mean(
            axis=1
        )
        local["fixed_ranking"] = fixed
        for subset in ("all", "ml_only"):
            selected = local if subset == "all" else local.loc[~local.index.isin(BASELINES)]
            for model in selected.columns.drop("measured"):
                regret, ties = selected_regret(
                    selected[model].to_numpy(), selected.measured.to_numpy()
                )
                winners = selected.index[
                    np.isclose(selected[model], selected[model].min(), atol=1e-12, rtol=0)
                ]
                rows.append(
                    dict(
                        protein=name,
                        budget=budget,
                        subset=subset,
                        model=model,
                        tau=tau(selected.measured, selected[model]),
                        chosen_regret=regret,
                        oracle_regret=selected.measured.min(),
                        excess_regret=regret - selected.measured.min(),
                        tied_choices=ties,
                        choices="|".join(winners),
                        random_regret=local.loc["random", "measured"],
                    )
                )
        measured = scores.loc[
            (scores.protein == name) & (scores.budget == budget) & (scores.model == "measured")
        ].pivot(index="trial", columns="method", values="regret")
        repeatability.append(dict(protein=name, budget=budget, **reliability(measured.to_numpy())))
        signs = np.array(list(itertools.product((-1, 1), repeat=len(measured))))
        rng = np.random.default_rng(2026092300)
        records = []
        for first, second in itertools.combinations(measured.columns, 2):
            delta = measured[first].to_numpy() - measured[second].to_numpy()
            estimate = delta.mean()
            low, high = np.quantile(
                delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1),
                [0.025, 0.975],
            )
            p = float((np.abs(signs @ delta / len(delta)) >= abs(estimate) - 1e-14).mean())
            records.append(
                dict(
                    first=first,
                    second=second,
                    sign=np.sign(estimate),
                    eligible=bool(low > 0 or high < 0),
                    p=p,
                    pointwise_exact=p <= 0.05,
                    sigmoid_sign=np.sign(
                        local.loc[first, "sigmoid"] - local.loc[second, "sigmoid"]
                    ),
                    fixed_sign=np.sign(
                        local.loc[first, "fixed_ranking"] - local.loc[second, "fixed_ranking"]
                    ),
                )
            )
        order = np.argsort([r["p"] for r in records])
        active = True
        for rank, item in enumerate(order):
            active = active and records[item]["p"] <= 0.05 / (len(records) - rank)
            records[item]["holm"] = active
        for eligibility in ("eligible", "pointwise_exact", "holm"):
            eligible = [r for r in records if r[eligibility]]
            pairs.append(
                dict(
                    protein=name,
                    budget=budget,
                    rule=eligibility,
                    count=len(eligible),
                    sigmoid_correct=sum(r["sign"] == r["sigmoid_sign"] for r in eligible),
                    fixed_correct=sum(r["sign"] == r["fixed_sign"] for r in eligible),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "original_per_protein.csv", index=False)
    table.groupby(["subset", "budget", "model"])[
        ["tau", "chosen_regret", "excess_regret"]
    ].mean().to_csv(OUT / "original_summary.csv")
    table.groupby(["subset", "budget", "model"]).tau.count().to_csv(
        OUT / "original_defined_counts.csv"
    )
    pd.DataFrame(repeatability).to_csv(OUT / "original_repeatability.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "original_pair_checks.csv", index=False)
    cluster_contrasts(
        table.loc[table.subset == "all"],
        "protein",
        ("tau", "chosen_regret"),
        "sigmoid",
        ("fixed_ranking", "additive", "main_singles"),
    ).to_csv(OUT / "original_contrasts.csv", index=False)
    frozen = pd.DataFrame(
        json.loads(Path("src/mutascape/data/validation/singles-only-research-v1.json").read_text())[
            "dms_search"
        ]
    )
    joined = table.loc[table.subset == "all"].merge(frozen, on=["protein", "budget", "model"])
    assert len(joined) == 40
    # The frozen JSON stores ten decimal places.
    assert (joined.tau - joined.kendall_tau).abs().max() < 5.1e-11
    csv_values = table.loc[table.subset == "all"].merge(
        pd.read_csv(OLD / "warm_summary.csv"), on=["protein", "budget", "model"]
    )
    assert (csv_values.tau - csv_values.kendall_tau).abs().max() < 1e-12
    print(
        "Old rows",
        len(scores),
        "maximum tau discrepancy",
        (joined.tau - joined.kendall_tau).abs().max(),
        flush=True,
    )
    print(
        table.loc[
            (table.subset == "all") & table.model.isin(["sigmoid", "fixed_ranking", "additive"])
        ].to_string(index=False),
        flush=True,
    )


def held_saved_audit():
    plan = json.loads((PANEL / "panel.json").read_text())
    groups = {r["landscape"]: r["genotype_space"] for r in plan["candidates"]}
    records = pd.DataFrame([json.loads(p.read_text()) for p in sorted(HELD.glob("*/*.json"))])
    rows = []
    for budget in (48, 96, 192):
        col = f"kendall_tau_{budget}"
        subset = records.loc[records.landscape.isin(plan["held_out"]) & records[col].notna()]
        names = sorted(subset.landscape.unique())
        measured = pd.DataFrame(
            {
                name: pd.read_csv(HELD / name / "measured.csv")
                .query("budget == @budget")
                .set_index("method")
                .regret
                for name in names
            }
        )
        old_fixed = None
        if budget in (48, 96):
            old_fixed = pd.DataFrame(
                {
                    p: pd.read_csv(OLD / p / "warm_measured_0.csv")
                    .query("budget == @budget")
                    .groupby("method")
                    .regret.mean()
                    for p in DEVELOPMENT
                }
            ).mean(axis=1)
        for name in names:
            others = [p for p in names if groups[p] != groups[name]]
            balanced = measured[others].T.groupby([groups[p] for p in others]).mean().mean()
            fixed = measured.drop(columns=name).mean(axis=1)
            for model, values in [
                ("fixed_assay_loo", fixed),
                ("fixed_group_loo", balanced),
                ("fixed_development", old_fixed),
            ]:
                if values is not None:
                    rows.append(
                        dict(
                            protein=name,
                            group=groups[name],
                            budget=budget,
                            model=model,
                            tau=tau(measured[name], values),
                        )
                    )
            for model, table in subset.loc[subset.landscape == name].groupby("arm"):
                rows.append(
                    dict(
                        protein=name,
                        group=groups[name],
                        budget=budget,
                        model=model,
                        tau=table[col].mean(),
                    )
                )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "held_saved_per_protein.csv", index=False)
    table.groupby(["budget", "model", "group"]).tau.mean().groupby(["budget", "model"]).agg(
        ["mean", "count"]
    ).to_csv(OUT / "held_saved_summary.csv")
    results = []
    for budget in (48, 96, 192):
        block = cluster_contrasts(
            table.loc[table.budget == budget],
            "group",
            ("tau",),
            "mutascape",
            ("shuffled_singles", "structure_only", "fixed_group_loo", "fixed_assay_loo"),
        )
        block["budget"] = budget
        results.append(block)
    pd.concat(results).to_csv(OUT / "held_saved_contrasts.csv", index=False)


def replay_one(name, model, seed, budgets):
    from scripts.heldout_panel import data
    from scripts.heldout_test import arm_values, campaign_rows

    path = OUT / "replays" / name / f"{model}_{seed}.csv"
    if path.exists():
        return
    started = time.monotonic()
    frame, design, _ = data(name)
    values = frame.fitness.to_numpy() if model == "measured" else arm_values(name, model, seed)
    rows = campaign_rows(frame, design, values, budgets)
    table = pd.DataFrame(rows)
    table.insert(0, "seed", seed)
    table.insert(0, "model", model)
    table.insert(0, "protein", name)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"Replayed {name} {model} {seed}, {time.monotonic() - started:.1f} seconds", flush=True)


def replay(jobs, only):
    plan = json.loads((PANEL / "panel.json").read_text())
    headroom = pd.read_csv(HELD / "headroom.csv").set_index("landscape")
    work = []
    for record in plan["candidates"]:
        name = record["landscape"]
        if name not in plan["held_out"] or headroom.loc[name, "start_regret"] <= 0:
            continue
        if only and name not in only.split(","):
            continue
        work.append((name, "measured", 0, record["budgets"]))
        for model in ("mutascape", "shuffled_singles"):
            work.extend((name, model, seed, record["budgets"]) for seed in SEEDS)
    Parallel(n_jobs=jobs)(delayed(replay_one)(*task) for task in work)


def replay_report():
    plan = json.loads((PANEL / "panel.json").read_text())
    groups = {r["landscape"]: r["genotype_space"] for r in plan["candidates"]}
    frames = [pd.read_csv(p) for p in sorted((OUT / "replays").glob("*/*.csv"))]
    scores = pd.concat(frames, ignore_index=True)
    assert len(scores) == 30240, "The complete replay contains 30,240 search trials"
    assert (scores.groupby(["protein", "budget", "model", "seed"]).size() == 120).all()
    assert not scores.duplicated(["protein", "model", "seed", "budget", "method", "trial"]).any()
    assert np.isfinite(scores.regret).all()
    means = scores.groupby(["protein", "budget", "model", "method"]).regret.mean()
    rows, checks, repeatability = [], [], []
    bootstraps = []
    rng = np.random.default_rng(20260924)
    bootstrap_indices = {}
    for budget in (48, 96, 192):
        names = sorted(scores.loc[scores.budget == budget, "protein"].unique())
        truth = pd.DataFrame({n: means.loc[n, budget, "measured"] for n in names})
        old_fixed = None
        if budget in (48, 96):
            old_fixed = pd.DataFrame(
                {
                    p: pd.read_csv(OLD / p / "warm_measured_0.csv")
                    .query("budget == @budget")
                    .groupby("method")
                    .regret.mean()
                    for p in DEVELOPMENT
                }
            ).mean(axis=1)
        for name in names:
            key = (budget, groups[name])
            if key not in bootstrap_indices:
                bootstrap_indices[key] = (
                    rng.integers(10, size=(2000, 10)),
                    rng.integers(3, size=(2000, 3)),
                )
            trial_samples, landscape_samples = bootstrap_indices[key]
            local = means.loc[name, budget].unstack("model")
            others = [n for n in names if groups[n] != groups[name]]
            local["fixed_group_loo"] = (
                truth[others].T.groupby([groups[n] for n in others]).mean().mean()
            )
            if old_fixed is not None:
                local["fixed_development"] = old_fixed
            raw = scores.loc[(scores.protein == name) & (scores.budget == budget)]
            measured = (
                raw.loc[raw.model == "measured"]
                .pivot(index="trial", columns="method", values="regret")
                .reindex(columns=local.index)
                .to_numpy()
            )
            stored = (
                pd.read_csv(HELD / name / "measured.csv")
                .query("budget == @budget")
                .set_index("method")
                .regret.reindex(local.index)
            )
            checks.append(
                dict(
                    protein=name,
                    budget=budget,
                    kind="measured_mean_error",
                    model="measured",
                    seed=0,
                    error=np.abs(measured.mean(axis=0) - stored).max(),
                )
            )
            repeatability.append(
                dict(protein=name, group=groups[name], budget=budget, **reliability(measured))
            )
            for subset in ("all", "ml_only"):
                chosen = local if subset == "all" else local.loc[~local.index.isin(BASELINES)]
                for model in chosen.columns.drop("measured"):
                    regret, ties = selected_regret(
                        chosen[model].to_numpy(), chosen.measured.to_numpy()
                    )
                    winners = chosen.index[
                        np.isclose(chosen[model], chosen[model].min(), atol=1e-12, rtol=0)
                    ]
                    rows.append(
                        dict(
                            protein=name,
                            group=groups[name],
                            budget=budget,
                            subset=subset,
                            model=model,
                            tau=tau(chosen.measured, chosen[model]),
                            chosen_regret=regret,
                            oracle_regret=chosen.measured.min(),
                            tied_choices=ties,
                            choices="|".join(winners),
                            random_regret=local.loc["random", "measured"],
                        )
                    )
            sampled_truth = measured[trial_samples].mean(axis=1)
            for model in ("mutascape", "shuffled_singles"):
                arrays = []
                for seed in SEEDS:
                    generated = (
                        raw.loc[(raw.model == model) & (raw.seed == seed)]
                        .pivot(index="trial", columns="method", values="regret")
                        .reindex(columns=local.index)
                        .to_numpy()
                    )
                    arrays.append(generated)
                    record = json.loads((HELD / name / f"{model}_{seed}.json").read_text())
                    checks.append(
                        dict(
                            protein=name,
                            budget=budget,
                            kind="sample_tau_error",
                            model=model,
                            seed=seed,
                            error=kendalltau(
                                raw.loc[raw.model == "measured"].groupby("method").regret.mean(),
                                raw.loc[(raw.model == model) & (raw.seed == seed)]
                                .groupby("method")
                                .regret.mean(),
                            ).statistic
                            - record[f"kendall_tau_{budget}"],
                        )
                    )
                cube = np.array(arrays)
                generated = cube[landscape_samples[:, :, None], trial_samples[:, None, :], :].mean(
                    axis=(1, 2)
                )
                sampled_tau = batch_tau(sampled_truth, generated)
                for index, value in enumerate(sampled_tau):
                    bootstraps.append(
                        dict(
                            protein=name,
                            group=groups[name],
                            budget=budget,
                            model=model,
                            sample=index,
                            tau=value,
                        )
                    )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "replay_per_protein.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "replay_checks.csv", index=False)
    assert pd.DataFrame(checks).error.abs().max() < 1e-12
    pd.DataFrame(repeatability).to_csv(OUT / "replay_repeatability.csv", index=False)
    table.groupby(["subset", "budget", "model", "group"])[["tau", "chosen_regret"]].mean().groupby(
        ["subset", "budget", "model"]
    ).agg(["mean", "count"]).to_csv(OUT / "replay_summary.csv")
    results = []
    for budget in (48, 96, 192):
        controls = ["shuffled_singles", "fixed_group_loo"]
        if budget < 192:
            controls.append("fixed_development")
        result = cluster_contrasts(
            table.loc[(table.budget == budget) & (table.subset == "all")],
            "group",
            ("tau", "chosen_regret"),
            "mutascape",
            controls,
        )
        result["budget"] = budget
        results.append(result)
    pd.concat(results).to_csv(OUT / "replay_contrasts.csv", index=False)
    boot = pd.DataFrame(bootstraps)
    boot.groupby(["protein", "budget", "model"]).tau.agg(["size", "count"]).to_csv(
        OUT / "bootstrap_defined_counts.csv"
    )
    hierarchical = []
    for budget in (48, 96, 192):
        a = (
            boot.loc[(boot.budget == budget) & (boot.model == "mutascape")]
            .groupby(["group", "sample"])
            .tau.mean()
            .unstack("group")
        )
        b = (
            boot.loc[(boot.budget == budget) & (boot.model == "shuffled_singles")]
            .groupby(["group", "sample"])
            .tau.mean()
            .unstack("group")
        )
        differences = (a - b).to_numpy()
        picks = rng.integers(a.shape[1], size=differences.shape)
        values = np.take_along_axis(differences, picks, axis=1).mean(axis=1)
        low, high = np.nanquantile(values, [0.025, 0.975])
        hierarchical.append(dict(budget=budget, groups=a.shape[1], low=low, high=high))
    pd.DataFrame(hierarchical).to_csv(OUT / "hierarchical_null_contrasts.csv", index=False)
    print(table.groupby(["subset", "budget", "model"]).tau.mean(), flush=True)


def permutation_check():
    """Check site relabelling on an eight-site binary pair model."""
    from scipy.special import expit

    rng = np.random.default_rng(20260924)
    codes = np.array(list(itertools.product((0, 1), repeat=8)))
    fields = rng.normal(size=8)
    pairs = rng.normal(size=(8, 8))
    pairs = np.triu(pairs, 1)
    pairs += pairs.T
    permutation = rng.permutation(8)
    # Relabel the fields, pair coefficients and genotype coordinates together.
    original = expit(codes @ fields + 0.15 * np.einsum("bi,ij,bj->b", codes, pairs, codes))
    relabelled = codes[:, permutation]
    permuted_pairs = pairs[np.ix_(permutation, permutation)]
    result = expit(
        relabelled @ fields[permutation]
        + 0.15 * np.einsum("bi,ij,bj->b", relabelled, permuted_pairs, relabelled)
    )
    error = float(np.max(abs(original - result)))
    assert error < 1e-14
    (OUT / "permutation_check.json").write_text(
        json.dumps(
            dict(
                sites=8,
                genotypes=len(codes),
                maximum_error=error,
                scope="A complete binary domain and a coupled permutation of the pair field",
            ),
            indent=2,
        )
        + "\n"
    )


def public_model_check():
    from mutascape.model import DMSLandscape
    from scripts.investigate_singles_only import SEEDS as ORIGINAL_SEEDS
    from scripts.investigate_singles_only import data

    maximum = 0.0
    count = 0
    for name in (*DEVELOPMENT, *CONFIRMATION):
        frame, _, metadata = data(name)
        indices = np.random.default_rng(20260924).choice(len(frame), 100, replace=False)
        for seed in ORIGINAL_SEEDS:
            model = DMSLandscape.from_singles(
                metadata["sequence"], frame.loc[frame.depth <= 1], seed=seed
            )
            values = model.evaluate_batch(frame.mutations.iloc[indices].tolist())
            original = np.load(OLD / name / f"fitness_sigmoid_{seed}.npy")[indices]
            np.testing.assert_allclose(values, original, atol=1e-12, rtol=0)
            maximum = max(maximum, float(np.max(abs(values - original))))
            count += len(values)
    (OUT / "public_model_check.json").write_text(
        json.dumps(dict(queries=count, maximum_error=maximum), indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["saved", "replay", "replay-report"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--only")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "replay":
        replay(args.jobs, args.only)
    elif args.phase == "replay-report":
        replay_report()
    else:
        old_audit()
        held_saved_audit()
        permutation_check()
        public_model_check()
    (OUT / "audit_source_sha256.txt").write_text(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "\n"
    )


if __name__ == "__main__":
    main()
