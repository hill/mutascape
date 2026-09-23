"""Score the frozen held-out panel against the generator and its controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau

from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features
from mutascape.model import DMSLandscape
from scripts.heldout_panel import OUT as PANEL
from scripts.heldout_panel import SEARCH_SEEDS, data
from scripts.investigate_singles_only import diagnostics, digest
from scripts.singles_warm_start import DMSOracle

OUT = Path("results/heldout_test_20260923")
TRIALS = 10
TOP_FRACTION = 0.01
# Matched to the generator's default pair strength, so the standard oracles get
# the same nominal amount of epistasis rather than a tuned advantage.
COUPLING = 0.3
ROUGHNESS = 0.3
ARMS = (
    "mutascape",
    "shuffled_singles",
    "structure_only",
    "additive",
    "potts",
    "rough_mount_fuji",
    "nk_k1",
    "nk_k2",
)
# These arms claim to reproduce the measured singles. The others do not, and a
# campaign on them simply starts from their own singles.
EXACT_SINGLES = ("mutascape",)


def panel():
    return json.loads((PANEL / "panel.json").read_text())


def budgets_for(record):
    return [int(budget) for budget in record["budgets"]]


def site_fields(frame, codes):
    """Per-residue fitness effect at each site, read off the measured singles."""
    depth = frame.depth.to_numpy()
    wild_row = codes[int(np.flatnonzero(depth == 0)[0])]
    wild_type = float(frame.fitness.to_numpy()[depth == 0][0])
    fields = [np.zeros(int(size) + 1) for size in codes.max(axis=0)]
    for row, fitness in zip(codes[depth == 1], frame.fitness.to_numpy()[depth == 1]):
        site = int(np.flatnonzero(row != wild_row)[0])
        fields[site][row[site]] = fitness - wild_type
    return wild_type, wild_row, fields


def additive_values(wild_type, fields, codes):
    return wild_type + sum(field[codes[:, site]] for site, field in enumerate(fields))


def arm_values(name, arm, seed):
    """Fitness of every measured genotype under one arm."""
    frame, design, meta = data(name)
    codes = design["codes"]
    shape = tuple(int(size) for size in design["shape"])
    singles = frame.loc[frame.depth <= 1].copy()
    if arm == "shuffled_singles":
        rng = np.random.default_rng(seed + 7001)
        body = singles.depth == 1
        singles.loc[body, "fitness"] = rng.permutation(singles.loc[body, "fitness"].to_numpy())
    if arm == "structure_only":
        rng = np.random.default_rng(seed + 9101)
        body = singles.depth == 1
        spread = float(singles.loc[body, "fitness"].std())
        centre = float(singles.loc[body, "fitness"].mean())
        singles.loc[body, "fitness"] = centre + spread * rng.standard_normal(int(body.sum()))
    if arm in ("mutascape", "shuffled_singles", "structure_only"):
        model = DMSLandscape.from_singles(meta["sequence"], singles, seed=seed)
        return model.evaluate_batch(frame.mutations.tolist())
    wild_type, wild_row, fields = site_fields(frame, codes)
    if arm == "additive":
        return additive_values(wild_type, fields, codes)
    rng = np.random.default_rng(seed + 4201)
    if arm in ("potts", "rough_mount_fuji"):
        values = additive_values(wild_type, fields, codes)
        spread = float(np.std(values)) or 1.0
        if arm == "rough_mount_fuji":
            return values + ROUGHNESS * spread * rng.standard_normal(len(values))
        for first in range(len(shape)):
            for second in range(first + 1, len(shape)):
                table = rng.standard_normal((shape[first], shape[second]))
                table[wild_row[first], :] = 0.0
                table[:, wild_row[second]] = 0.0
                values = values + COUPLING * spread * table[codes[:, first], codes[:, second]]
        return values
    k = int(arm.removeprefix("nk_k"))
    sites = len(shape)
    total = np.zeros(len(codes))
    for site in range(sites):
        others = [index for index in range(sites) if index != site]
        picked = rng.choice(others, size=min(k, len(others)), replace=False)
        group = sorted([site, *picked.tolist()])
        table = rng.standard_normal(tuple(shape[index] for index in group))
        total = total + table[tuple(codes[:, index] for index in group)]
    # Put NK on the measured scale so regret is comparable across arms.
    measured = frame.fitness.to_numpy()
    total = (total - total.mean()) / (np.std(total) or 1.0)
    return measured.mean() + np.std(measured) * total


def campaign_rows(frame, design, values, budgets):
    """Run every method from a campaign that already holds wild type and singles.

    Unlike the warm start on measured data, an arm is free to disagree with the
    measured singles. The campaign then starts from that arm's own singles.
    """
    flat = np.ravel_multi_index(design["codes"].T, tuple(design["shape"]))
    selected = np.argsort(flat)
    depth = frame.depth.to_numpy()[selected]
    known = np.flatnonzero(depth <= 1)
    unknown = np.flatnonzero(depth > 1)
    values = np.asarray(values, dtype=float)[selected]
    codes = design["codes"][selected]
    threshold = float(np.quantile(values, 1.0 - TOP_FRACTION))
    space = SearchSpace(
        codes,
        tuple(int(size) for size in design["shape"]),
        selected,
        int(known[np.argmax(values[known])]),
        {tuple(int(value) for value in code): index for index, code in enumerate(codes)},
    )
    features = make_features(space)
    span = max(float(np.ptp(values)), 1e-12)
    rows = []
    for budget in budgets:
        for method, function in METHODS.items():
            for trial in range(TRIALS):
                search_seed = 2026092400 + 137 * trial
                oracle = DMSOracle(values, budget, known)
                if method == "random":
                    rng = np.random.default_rng(search_seed)
                    oracle.query(rng.choice(unknown, size=budget, replace=False))
                else:
                    function(space, oracle, features, search_seed)
                assert oracle.spent <= budget
                rows.append(
                    dict(
                        budget=budget,
                        method=method,
                        trial=trial,
                        regret=(values.max() - oracle.best()) / span,
                        success=float(oracle.best() >= threshold),
                    )
                )
    return rows


def ranking(rows):
    table = pd.DataFrame(rows).groupby(["budget", "method"])[["regret", "success"]].mean()
    return table.sort_index()


def measured_ranking(name, budgets):
    path = OUT / name / "measured.csv"
    if path.exists():
        return pd.read_csv(path, index_col=[0, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame, design, _ = data(name)
    table = ranking(campaign_rows(frame, design, frame.fitness.to_numpy(), budgets))
    table.to_csv(path)
    print(f"Measured {name}", flush=True)
    return table


def evaluate(name, arm, seed, budgets):
    path = OUT / name / f"{arm}_{seed}.json"
    if path.exists():
        return
    frame, design, _ = data(name)
    values = np.asarray(arm_values(name, arm, seed), dtype=float)
    if arm in EXACT_SINGLES:
        np.testing.assert_allclose(
            values[frame.depth <= 1], frame.loc[frame.depth <= 1, "fitness"], atol=1e-9
        )
    measured = measured_ranking(name, budgets)
    generated = ranking(campaign_rows(frame, design, values, budgets))
    record = dict(landscape=name, arm=arm, seed=seed, **diagnostics(name, values, frame, design))
    for budget in budgets:
        left = measured.loc[budget]
        right = generated.loc[budget].reindex(left.index)
        record[f"kendall_tau_{budget}"] = float(kendalltau(left.regret, right.regret).statistic)
        record[f"success_tau_{budget}"] = float(kendalltau(left.success, right.success).statistic)
        record[f"success_rate_{budget}"] = float(right.success.mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Scored {name} {arm} {seed}", flush=True)


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = panel()
    record = dict(
        purpose=plan["purpose"],
        panel=plan["panel"],
        held_out=plan["held_out"],
        arms=ARMS,
        fixed_ranking=(
            "Reported from the measured rankings of the other panel landscapes, "
            "leaving out the landscape being scored"
        ),
        seeds=SEARCH_SEEDS,
        trials=TRIALS,
        top_fraction=TOP_FRACTION,
        coupling=COUPLING,
        roughness=ROUGHNESS,
        criterion=plan["analysis"]["criterion"],
        panel_sha256=digest(PANEL / "panel.json"),
        source_sha256={p: digest(p) for p in (__file__, "src/mutascape/model.py")},
    )
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(record)), "protocol changed"
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")


def tasks():
    plan = panel()
    accepted = [r for r in plan["candidates"] if "excluded" not in r]
    return [
        (record["landscape"], arm, seed, budgets_for(record))
        for record in accepted
        for arm in ARMS
        for seed in SEARCH_SEEDS
    ]


def fixed_ranking_tau(scores, budget):
    """Kendall tau of each landscape against the mean ranking of the others."""
    table = {}
    for name in scores:
        measured = pd.read_csv(OUT / name / "measured.csv", index_col=[0, 1])
        if budget in measured.index.get_level_values(0):
            table[name] = measured.loc[budget].regret
    frame = pd.DataFrame(table)
    out = {}
    for name in frame:
        others = frame.drop(columns=name).mean(axis=1)
        out[name] = float(kendalltau(frame[name], others).statistic)
    return out


def report():
    records = [
        json.loads(path.read_text()) for path in sorted(OUT.glob("*/*.json")) if path.stem != "."
    ]
    records = [r for r in records if "arm" in r]
    scores = pd.DataFrame(records)
    plan = panel()
    held = set(plan["held_out"])
    space = {r["landscape"]: r["genotype_space"] for r in plan["candidates"]}
    scores["genotype_space"] = scores.landscape.map(space)
    scores["held_out"] = scores.landscape.isin(held)
    lines = []
    for budget in plan["analysis"]["budgets"]:
        column = f"kendall_tau_{budget}"
        subset = scores.loc[scores.held_out & scores[column].notna()]
        if subset.empty:
            continue
        per = subset.groupby(["genotype_space", "landscape", "arm"])[column].mean().unstack("arm")
        fixed = fixed_ranking_tau(sorted(set(subset.landscape)), budget)
        per["fixed_ranking"] = pd.Series(
            {k: fixed[k] for k in per.index.get_level_values("landscape")}
        ).to_numpy()
        summary = paired_bootstrap(per)
        summary.insert(0, "budget", budget)
        lines.append(summary)
        per.round(3).to_csv(OUT / f"per_landscape_{budget}.csv")
    table = pd.concat(lines, ignore_index=True)
    table.round(4).to_csv(OUT / "summary.csv", index=False)
    print(table.round(3).to_string(index=False))


def paired_bootstrap(per, draws=10000, seed=20260923):
    """Resample genotype spaces, since several landscapes share one."""
    spaces = per.index.get_level_values("genotype_space")
    groups = [per.loc[key].mean(axis=0) for key in sorted(set(spaces))]
    stacked = pd.DataFrame(groups)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(stacked), size=(draws, len(stacked)))
    samples = stacked.to_numpy()[picks].mean(axis=1)
    rows = []
    for index, arm in enumerate(stacked.columns):
        row = dict(arm=arm, mean_tau=float(stacked[arm].mean()))
        for control in ("shuffled_singles", "structure_only", "fixed_ranking"):
            if control == arm or control not in stacked.columns:
                continue
            other = list(stacked.columns).index(control)
            difference = samples[:, index] - samples[:, other]
            row[f"vs_{control}"] = float(stacked[arm].mean() - stacked[control].mean())
            row[f"vs_{control}_lo"] = float(np.quantile(difference, 0.025))
            row[f"vs_{control}_hi"] = float(np.quantile(difference, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["protocol", "measure", "evaluate", "report"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    if args.phase == "protocol":
        protocol()
        return
    if args.phase == "report":
        report()
        return
    protocol()
    work = tasks()
    if args.only:
        work = [task for task in work if task[0] == args.only]
    if args.phase == "measure":
        seen = {(name, tuple(budgets)) for name, _, _, budgets in work}
        Parallel(n_jobs=args.jobs)(
            delayed(measured_ranking)(name, list(budgets)) for name, budgets in sorted(seen)
        )
        return
    Parallel(n_jobs=args.jobs)(delayed(evaluate)(*task) for task in work)


if __name__ == "__main__":
    main()
