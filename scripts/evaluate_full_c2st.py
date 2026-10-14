"""Evaluate both manuscript generators on every eligible experimental multi-mutant."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from scripts.investigate_singles_only import (
    CONFIRMATION,
    DEVELOPMENT,
    SEEDS,
    data,
    digest,
)
from scripts.investigate_singles_only import OUT as SINGLE_SITE
from scripts.recompute_review_distributions import OUT as CALIBRATED

OUT = Path("results/manuscript_full_c2st_20261006")
PROTEINS = (*DEVELOPMENT, *CONFIRMATION)
MODELS = ("sigmoid", "calibration96")
CLASSIFIER_SEED = 2026092300
FOLD_SEED = 2026100601


def paired_folds(rows, seed):
    """Randomise unique genotype groups and keep paired examples in the same fold."""
    groups = np.random.default_rng(seed).permutation(len(rows))
    paired = np.tile(groups, 2)
    labels = np.repeat([0, 1], len(rows))
    folds = np.full(len(paired), -1, dtype=np.int8)
    for fold, (_, test) in enumerate(GroupKFold(5).split(paired[:, None], labels, paired)):
        folds[test] = fold
    assert np.array_equal(folds[: len(rows)], folds[len(rows) :])
    assert set(folds) == set(range(5))
    return groups, folds[: len(rows)]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        proteins=PROTEINS,
        models=MODELS,
        landscape_seeds=SEEDS,
        classifier="HistGradientBoostingClassifier",
        classifier_parameters=dict(
            max_iter=100,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=CLASSIFIER_SEED,
        ),
        folds=5,
        fold_seed=FOLD_SEED,
        group_assignment="Seeded random unique genotype codes before paired GroupKFold",
        sample="All experimental multi-mutants in the supported search space except calibration",
        calibration_genotypes=96,
        calibration_sets_per_protein=1,
        features=["base", "sequence"],
        permutations=0,
        model_fitting="Reuse existing fitted generators and generated fitness arrays",
        source_sha256={
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("src/mutascape/benchmarks/evaluation.py"),
                CALIBRATED / "protocol.json",
            ]
        },
    )
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        path.write_text(json.dumps(protocol, indent=2) + "\n")
    for index, name in enumerate(PROTEINS):
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        rows = np.load(CALIBRATED / name / "distribution_rows.npy")
        frame, _, _ = data(name)
        calibration = pd.read_csv(CALIBRATED / name / "calibration.csv").row.to_numpy()
        assert len(np.unique(rows)) == len(rows)
        assert (frame.iloc[rows].depth > 1).all()
        assert not np.intersect1d(rows, calibration).size
        assert len(rows) + 96 == int((frame.depth > 1).sum())
        groups, folds = paired_folds(rows, FOLD_SEED + index)
        np.savez(folder / "folds.npz", rows=rows, groups=groups, folds=folds)
        print(f"Prepared {name}, {len(rows):,} genotypes, five paired folds", flush=True)


def evaluate(name, model, seed):
    folder = OUT / name
    target = folder / f"{model}_{seed}.json"
    if target.exists():
        return
    frame, design, _ = data(name)
    split = np.load(folder / "folds.npz")
    rows, groups = split["rows"], split["groups"]
    values_path = (
        SINGLE_SITE / name / f"fitness_sigmoid_{seed}.npy"
        if model == "sigmoid"
        else CALIBRATED / name / f"evaluation_{seed}.npy"
    )
    values = np.load(values_path)
    singles, evaluation = frame.loc[frame.depth <= 1], frame.iloc[rows]
    real = completion_fidelity_features(singles, evaluation).to_numpy()
    synthetic = completion_fidelity_features(singles, evaluation, values[rows]).to_numpy()
    codes = design["codes"][rows]
    indicators = np.column_stack(
        [
            codes[:, site] == residue
            for site, size in enumerate(design["shape"])
            for residue in range(size)
        ]
    )
    result = dict(
        protein=name,
        model=model,
        seed=seed,
        genotypes=len(rows),
        observations=2 * len(rows),
        folds_sha256=digest(folder / "folds.npz"),
        fitness_sha256=digest(values_path),
    )
    for feature_set in ("base", "sequence"):
        start = perf_counter()
        x, y = (
            (real, synthetic)
            if feature_set == "base"
            else (np.column_stack([real, indicators]), np.column_stack([synthetic, indicators]))
        )
        score = classifier_two_sample_test(
            x,
            y,
            real_groups=groups,
            synthetic_groups=groups,
            classifier="histogram",
            n_permutations=0,
            seed=CLASSIFIER_SEED,
        )
        details = asdict(score)
        details["permutation_pvalue"] = None
        result[feature_set] = details
        result[f"auc_{feature_set}"] = score.auc
        result[f"seconds_{feature_set}"] = perf_counter() - start
        print(
            f"{name} {model} {seed} {feature_set}: "
            f"AUC {score.auc:.4f}, {perf_counter() - start:.1f}s",
            flush=True,
        )
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def summarise():
    results = [
        json.loads((OUT / name / f"{model}_{seed}.json").read_text())
        for name in PROTEINS
        for model in MODELS
        for seed in SEEDS
    ]
    table = pd.DataFrame(results)
    medians = table.groupby(["protein", "model"]).median(numeric_only=True).reset_index()
    study = json.loads((CALIBRATED / "singles-full-distributions.json").read_text())
    calibration = json.loads((CALIBRATED / "calibration96-full-distributions.json").read_text())
    records = {(r["protein"], r["model"], r["seed"]): r for r in results}
    for payload, key, model in [
        (study, "fidelity", "sigmoid"),
        (calibration, "metrics", "calibration96"),
    ]:
        for row in payload[key]:
            if model == "sigmoid" and row["model"] != model:
                continue
            record = records[row["protein"], model, int(row["seed"])]
            for feature in ("base", "sequence"):
                metric = f"auc_{feature}"
                row[f"{metric}_sample1000"] = row[metric]
                row[metric] = record[metric]
            row["c2st_n"] = record["genotypes"]
        group = ["protein", "model"] if key == "fidelity" else ["protein"]
        summary_key = "fidelity_medians" if key == "fidelity" else "medians"
        payload[summary_key] = json.loads(
            pd.DataFrame(payload[key])
            .groupby(group)
            .median(numeric_only=True)
            .reset_index()
            .to_json(orient="records")
        )
        payload["c2st_update"] = json.loads((OUT / "protocol.json").read_text())
    (OUT / "singles-full-evaluation.json").write_text(
        json.dumps(study, indent=2, allow_nan=False) + "\n"
    )
    (OUT / "calibration96-full-evaluation.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False) + "\n"
    )
    medians.to_csv(OUT / "summary.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(
        medians[["protein", "model", "genotypes", "auc_base", "auc_sequence"]]
        .round(4)
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "summarise"])
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--proteins", nargs="+", choices=PROTEINS, default=PROTEINS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "run":
        Parallel(n_jobs=args.jobs)(
            delayed(evaluate)(name, model, seed)
            for name in args.proteins
            for model in MODELS
            for seed in SEEDS
        )
    else:
        summarise()


if __name__ == "__main__":
    main()
