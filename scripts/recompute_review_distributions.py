"""Refit with 96 calibration genotypes and evaluate full experimental distributions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.calibrated import QuantileCalibratedLandscape
from scripts.investigate_singles_only import (
    CONFIRMATION,
    DEVELOPMENT,
    SEEDS,
    data,
    diagnostics,
    digest,
)
from scripts.investigate_singles_only import (
    OUT as ORIGINAL,
)

OUT = Path("results/manuscript_review_20261006")
PREVIOUS = Path("results/singles_pilot_extension_20260923")
FROZEN = Path("src/mutascape/data/validation/singles-only-research-v1.json")
PROTEINS = (*DEVELOPMENT, *CONFIRMATION)


def evaluation_rows(depth, calibration):
    """Use every experimental multi-mutant except those supplied for calibration."""
    return np.setdiff1d(np.flatnonzero(np.asarray(depth) > 1), calibration)


def distribution_metrics(real, generated, rows):
    """Compare distributions on identical genotypes and the experimental fitness scale."""
    experimental, synthetic = np.asarray(real)[rows], np.asarray(generated)[rows]
    if not np.isfinite(experimental).all() or not np.isfinite(synthetic).all():
        raise ValueError("Distribution evaluation contains missing fitness values")
    scale = float(np.std(experimental))
    if scale <= 0:
        raise ValueError("Experimental fitness must vary")
    raw = float(wasserstein_distance(experimental, synthetic))
    return dict(w1=raw / scale, w1_raw=raw, experimental_std=scale, distribution_n=len(rows))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        calibration_size=96,
        calibration_selection="First 96 rows of the original uniform 192-genotype sample",
        calibration_replicates=1,
        calibration_exclusions="WT, single-site variants, C2ST genotypes and Walsh corners",
        distribution_evaluation="All experimental multi-mutants except the 96 calibration rows",
        c2st_evaluation="Original fixed sample of 1000 genotypes, grouped five-fold validation",
        spectral_evaluation="Original 256 complete three-site subcubes",
        seeds=SEEDS,
        proteins=PROTEINS,
        model_selection="Keep the previously selected generators and hyperparameters fixed",
        original_study_sha256=digest(FROZEN),
        source_sha256={
            str(p): digest(p) for p in (Path(__file__), Path("src/mutascape/calibrated.py"))
        },
    )
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        path.write_text(json.dumps(protocol, indent=2) + "\n")
    for name in PROTEINS:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        frame, design, meta = data(name)
        calibration = pd.read_csv(PREVIOUS / name / "pilot.csv").iloc[:96].copy()
        rows = calibration.row.to_numpy(dtype=int)
        assert len(np.unique(rows)) == 96
        assert (frame.iloc[rows].depth > 1).all()
        excluded = np.unique(np.r_[design["test"], design["cubes"].ravel()])
        assert not np.intersect1d(rows, excluded).size
        evaluation = evaluation_rows(frame.depth, rows)
        assert np.isin(design["test"], evaluation).all()
        calibration.to_csv(folder / "calibration.csv", index=False)
        np.save(folder / "distribution_rows.npy", evaluation)
        model = QuantileCalibratedLandscape.fit(
            meta["sequence"], frame.loc[frame.depth <= 1], frame.iloc[rows]
        )
        (folder / "model.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
        print(
            f"{name}: 96 calibration genotypes, {len(evaluation):,} distribution genotypes",
            flush=True,
        )


def evaluate(name, seed):
    folder = OUT / name
    target = folder / f"metrics_{seed}.json"
    if target.exists():
        return
    frame, design, _ = data(name)
    model = QuantileCalibratedLandscape.from_dict(json.loads((folder / "model.json").read_text()))
    landscape = model.draw(seed, reference_size=None)
    values = np.concatenate(
        [
            landscape.evaluate_batch(frame.iloc[start : start + 8192].mutations.tolist())
            for start in range(0, len(frame), 8192)
        ]
    )
    singles = frame.loc[frame.depth <= 1]
    np.testing.assert_array_equal(values[singles.index], singles.fitness)
    np.save(folder / f"evaluation_{seed}.npy", values)
    rows = np.load(folder / "distribution_rows.npy")
    result = dict(
        protein=name, seed=seed, calibration_size=96, **diagnostics(name, values, frame, design)
    )
    result["w1_sample1000"] = result.pop("w1")
    result.update(distribution_metrics(frame.fitness.to_numpy(), values, rows))
    test = frame.iloc[design["test"]]
    real = completion_fidelity_features(singles, test).to_numpy()
    synthetic = completion_fidelity_features(singles, test, values[design["test"]]).to_numpy()
    codes = design["codes"][design["test"]]
    indicators = np.column_stack(
        [
            codes[:, site] == residue
            for site, size in enumerate(design["shape"])
            for residue in range(size)
        ]
    )
    for suffix, x, y in [
        ("base", real, synthetic),
        ("sequence", np.column_stack([real, indicators]), np.column_stack([synthetic, indicators])),
    ]:
        result[f"auc_{suffix}"] = classifier_two_sample_test(
            x,
            y,
            real_groups=design["test"],
            synthetic_groups=design["test"],
            classifier="histogram",
            n_permutations=0,
            seed=2026092300,
        ).auc
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Completed {name}, seed {seed}", flush=True)


def summarise():
    study = json.loads(FROZEN.read_text())
    for record in study["fidelity"]:
        name, label, seed = record["protein"], record["model"], int(record["seed"])
        frame, _, _ = data(name)
        values = np.load(ORIGINAL / name / f"fitness_{label}_{seed}.npy")
        rows = np.load(OUT / name / "distribution_rows.npy")
        record["w1_sample1000"] = record.pop("w1")
        record.update(distribution_metrics(frame.fitness.to_numpy(), values, rows))
    table = pd.DataFrame(study["fidelity"])
    study["fidelity_medians"] = json.loads(
        table.groupby(["protein", "model"])
        .median(numeric_only=True)
        .reset_index()
        .to_json(orient="records")
    )
    study["distribution_update"] = json.loads((OUT / "protocol.json").read_text())
    (OUT / "singles-full-distributions.json").write_text(
        json.dumps(study, indent=2, allow_nan=False) + "\n"
    )
    rows = [
        json.loads((OUT / name / f"metrics_{seed}.json").read_text())
        for name in PROTEINS
        for seed in SEEDS
    ]
    medians = pd.DataFrame(rows).groupby("protein").median(numeric_only=True).reset_index()
    payload = dict(
        protocol=json.loads((OUT / "protocol.json").read_text()),
        metrics=rows,
        medians=json.loads(medians.to_json(orient="records")),
    )
    (OUT / "calibration96-full-distributions.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    medians.to_csv(OUT / "calibration96_summary.csv", index=False)
    table.to_csv(OUT / "singles_full_metrics.csv", index=False)
    print(
        medians[
            ["protein", "distribution_n", "w1", "w1_sample1000", "auc_sequence", "local_walsh_l1"]
        ]
        .round(4)
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "summarise"])
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "run":
        Parallel(n_jobs=args.jobs)(delayed(evaluate)(n, s) for n in PROTEINS for s in SEEDS)
    else:
        summarise()


if __name__ == "__main__":
    main()
