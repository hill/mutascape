"""Compare sampled percentile maps with exact archived calibration landscapes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr, wasserstein_distance

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.calibrated import DEFAULT_REFERENCE_SIZE, QuantileCalibratedLandscape
from scripts.investigate_singles_only import (
    CONFIRMATION,
    DEVELOPMENT,
    SEEDS,
    data,
    digest,
    local_geometry,
)

ARCHIVE = Path("results/manuscript_review_20261006")
FULL_C2ST = Path("results/manuscript_full_c2st_20261006")
PROTEINS = (*DEVELOPMENT, *CONFIRMATION)


def predictions(landscape, frame):
    """Evaluate the same measured genotypes in fixed-size batches."""
    return np.concatenate(
        [
            landscape.evaluate_batch(frame.iloc[start : start + 8192].mutations.tolist())
            for start in range(0, len(frame), 8192)
        ]
    )


def sequence_auc(values, frame, design, rows, groups):
    """Reuse the chosen archived genotype groups and classifier settings."""
    singles, test = frame.loc[frame.depth <= 1], frame.iloc[rows]
    real = completion_fidelity_features(singles, test).to_numpy()
    generated = completion_fidelity_features(singles, test, values[rows]).to_numpy()
    codes = design["codes"][rows]
    indicators = np.column_stack(
        [
            codes[:, site] == residue
            for site, size in enumerate(design["shape"])
            for residue in range(size)
        ]
    )
    return classifier_two_sample_test(
        np.column_stack([real, indicators]),
        np.column_stack([generated, indicators]),
        real_groups=groups,
        synthetic_groups=groups,
        classifier="histogram",
        n_permutations=0,
        seed=2026092300,
    ).auc


def evaluate(protein, seed, sizes, reference_seeds, output, c2st_scope):
    """Compare maps without refitting or changing experimental calibration inputs."""
    frame, design, _ = data(protein)
    folder = ARCHIVE / protein
    model = QuantileCalibratedLandscape.from_dict(json.loads((folder / "model.json").read_text()))
    rows = np.load(folder / "distribution_rows.npy")
    measured = frame.fitness.to_numpy()
    scale = measured[rows].std()
    measured_spectrum, _ = local_geometry(measured, design["cubes"])
    start = time.perf_counter()
    exact = model.draw(seed, reference_size=None)
    exact_seconds = time.perf_counter() - start
    exact_values = predictions(exact, frame)
    archived = np.load(folder / f"evaluation_{seed}.npy")
    np.testing.assert_allclose(exact_values, archived, rtol=0, atol=1e-12)
    exact_spectrum, _ = local_geometry(exact_values, design["cubes"])
    exact_w1 = wasserstein_distance(measured[rows], exact_values[rows]) / scale
    exact_walsh = np.abs(measured_spectrum - exact_spectrum).sum()
    if c2st_scope == "full":
        split = np.load(FULL_C2ST / protein / "folds.npz")
        c2st_rows, groups = split["rows"], split["groups"]
        np.testing.assert_array_equal(c2st_rows, rows)
        archived_auc = json.loads((FULL_C2ST / protein / f"calibration96_{seed}.json").read_text())
        assert archived_auc["fitness_sha256"] == digest(folder / f"evaluation_{seed}.npy")
        assert archived_auc["folds_sha256"] == digest(FULL_C2ST / protein / "folds.npz")
        exact_auc = archived_auc["auc_sequence"]
        if protein == "GB1" and seed == SEEDS[0]:
            replay_auc = sequence_auc(exact_values, frame, design, c2st_rows, groups)
            np.testing.assert_allclose(replay_auc, exact_auc, rtol=0, atol=1e-12)
    else:
        c2st_rows = groups = design["test"]
        exact_auc = sequence_auc(exact_values, frame, design, c2st_rows, groups)
    singles = frame.loc[frame.depth <= 1]
    records = []
    for size in sizes:
        for reference_seed in reference_seeds:
            start = time.perf_counter()
            sampled = model.draw(seed, reference_size=size, reference_seed=reference_seed)
            seconds = time.perf_counter() - start
            values = predictions(sampled, frame)
            np.testing.assert_array_equal(values[singles.index], singles.fitness)
            np.testing.assert_allclose(
                sampled.evaluate_batch(frame.iloc[design["test"][::-1]].mutations.tolist()),
                values[design["test"][::-1]],
                rtol=0,
                atol=1e-12,
            )
            spectrum, _ = local_geometry(values, design["cubes"])
            w1 = wasserstein_distance(measured[rows], values[rows]) / scale
            walsh = np.abs(measured_spectrum - spectrum).sum()
            auc = (
                exact_auc
                if sampled.reference_mode == "exact"
                else sequence_auc(values, frame, design, c2st_rows, groups)
            )
            error = np.abs(values[rows] - exact_values[rows]) / scale
            # Compare midpoint CDFs on the exact score support, without any fitness labels.
            probabilities = np.interp(
                exact.score_knots, sampled.score_knots, sampled.probabilities, left=0, right=1
            )
            records.append(
                dict(
                    protein=protein,
                    seed=seed,
                    reference_seed=reference_seed,
                    requested_reference_size=size,
                    reference_size=sampled.reference_size,
                    reference_mode=sampled.reference_mode,
                    search_space_size=model.reference.domain.count(),
                    evaluation_n=len(rows),
                    calibration_n=len(model.pilot_fitness),
                    c2st_scope=c2st_scope,
                    c2st_n=len(c2st_rows),
                    exact_seconds=exact_seconds,
                    sampled_seconds=seconds,
                    archived_max_error=float(np.max(np.abs(exact_values - archived))),
                    cdf_max_error=float(np.max(np.abs(probabilities - exact.probabilities))),
                    scaled_mae=float(error.mean()),
                    scaled_p99_error=float(np.quantile(error, 0.99)),
                    scaled_max_error=float(error.max()),
                    prediction_spearman=float(
                        spearmanr(exact_values[rows], values[rows]).statistic
                    ),
                    exact_w1=float(exact_w1),
                    sampled_w1=float(w1),
                    w1_change=float(w1 - exact_w1),
                    exact_walsh=float(exact_walsh),
                    sampled_walsh=float(walsh),
                    walsh_change=float(walsh - exact_walsh),
                    exact_auc=float(exact_auc),
                    sampled_auc=float(auc),
                    auc_change=float(auc - exact_auc),
                )
            )
    path = output / f"{protein}_{seed}.json"
    path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print(f"Completed {protein}, landscape seed {seed}", flush=True)
    return records


def summarise(records, output):
    table = pd.DataFrame(records)
    table.to_csv(output / "comparisons.csv", index=False)
    default = table.loc[table.requested_reference_size == DEFAULT_REFERENCE_SIZE]
    rows = []
    for protein, group in default.groupby("protein"):
        row = dict(
            protein=protein,
            mode=group.reference_mode.iloc[0],
            search_space=int(group.search_space_size.iloc[0]),
            max_cdf_error=float(group.cdf_max_error.max()),
            mean_scaled_mae=float(group.scaled_mae.mean()),
            max_absolute_w1_change=float(group.w1_change.abs().max()),
            max_absolute_walsh_change=float(group.walsh_change.abs().max()),
            max_absolute_auc_change=float(group.auc_change.abs().max()),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False)
    # First take the median across landscape seeds, then give proteins equal weight.
    evaluation = default.loc[default.protein.isin(CONFIRMATION)]
    means = evaluation.groupby(["reference_seed", "protein"]).median(numeric_only=True)
    means = means.groupby("reference_seed").mean(numeric_only=True)
    means.to_csv(output / "evaluation_means.csv")
    print(summary.round(5).to_string(index=False))
    print(
        means[
            ["exact_w1", "sampled_w1", "exact_walsh", "sampled_walsh", "exact_auc", "sampled_auc"]
        ]
        .round(6)
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/reference_sampling_20261014"))
    parser.add_argument("--proteins", nargs="+", default=PROTEINS, choices=PROTEINS)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10_000, DEFAULT_REFERENCE_SIZE])
    parser.add_argument("--reference-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--c2st-scope", choices=["sample1000", "full"], default="full")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path("src/mutascape/calibrated.py")]
    for protein in args.proteins:
        sources.extend(ARCHIVE / protein / file for file in ("model.json", "calibration.csv"))
        sources.extend(
            Path("results/singles_only_restart_20260923") / protein / file
            for file in ("frame.pkl", "design.npz")
        )
        sources.extend(ARCHIVE / protein / f"evaluation_{seed}.npy" for seed in SEEDS)
        if args.c2st_scope == "full":
            sources.append(FULL_C2ST / protein / "folds.npz")
            sources.extend(FULL_C2ST / protein / f"calibration96_{seed}.json" for seed in SEEDS)
    protocol = dict(
        proteins=list(args.proteins),
        landscape_seeds=SEEDS,
        reference_seeds=args.reference_seeds,
        sizes=args.sizes,
        default_reference_size=DEFAULT_REFERENCE_SIZE,
        sampling="Uniform with replacement over supported multi-mutants, no fitness labels",
        small_spaces="Exact when the number of multi-mutants is no larger than reference_size",
        experimental_inputs="Frozen 96-genotype calibration sets, unchanged model parameters",
        evaluation="All measured multi-mutants except calibration, fixed Walsh cubes",
        c2st_scope=args.c2st_scope,
        source_sha256={str(path): digest(path) for path in sources},
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        groups = Parallel(n_jobs=args.jobs)(
            delayed(evaluate)(
                protein, seed, args.sizes, args.reference_seeds, args.output, args.c2st_scope
            )
            for protein in args.proteins
            for seed in SEEDS
        )
    summarise([record for group in groups for record in group], args.output)


if __name__ == "__main__":
    main()
