"""Test whether structure predicts which residue pairs interact.

The pooled prior gives every pair of a given order the same distribution, so it
cannot say which pair matters. If a deposited backbone carries that information,
a model trained on other proteins should predict a held-out protein's normalised
interaction coefficients better than the pooled mean does.

Reported as held-out R2 per protein group. A null here is itself a result, and
it argues that pairwise epistasis does not transfer between proteins on
structure alone.

Example:

    uv run python scripts/run_covariate_prior.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --output notebooks/iclr/output_260821_e2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from mutascape.benchmarks.covariates import (
    FEATURE_COLUMNS,
    build_feature_table,
    structure_index,
)
from mutascape.benchmarks.datasets import load_ssmula_landscapes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def _pdb_path(root: Path, dataset) -> Path | None:
    """SSMuLA keeps one structure per protein directory."""
    directory = Path(dataset.spec.relative_path).parts[1]
    candidate = root / "data" / directory / f"{directory}.pdb"
    return candidate if candidate.exists() else None


def main() -> None:
    args = _parse_args()
    datasets = load_ssmula_landscapes(args.ssmula_root, strict=False)

    structures = {}
    coverage = []
    for dataset in datasets:
        path = _pdb_path(args.ssmula_root, dataset)
        index = structure_index(dataset, path) if path is not None else None
        structures[dataset.name] = index
        coverage.append(
            {
                "dataset": dataset.name,
                "protein": dataset.protein,
                "pdb": str(path.name) if path else None,
                "mapped": index is not None,
                "source_positions": list(dataset.spec.positions or ()),
                "min_cb_distance": (
                    round(min(index.distances.values()), 2) if index and index.distances else None
                ),
                "max_cb_distance": (
                    round(max(index.distances.values()), 2) if index and index.distances else None
                ),
            }
        )

    table = build_feature_table(datasets, structures)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(args.output / "structure_coverage.csv", index=False)
    table.to_csv(args.output / "pair_features.csv", index=False)
    if table.empty:
        raise SystemExit("no measured pairs available")

    features = list(FEATURE_COLUMNS)
    rows = []
    for protein in sorted(table["protein"].unique()):
        train = table[table["protein"] != protein]
        test = table[table["protein"] == protein]
        if train.empty or len(test) < 10:
            continue
        x_train = train[features].to_numpy(dtype=float)
        x_test = test[features].to_numpy(dtype=float)
        y_train = train["beta"].to_numpy(dtype=float)
        y_test = test["beta"].to_numpy(dtype=float)

        model = HistGradientBoostingRegressor(random_state=args.seed, max_iter=300)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)

        baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
        flat = baseline.predict(x_test)

        def r2(pred):
            residual = float(np.sum((y_test - pred) ** 2))
            total = float(np.sum((y_test - np.mean(y_test)) ** 2))
            return 1.0 - residual / total if total > 0 else float("nan")

        # Several libraries mutate adjacent residues, so every pair is in
        # contact and distance cannot discriminate inside that landscape. A
        # per-protein correlation would hide that behind between-landscape
        # differences, so score each landscape on its own as well.
        within = []
        for name, block in test.assign(prediction=prediction).groupby("dataset"):
            if len(block) < 10:
                continue
            value = block["prediction"].corr(block["beta"], method="spearman")
            if np.isfinite(value):
                within.append(float(value))

        rows.append(
            {
                "protein": protein,
                "spearman_within_landscape": float(np.median(within)) if within else float("nan"),
                "n_landscapes": len(within),
                "n_train": len(train),
                "n_test": len(test),
                "r2_covariates": r2(prediction),
                "r2_pooled_mean": r2(flat),
                "spearman": float(pd.Series(prediction).corr(pd.Series(y_test), method="spearman")),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.output / "holdout_r2.csv", index=False)

    summary = {
        "seed": args.seed,
        "n_pairs": int(len(table)),
        "n_proteins": int(table["protein"].nunique()),
        "structures_mapped": int(sum(1 for c in coverage if c["mapped"])),
        "median_r2_covariates": float(results["r2_covariates"].median()),
        "median_r2_pooled_mean": float(results["r2_pooled_mean"].median()),
        "proteins_beating_pooled_mean": int(
            (results["r2_covariates"] > results["r2_pooled_mean"]).sum()
        ),
        "median_spearman_pooled": float(results["spearman"].median()),
        "median_spearman_within_landscape": float(results["spearman_within_landscape"].median()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
