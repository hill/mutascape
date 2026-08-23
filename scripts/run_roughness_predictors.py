"""Can wild type and singles alone say how rough a landscape will be?

Measured non-additive variance share runs from 0.168 to 0.973 and the measured
order-three to order-two ratio from 0.116 to 1.052. The pooled prior emits one
roughness whatever the target, which is why generated shares span only 0.723 to
0.910 and correlate with the measured ones at 0.374.

A pooled constant cannot fix that. Either the target's own singles carry a
signal for roughness or no held-out generator can match a landscape it has not
seen above first order.

Example:

    uv run python scripts/run_roughness_predictors.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.linear_model import RidgeCV

from mutascape.benchmarks import (
    load_combingym_landscapes,
    load_ssmula_landscapes,
    mutation_order_split,
)
from mutascape.benchmarks.splits import has_informative_singles
from mutascape.benchmarks.topology import measured_tensor, summarise, supported_cube

FEATURES = [
    "n_sites",
    "mean_alphabet",
    "single_std",
    "single_skew",
    "single_kurtosis",
    "deleterious_fraction",
    "beneficial_fraction",
    "floor_fraction",
    "worst_single",
    "best_single",
]
TARGETS = ["walsh_nonadditive", "order_ratio", "log_optima"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("notebooks/iclr/output_roughness"))
    parser.add_argument("--max-cube", type=int, default=200_000)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    return parser.parse_args()


def _single_features(train: pd.DataFrame, cube) -> dict[str, float]:
    """Everything a generator is allowed to see about the target."""
    wt = float(train.loc[train["depth"] == 0, "fitness"].iloc[0])
    effects = train.loc[train["depth"] == 1, "fitness"].to_numpy(dtype=float) - wt
    if effects.size < 4:
        return {}
    worst = float(np.min(effects))
    span = float(np.max(effects) - worst) or 1.0
    return {
        "n_sites": float(len(cube.positions)),
        "mean_alphabet": float(np.mean([len(cube.alphabets[p]) for p in cube.positions])),
        "single_std": float(np.std(effects)),
        "single_skew": float(skew(effects)),
        "single_kurtosis": float(kurtosis(effects)),
        "deleterious_fraction": float(np.mean(effects < 0)),
        "beneficial_fraction": float(np.mean(effects > 0)),
        # Singles piled at the bottom of their own range is the visible trace
        # of a censored assay, and censoring is what flattens a landscape.
        "floor_fraction": float(np.mean(effects < worst + 0.05 * span)),
        "worst_single": worst,
        "best_single": float(np.max(effects)),
    }


def _leave_one_protein_out(frame: pd.DataFrame, target: str) -> dict[str, float]:
    """Predict one protein's roughness from every other protein's."""
    design = frame[FEATURES].to_numpy(dtype=float)
    values = frame[target].to_numpy(dtype=float)
    keep = np.isfinite(values) & np.isfinite(design).all(axis=1)
    design, values, proteins = design[keep], values[keep], frame["protein"].to_numpy()[keep]
    if len(values) < 6:
        return {"n": float(len(values))}

    predicted = np.empty_like(values)
    for protein in np.unique(proteins):
        test = proteins == protein
        model = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(design[~test], values[~test])
        predicted[test] = model.predict(design[test])

    pooled = np.full_like(values, float(np.mean(values)))
    residual = float(np.sum((values - predicted) ** 2))
    total = float(np.sum((values - pooled) ** 2))
    return {
        "n": float(len(values)),
        "spearman": float(spearmanr(values, predicted).statistic),
        "r2": float(1.0 - residual / total) if total > 0 else float("nan"),
        "measured_min": float(np.min(values)),
        "measured_max": float(np.max(values)),
    }


def main() -> None:
    args = _parse_args()
    datasets = load_ssmula_landscapes(args.ssmula_root, strict=False)
    if args.combingym_root is not None:
        datasets = datasets + load_combingym_landscapes(args.combingym_root, strict=False)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        split = mutation_order_split(dataset.variants, max_train_depth=1)
        if not has_informative_singles(split.train):
            continue
        cube = supported_cube(dataset, split.train)
        if cube.size > args.max_cube or cube.size < 64:
            continue
        features = _single_features(split.train, cube)
        if not features:
            continue
        measured, coverage = measured_tensor(dataset, cube)
        if coverage < args.min_coverage or not np.isfinite(measured).any():
            continue
        shape = summarise(measured)
        order_two = shape.get("walsh_order_2", 0.0)
        rows.append(
            {
                "dataset": dataset.name,
                "protein": dataset.protein,
                "coverage": coverage,
                **features,
                "walsh_nonadditive": shape["walsh_nonadditive"],
                "order_ratio": (
                    shape.get("walsh_order_3", 0.0) / order_two if order_two > 0 else np.nan
                ),
                "log_optima": float(np.log1p(shape["n_optima"])),
                "n_optima": shape["n_optima"],
            }
        )
        print(f"  {dataset.name}: cube {cube.size}, coverage {coverage:.3f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "roughness.csv", index=False)

    scores = {target: _leave_one_protein_out(frame, target) for target in TARGETS}
    correlations = {
        target: {
            feature: float(spearmanr(frame[feature], frame[target], nan_policy="omit").statistic)
            for feature in FEATURES
        }
        for target in TARGETS
    }
    summary = {
        "n_landscapes": int(len(frame)),
        "n_proteins": int(frame["protein"].nunique()),
        "held_out": scores,
        "single_feature_spearman": correlations,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
