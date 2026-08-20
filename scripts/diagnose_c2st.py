"""Attribute the classifier's separation to individual features.

A single AUC says a synthetic landscape is distinguishable. It does not say
what gives it away, and the model change depends on that. This scores each
feature on its own, for the synthetic arm and for two real assays of the same
library, so every feature carries its own floor.

A feature already at its real-versus-real floor needs no work. A feature far
above it names the next model change.

Example:

    uv run python scripts/diagnose_c2st.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym \
        --output notebooks/iclr/output_260820_c2st_diagnosis
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.evaluation import completion_fidelity_features, per_feature_c2st
from mutascape.benchmarks.splits import (
    has_informative_singles,
    iter_protein_holdouts,
    mutation_order_split,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--floor-policy", choices=["always", "censored", "never"], default="always")
    return parser.parse_args()


def _supported(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    singles = {
        tuple(mutations.items())[0] for mutations in train.loc[train["depth"] == 1, "mutations"]
    }
    keep = test["mutations"].map(lambda m: all(item in singles for item in m.items()))
    return test.loc[keep].reset_index(drop=True)


def _coefficients(real: pd.DataFrame, synthetic: pd.DataFrame, seed: int) -> dict[str, float]:
    """Standardised logistic weights, so magnitudes compare across features."""
    features = np.vstack([real.to_numpy(float), synthetic.to_numpy(float)])
    labels = np.concatenate([np.zeros(len(real)), np.ones(len(synthetic))])
    scaled = StandardScaler().fit_transform(features)
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(scaled, labels)
    return dict(zip(real.columns, model.coef_[0]))


def _source_positions(dataset) -> dict[int, int]:
    spec_positions = dataset.spec.positions
    if spec_positions is None or len(spec_positions) != len(dataset.mutable_positions):
        return {}
    return dict(zip(dataset.mutable_positions, spec_positions))


def _keyed(dataset, *, align: bool) -> dict[tuple, float]:
    mapping = _source_positions(dataset) if align else {}
    frame = dataset.variants
    keyed = {}
    for mutations, fitness in frame[["mutations", "fitness"]].itertuples(index=False, name=None):
        key = tuple(sorted((mapping.get(int(p), int(p)), aa) for p, aa in mutations.items()))
        keyed[key] = float(fitness)
    return keyed


def _synthetic_rows(datasets, args) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(iter_protein_holdouts(datasets)):
        prior = LandscapePrior.fit(list(fold.train), max_order=args.max_order)
        for dataset_index, dataset in enumerate(fold.test):
            split = mutation_order_split(dataset.variants, max_train_depth=1)
            if not has_informative_singles(split.train):
                continue
            test = _supported(split.test, split.train)
            if len(test) < 50:
                continue
            if len(test) > args.limit:
                test = test.sample(args.limit, random_state=args.seed).reset_index(drop=True)
            posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)
            seed = args.seed + 1000 * fold_index + dataset_index
            generated = posterior.sample(seed).evaluate_batch(test["mutations"].tolist())
            real = completion_fidelity_features(split.train, test)
            synthetic = completion_fidelity_features(split.train, test, generated)
            groups = np.arange(len(test))
            scores = per_feature_c2st(real, synthetic, groups=groups, seed=seed)
            weights = _coefficients(real, synthetic, seed)
            for feature, auc in scores.items():
                rows.append(
                    {
                        "arm": "synthetic",
                        "label": dataset.name,
                        "feature": feature,
                        "auc": auc,
                        "weight": weights[feature],
                    }
                )
            print(f"  synthetic {dataset.name}")
    return rows


def _real_rows(datasets, args) -> list[dict[str, object]]:
    by_protein: dict[str, list] = {}
    for dataset in datasets:
        by_protein.setdefault(dataset.protein, []).append(dataset)

    rows: list[dict[str, object]] = []
    for group in by_protein.values():
        for left, right in combinations(group, 2):
            align = left.spec.source != right.spec.source
            left_sites = set(_source_positions(left).values()) or set(left.mutable_positions)
            right_sites = set(_source_positions(right).values()) or set(right.mutable_positions)
            if left_sites != right_sites:
                continue
            mapping = _source_positions(left) if align else {}
            inverse = {v: k for k, v in mapping.items()}

            def wt_at(dataset):
                m = _source_positions(dataset)
                order = sorted(m.values()) if m else sorted(dataset.mutable_positions)
                inv = {v: k for k, v in m.items()} if m else {}
                return "".join(dataset.wild_type_sequence[inv.get(s, s) - 1] for s in order)

            if wt_at(left) != wt_at(right):
                continue

            left_values, right_values = _keyed(left, align=align), _keyed(right, align=align)
            shared = sorted(set(left_values) & set(right_values))
            split = mutation_order_split(left.variants, max_train_depth=1)
            if not has_informative_singles(split.train):
                continue
            measured = {
                (mapping.get(int(p), int(p)), aa)
                for mutations in split.train.loc[split.train["depth"] == 1, "mutations"]
                for p, aa in mutations.items()
            }
            higher = [k for k in shared if len(k) > 1 and all(i in measured for i in k)]
            if len(higher) < 50:
                continue
            rng = np.random.default_rng(args.seed)
            if len(higher) > args.limit:
                pick = rng.choice(len(higher), size=args.limit, replace=False)
                higher = [higher[int(i)] for i in pick]

            variants = pd.DataFrame(
                {
                    "mutations": [{inverse.get(p, p): aa for p, aa in k} for k in higher],
                    "fitness": [left_values[k] for k in higher],
                    "depth": [len(k) for k in higher],
                }
            )
            first = completion_fidelity_features(split.train, variants)
            second = completion_fidelity_features(
                split.train,
                variants,
                np.asarray([right_values[k] for k in higher], dtype=float),
            )
            groups = np.arange(len(higher))
            scores = per_feature_c2st(first, second, groups=groups, seed=args.seed)
            weights = _coefficients(first, second, args.seed)
            label = f"{left.name} vs {right.name}"
            for feature, auc in scores.items():
                rows.append(
                    {
                        "arm": "real",
                        "label": label,
                        "feature": feature,
                        "auc": auc,
                        "weight": weights[feature],
                    }
                )
            print(f"  real {label}")
    return rows


def main() -> None:
    args = _parse_args()
    datasets = []
    if args.ssmula_root is not None:
        datasets.extend(load_ssmula_landscapes(args.ssmula_root, strict=False))
    if args.combingym_root is not None:
        datasets.extend(load_combingym_landscapes(args.combingym_root, strict=False))
    if not datasets:
        raise SystemExit("No landscapes found. Pass at least one dataset root.")

    rows = _synthetic_rows(datasets, args) + _real_rows(datasets, args)
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "per_feature.csv", index=False)

    table = frame.pivot_table(index="feature", columns="arm", values="auc", aggfunc="median")
    table["gap"] = table["synthetic"] - table["real"]
    table = table.sort_values("gap", ascending=False)
    weights = (
        frame[frame.arm == "synthetic"]
        .assign(absw=lambda d: d.weight.abs())
        .groupby("feature")["absw"]
        .median()
    )
    table["abs_weight"] = weights
    table.to_csv(args.output / "feature_gap.csv")

    summary = {
        "seed": args.seed,
        "max_order": args.max_order,
        "worst_feature": str(table.index[0]),
        "worst_gap": float(table["gap"].iloc[0]),
        "median_synthetic_auc": float(table["synthetic"].median()),
        "median_real_auc": float(table["real"].median()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== per-feature AUC, synthetic against the real-versus-real floor ===")
    print(table.round(3).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
