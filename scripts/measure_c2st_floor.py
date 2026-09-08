"""Compare assay proxies, which do not establish a classifier noise floor.

GB1 appears in two collections that reuse the same measurements. eqFP611 and
RhlA contribute different phenotypes. Neither comparison is a replicate test.
Use audit_creilov_c2st.py for the raw CreiLOV replicate comparison.
Independent replicates can score near chance. Their disagreement depends on
the classifier, features, sample size, and assay normalisation.

Example:

    uv run python scripts/measure_c2st_floor.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym \
        --output notebooks/iclr/output_260820_c2st_floor
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.evaluation import (
    classifier_two_sample_test,
    completion_fidelity_features,
    maximum_mean_discrepancy,
)
from mutascape.benchmarks.splits import has_informative_singles, mutation_order_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--c2st-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def _source_positions(dataset) -> dict[int, int]:
    """Map a landscape's own site numbering onto shared residue numbers.

    SSMuLA numbers its mutable sites 1..k inside the mutable-site sequence and
    keeps the residue numbers in the spec. CombinGym uses residue numbers
    directly. Aligning them lets one library measured by two groups be compared.
    The mapping serves the comparison only and never reaches the model.
    """
    spec_positions = dataset.spec.positions
    if spec_positions is None or len(spec_positions) != len(dataset.mutable_positions):
        return {}
    return dict(zip(dataset.mutable_positions, spec_positions))


def _keyed(dataset, *, align: bool = False) -> dict[tuple, float]:
    """Map each variant to its fitness, keyed on the sorted mutation set."""
    frame = dataset.variants
    mapping = _source_positions(dataset) if align else {}
    keyed = {}
    for mutations, fitness in frame[["mutations", "fitness"]].itertuples(index=False, name=None):
        key = tuple(sorted((mapping.get(int(p), int(p)), aa) for p, aa in mutations.items()))
        keyed[key] = float(fitness)
    return keyed


def main() -> None:
    args = _parse_args()
    datasets = []
    if args.ssmula_root is not None:
        datasets.extend(load_ssmula_landscapes(args.ssmula_root, strict=False))
    if args.combingym_root is not None:
        datasets.extend(load_combingym_landscapes(args.combingym_root, strict=False))
    if not datasets:
        raise SystemExit("No landscapes found. Pass at least one dataset root.")

    by_protein: dict[str, list] = {}
    for dataset in datasets:
        by_protein.setdefault(dataset.protein, []).append(dataset)

    rows: list[dict[str, object]] = []
    for protein, group in sorted(by_protein.items()):
        for left, right in combinations(group, 2):
            align = left.spec.source != right.spec.source
            # Two landscapes are a control only when they measure the same
            # library. SSMuLA numbers every three-site TrpB library 1..3, so
            # without this check libraries at different residues would look
            # like repeated measurements of one.
            left_sites = set(_source_positions(left).values()) or set(left.mutable_positions)
            right_sites = set(_source_positions(right).values()) or set(right.mutable_positions)
            if left_sites != right_sites:
                continue

            # Same sites is not enough. ParD2 and ParD3 mutate the same three
            # residues of different antitoxins, so their wild types differ and
            # they are two proteins rather than one measured twice.
            def wt_at(dataset):
                mapping = _source_positions(dataset)
                order = sorted(mapping.values()) if mapping else sorted(dataset.mutable_positions)
                inverse = {v: k for k, v in mapping.items()} if mapping else {}
                return "".join(
                    dataset.wild_type_sequence[inverse.get(site, site) - 1] for site in order
                )

            if wt_at(left) != wt_at(right):
                continue
            left_values, right_values = _keyed(left, align=align), _keyed(right, align=align)
            shared = sorted(set(left_values) & set(right_values))
            # Both arms must describe the same genotypes, or the classifier
            # would separate them on library composition rather than on fitness.
            if len(shared) < 50:
                continue

            split = mutation_order_split(left.variants, max_train_depth=1)
            if not has_informative_singles(split.train):
                continue

            rng = np.random.default_rng(args.seed)
            # The features are defined relative to measured singles, so a
            # variant is only scoreable when every substitution in it was
            # measured on its own.
            mapping = _source_positions(left) if align else {}
            measured_singles = {
                (mapping.get(int(p), int(p)), aa)
                for mutations in split.train.loc[split.train["depth"] == 1, "mutations"]
                for p, aa in mutations.items()
            }
            higher = [
                key
                for key in shared
                if len(key) > 1 and all(item in measured_singles for item in key)
            ]
            if len(higher) > args.limit:
                chosen = rng.choice(len(higher), size=args.limit, replace=False)
                higher = [higher[int(index)] for index in chosen]
            if len(higher) < 50:
                continue

            inverse = {v: k for k, v in mapping.items()}
            variants = pd.DataFrame(
                {
                    "mutations": [{inverse.get(p, p): aa for p, aa in key} for key in higher],
                    "fitness": [left_values[key] for key in higher],
                    "depth": [len(key) for key in higher],
                }
            )
            features_left = completion_fidelity_features(split.train, variants)
            features_right = completion_fidelity_features(
                split.train,
                variants,
                np.asarray([right_values[key] for key in higher], dtype=float),
            )
            groups = np.arange(len(higher))
            result = classifier_two_sample_test(
                features_left.to_numpy(),
                features_right.to_numpy(),
                real_groups=groups,
                synthetic_groups=groups,
                n_permutations=args.c2st_permutations,
                seed=args.seed,
            )
            mmd = maximum_mean_discrepancy(features_left.to_numpy(), features_right.to_numpy())
            rows.append(
                {
                    "protein": protein,
                    "assay_a": left.name,
                    "assay_b": right.name,
                    "n_shared": len(shared),
                    "n_scored": len(higher),
                    "c2st_auc": result.auc,
                    "c2st_pvalue": result.permutation_pvalue,
                    "mmd_squared": mmd["mmd_squared"],
                }
            )
            print(f"  {left.name} vs {right.name}: AUC {result.auc:.3f}")

    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "real_vs_real.csv", index=False)
    summary = {
        "control_kind": "assay proxies, not independent replicate measurements",
        "seed": args.seed,
        "n_pairs": int(len(frame)),
        "median_c2st_auc": float(frame["c2st_auc"].median()) if not frame.empty else None,
        "min_c2st_auc": float(frame["c2st_auc"].min()) if not frame.empty else None,
        "max_c2st_auc": float(frame["c2st_auc"].max()) if not frame.empty else None,
        "median_mmd_squared": float(frame["mmd_squared"].median()) if not frame.empty else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(frame.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
