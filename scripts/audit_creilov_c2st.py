"""Compare CreiLOV replicates and synthetic draws with a paired classifier test."""

import argparse
import hashlib
import json
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from run_creilov_combinatorial_validation import parse_mutant
from run_creilov_shape_validation import _full_dms_observations
from sklearn.model_selection import GroupKFold

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.benchmarks.splits import mutation_order_split
from mutascape.completion import LandscapePrior, PosteriorLandscape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(".mutascape/benchmark-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--permutations", type=int, default=99)
    args = parser.parse_args()
    raw_path = args.data_root / "combingym/Data/DMS/Raw/CreiLOV_raw.xlsx"
    raw = pd.read_excel(raw_path).rename(columns={"Unnamed: 0": "variant"})
    raw["mutations"] = [{p: aa for p, _, aa in parse_mutant(label)} for label in raw.variant]
    raw["key"] = [tuple(sorted(m.items())) for m in raw.mutations]
    if raw.key.duplicated().any():
        raise ValueError("raw replicate genotypes must be unique")
    raw["depth"] = raw.mutations.map(len)
    for index in (1, 2, 3):
        if not np.allclose(
            np.log10(raw[f"Rep{index}"]), raw[f"Rep{index}_log"], equal_nan=True, atol=1e-10
        ):
            raise ValueError("replicate log columns do not match log10 fluorescence")
    raw = raw.set_index("key", drop=False)
    combingym = load_combingym_landscapes(args.data_root / "combingym", strict=False)
    target = next(d for d in combingym if d.protein == "CreiLOV")
    training = [
        d
        for d in [*load_ssmula_landscapes(args.data_root / "ssmula", strict=False), *combingym]
        if d.protein != "CreiLOV"
    ]
    prior = LandscapePrior.fit(training, max_order=3)
    split = mutation_order_split(target.variants, max_train_depth=1)
    observations = _full_dms_observations(Path("data/creilov/creilov.csv"), target, split.train)
    generator = PosteriorLandscape.condition(target.wild_type_sequence, observations, prior)
    supported = set(generator.single_effects)
    raw = raw.loc[[all(item in supported for item in key) for key in raw.key]].copy()
    columns = ["Rep1_log", "Rep2_log", "Rep3_log", "mean_log"]
    wt = raw.loc[raw.depth == 0, columns].iloc[0]
    eligible = raw.loc[(raw.depth > 1) & np.isfinite(raw[columns]).all(axis=1)].copy()
    # Check source alignment before treating the replicate workbook as the target assay.
    target_values = {
        tuple(sorted(m.items())): f
        for m, f in target.variants[["mutations", "fitness"]].itertuples(index=False, name=None)
    }
    shared = [key for key in eligible.key if key in target_values]
    raw_means = dict(zip(raw.key, raw.mean_log))
    difference = np.asarray([raw_means[key] - target_values[key] for key in shared])
    if not np.allclose(difference, 0, atol=1e-10):
        raise ValueError("raw mean_log does not match the prepared target fitness")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        chosen = np.sort(
            np.random.default_rng(seed).choice(
                len(eligible), min(args.samples, len(eligible)), replace=False
            )
        )
        sample = eligible.iloc[chosen].copy().reset_index(drop=True)
        variants = sample[["mutations", "depth"]].copy()
        variants["fitness"] = sample.mean_log.to_numpy() - float(wt.mean_log)
        arms = {f"rep{i}": sample[f"Rep{i}_log"].to_numpy() - wt[f"Rep{i}_log"] for i in (1, 2, 3)}
        arms["measured"] = variants.fitness.to_numpy()
        for draw in range(3):
            arms[f"synthetic_{draw}"] = generator.draw(20260826 + draw).evaluate_batch(
                variants.mutations.tolist()
            )
        arms["no_pairs"] = generator.sample_dms_pairwise(20260826, coupling=0).evaluate_batch(
            variants.mutations.tolist()
        )
        arms["identical"] = arms["measured"].copy()
        groups = np.arange(len(sample))
        pairs = [
            *combinations(["rep1", "rep2", "rep3"], 2),
            *[
                ("measured", arm)
                for arm in ("identical", "no_pairs", "synthetic_0", "synthetic_1", "synthetic_2")
            ],
        ]
        saved = pd.DataFrame({"variant": sample.variant, "depth": sample.depth, **arms})
        features = {
            arm: completion_fidelity_features(observations, variants, values)
            for arm, values in arms.items()
        }
        np.savez_compressed(
            args.output / f"features_{seed}.npz",
            groups=groups,
            **{arm: frame.to_numpy() for arm, frame in features.items()},
        )
        fold_ids = np.empty(len(groups), dtype=int)
        for fold, (_, test) in enumerate(GroupKFold(5).split(groups, groups, groups)):
            fold_ids[test] = fold
        saved["fold"] = fold_ids
        saved.to_csv(args.output / f"samples_{seed}.csv", index=False)
        for left, right in pairs:
            for classifier in ("logistic", "histogram"):
                result = classifier_two_sample_test(
                    features[left].to_numpy(),
                    features[right].to_numpy(),
                    real_groups=groups,
                    synthetic_groups=groups,
                    classifier=classifier,
                    seed=seed,
                    n_permutations=args.permutations if seed == args.seeds[0] else 0,
                )
                row = {
                    "seed": seed,
                    "left": left,
                    "right": right,
                    "classifier": classifier,
                    "n_pairs": len(sample),
                    "auc": result.auc,
                    "balanced_accuracy": result.accuracy,
                    "pvalue": result.permutation_pvalue,
                    "fold_aucs": json.dumps(result.fold_aucs),
                    "fold_accuracies": json.dumps(result.fold_accuracies),
                }
                rows.append(row)
                print(
                    f"{seed} {left}/{right} {classifier} AUC={result.auc:.3f} "
                    f"accuracy={result.accuracy:.3f}",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    metadata = {
        "protocol": "paired-c2st-v2",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_path": str(raw_path),
        "samples": args.samples,
        "seeds": args.seeds,
        "permutations_first_seed": args.permutations,
        "oracle_draw_seeds": [20260826, 20260827, 20260828],
        "coupling": 0.1,
        "software": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "folds": 5,
        "features": list(next(iter(features.values())).columns),
        "normalisation": "log10 fluorescence minus each assay's measured WT log10 fluorescence",
        "feature_scale": "shared measured DMS single-effect standard deviation",
        "prior": prior.to_dict(),
        "observations": len(observations),
        "eligible_multi_mutants": len(eligible),
        "verified_shared_variants": len(shared),
        "claim": "Replicate disagreement is a reference, not a universal lower bound.",
        "code_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "scripts/audit_creilov_c2st.py",
                "src/mutascape/benchmarks/evaluation.py",
                "src/mutascape/completion.py",
            )
        },
        "dms_sha256": hashlib.sha256(Path("data/creilov/creilov.csv").read_bytes()).hexdigest(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
