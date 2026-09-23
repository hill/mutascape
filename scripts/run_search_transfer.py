"""Does a method's regret on a synthetic landscape predict its regret on the real one?

A benchmark generator exists to rank methods. This scores twelve search methods
on every complete measured cube, then on synthetic cubes built from wild type
and singles alone, and asks whether the two rankings agree.

Example:

    uv run python scripts/run_search_transfer.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym \
        --target-source combingym
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, pearsonr, spearmanr

from mutascape.benchmarks import (
    iter_protein_holdouts,
    load_combingym_landscapes,
    load_graphfla_landscapes,
    load_ssmula_landscapes,
    mutation_order_split,
)
from mutascape.benchmarks.generators import (
    build_generators,
    build_primary_generators,
    dms_pairwise_readout_tensor,
)
from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features, run_method
from mutascape.benchmarks.splits import has_informative_singles
from mutascape.benchmarks.topology import (
    additive_tensor,
    measured_tensor,
    posterior_tensors,
    supported_cube,
)
from mutascape.benchmarks.validity import (
    measured_rank_reliability,
    pairwise_winner_transfer,
    reliability_adjusted_transfer,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape

TARGET_SOURCES = ("ssmula", "combingym")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument(
        "--target-source",
        choices=[*TARGET_SOURCES, "all"],
        default="ssmula",
        help="Score SSMuLA, CombinGym, or both through protein-group holdout",
    )
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.5,
        help="Minimum finite coverage of the single-supported mutation domain",
    )
    parser.add_argument(
        "--minimum-candidates",
        type=int,
        default=384,
        help="Minimum finite candidates, fixed across every query budget",
    )
    parser.add_argument(
        "--graphfla-root",
        type=Path,
        help="Score GraphFLA with its target protein excluded from prior fitting",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("notebooks/iclr/output_search_transfer")
    )
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to named landscapes")
    parser.add_argument("--methods", nargs="*", default=None, help="Restrict to named methods")
    parser.add_argument(
        "--floor-policy",
        choices=["always", "censored", "never"],
        default="censored",
        help="Clip at the worst single always, only on censoring evidence, or never",
    )
    parser.add_argument(
        "--no-global-epistasis",
        dest="global_epistasis",
        action="store_false",
        help="Sum raw measured singles instead of composing through a fitted readout",
    )
    parser.add_argument(
        "--generators",
        nargs="?",
        const="full",
        choices=["primary", "full"],
        help="Add matched main comparators, or the full exploratory generator set",
    )
    parser.add_argument(
        "--dms-pairwise-coupling",
        type=float,
        default=0.1,
        help="Interaction scale for the DMS-pairwise generator",
    )
    return parser.parse_args()


def _regret(values: np.ndarray, best: float) -> float:
    """Simple regret normalised by the landscape's own range."""
    top = float(np.max(values))
    bottom = float(np.min(values))
    if not np.isfinite(top - bottom) or top <= bottom:
        return float("nan")
    return (top - best) / (top - bottom)


def _include_posterior_baseline(generators: str | None) -> bool:
    """Keep the older empirical-prior arm outside the primary comparison."""
    return generators != "primary"


def _validate_scores(scores: pd.DataFrame, *, expected_trials: int) -> None:
    """Reject incomplete or invalid trial tables before aggregation."""
    key = ["dataset", "arm", "method", "trial"]
    if bool(scores.duplicated(key).any()):
        raise ValueError("score table has duplicate trial rows")
    numeric = scores[["best_fitness", "regret"]].to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()):
        raise ValueError("score table has non-finite fitness or regret")
    if bool(((scores["regret"] < 0) | (scores["regret"] > 1)).any()):
        raise ValueError("score table has regret outside zero to one")
    trial_counts = scores.groupby(["dataset", "arm", "method"])["trial"].nunique()
    if not bool((trial_counts == expected_trials).all()):
        raise ValueError("score table has an incomplete trial group")


def _score_landscape(dataset, prior, args, fold_index, dataset_index) -> list[dict[str, object]]:
    """Run every method on every arm of one landscape."""
    split = mutation_order_split(dataset.variants, max_train_depth=1)
    if not has_informative_singles(split.train):
        return []
    cube = supported_cube(dataset, split.train)
    measured, coverage = measured_tensor(dataset, cube)
    mask = np.isfinite(measured)
    if int(mask.sum()) < 4 or coverage < args.minimum_coverage:
        return []

    space = SearchSpace.from_cube(cube, mask)
    features = make_features(space)
    seed = args.seed + 1000 * fold_index + 10 * dataset_index
    posterior = PosteriorLandscape.condition(
        dataset.wild_type_sequence,
        split.train,
        prior,
        floor_policy=args.floor_policy,
        global_epistasis=args.global_epistasis,
    )

    arms: dict[str, np.ndarray] = {
        "measured": space.arm_values(measured),
        "additive": space.arm_values(additive_tensor(split.train, cube)),
    }
    if _include_posterior_baseline(args.generators):
        for draw, tensor in enumerate(
            posterior_tensors(posterior, cube, n_samples=args.draws, seed=seed)
        ):
            arms[f"posterior_{draw}"] = space.arm_values(tensor)
    if args.generators:
        if args.generators == "primary":
            generators = build_primary_generators(
                cube,
                split.train,
                seed=seed,
                posterior=posterior,
                coupling=args.dms_pairwise_coupling,
                draws=args.draws,
            )
        else:
            generators = build_generators(
                cube,
                split.train,
                seed=seed,
                posterior=posterior,
                dms_pairwise_coupling=args.dms_pairwise_coupling,
                dms_pairwise_draws=args.draws,
            )
        for name, tensor in generators.items():
            arms[name] = space.arm_values(tensor)

    if getattr(args, "readout_ablation", False):
        for draw in range(args.draws):
            arms[f"dms_no_readout_{draw}"] = space.arm_values(
                dms_pairwise_readout_tensor(
                    cube,
                    posterior,
                    coupling=args.dms_pairwise_coupling,
                    seed=seed + draw,
                    use_readout=False,
                )
            )
    if getattr(args, "score_models", None):
        arms = {name: values for name, values in arms.items() if name in args.score_models}
    names = args.methods or list(METHODS)
    rows: list[dict[str, object]] = []
    for arm, values in arms.items():
        for method in names:
            for trial in range(args.seeds):
                started = time.perf_counter()
                best = run_method(
                    method,
                    space,
                    values,
                    features,
                    budget=args.budget,
                    seed=seed + 137 * trial,
                )
                rows.append(
                    {
                        "dataset": dataset.name,
                        "protein": dataset.protein,
                        "source": dataset.spec.source,
                        "arm": arm,
                        "method": method,
                        "trial": trial,
                        "best_fitness": best,
                        "regret": _regret(values, best),
                        "space_size": space.size,
                        "coverage": coverage,
                        "seconds": time.perf_counter() - started,
                    }
                )
    return rows


def _target_sources(value: str) -> set[str]:
    """Expand the target-source CLI value into source names."""
    return set(TARGET_SOURCES) if value == "all" else {value}


def _target_eligibility(
    dataset,
    minimum_coverage: float,
    minimum_candidates: int = 384,
) -> dict[str, object]:
    """Record whether one measured landscape supports the search task."""
    split = mutation_order_split(dataset.variants, max_train_depth=1)
    informative = has_informative_singles(split.train)
    cube = supported_cube(dataset, split.train)
    measured, coverage = measured_tensor(dataset, cube)
    finite = int(np.isfinite(measured).sum())
    reason = "included"
    if not informative:
        reason = "single mutants carry no variance"
    elif finite < minimum_candidates:
        reason = f"fewer than {minimum_candidates} finite candidates"
    elif coverage < minimum_coverage:
        reason = f"coverage below {minimum_coverage:.3f}"
    return {
        "dataset": dataset.name,
        "protein": dataset.protein,
        "source": dataset.spec.source,
        "supported_domain": cube.size,
        "finite_candidates": finite,
        "coverage": coverage,
        "informative_singles": informative,
        "included": reason == "included",
        "reason": reason,
    }


def _fold_rows(fold, fold_index, cube_names, args) -> list[dict[str, object]]:
    targets = [dataset for dataset in fold.test if dataset.name in cube_names]
    if not targets:
        return []
    prior = LandscapePrior.fit(list(fold.train), max_order=args.max_order)
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(targets):
        if args.only and dataset.name not in set(args.only):
            continue
        started = time.perf_counter()
        found = _score_landscape(dataset, prior, args, fold_index, dataset_index)
        rows.extend(found)
        print(
            f"  {dataset.name}: {len(found)} runs in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return rows


def _graphfla_training(dataset, training_datasets):
    """Exclude every assay of the target protein from prior fitting."""
    training = [item for item in training_datasets if item.protein != dataset.protein]
    if not training:
        raise ValueError(f"no prior training datasets remain for {dataset.protein}")
    return training


def _graphfla_targets(targets, only):
    """Keep each target's seed index fixed when selecting a subset."""
    return [
        (index, dataset)
        for index, dataset in enumerate(targets)
        if not only or dataset.name in set(only)
    ]


def _graphfla_rows(dataset, training_datasets, index, args) -> list[dict[str, object]]:
    prior = LandscapePrior.fit(
        _graphfla_training(dataset, training_datasets), max_order=args.max_order
    )
    started = time.perf_counter()
    rows = _score_landscape(dataset, prior, args, index, 0)
    print(
        f"  {dataset.name}: {len(rows)} runs in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return rows


def _transfer(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank and value agreement between each synthetic arm and the measured cube."""
    rows: list[dict[str, object]] = []
    mean = scores.groupby(["dataset", "arm", "method"])["regret"].mean().reset_index()
    for dataset, group in mean.groupby("dataset"):
        pivot = group.pivot(index="method", columns="arm", values="regret")
        if "measured" not in pivot:
            continue
        for prefix in ("posterior", "dms_pairwise", "dms_no_readout", "potts_low", "rmf_high"):
            draws = [column for column in pivot.columns if column.startswith(f"{prefix}_")]
            if draws:
                # One draw is a benchmark instance. Their mean is what a user who
                # runs the whole ensemble actually sees.
                pivot[f"{prefix}_mean"] = pivot[draws].mean(axis=1)
        real = pivot["measured"]
        for arm in pivot.columns:
            if arm == "measured":
                continue
            other = pivot[arm]
            keep = real.notna() & other.notna()
            if int(keep.sum()) < 4:
                continue
            if real[keep].nunique() < 2 or other[keep].nunique() < 2:
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "n_methods": int(keep.sum()),
                        "kendall_tau": float("nan"),
                        "spearman": float("nan"),
                        "pearson": float("nan"),
                    }
                )
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "n_methods": int(keep.sum()),
                    "kendall_tau": float(kendalltau(real[keep], other[keep]).statistic),
                    "spearman": float(spearmanr(real[keep], other[keep]).statistic),
                    "pearson": float(pearsonr(real[keep], other[keep]).statistic),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "arm",
            "n_methods",
            "kendall_tau",
            "spearman",
            "pearson",
        ],
    )


def _with_ensemble_arms(scores: pd.DataFrame) -> pd.DataFrame:
    """Add expected-regret rows for generator ensembles."""
    frames = [scores]
    for prefix in ("posterior", "dms_pairwise", "dms_no_readout", "potts_low", "rmf_high"):
        selected = scores.loc[scores["arm"].str.startswith(f"{prefix}_")]
        if selected.empty:
            continue
        group_columns = ["dataset", "protein", "method", "trial"]
        if "source" in selected:
            group_columns.insert(2, "source")
        ensemble = (
            selected.groupby(group_columns, as_index=False, dropna=False)["regret"]
            .mean()
            .assign(arm=f"{prefix}_mean")
        )
        frames.append(ensemble)
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.minimum_coverage <= 1.0:
        raise SystemExit("minimum coverage must be between zero and one")
    if args.minimum_candidates < 4:
        raise SystemExit("minimum candidates must be at least four")
    if args.target_source in {"combingym", "all"} and args.combingym_root is None:
        raise SystemExit("CombinGym targets require --combingym-root")
    training_datasets = load_ssmula_landscapes(args.ssmula_root, strict=False)
    if args.combingym_root is not None:
        training_datasets = training_datasets + load_combingym_landscapes(
            args.combingym_root, strict=False
        )

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.graphfla_root is not None:
        targets = load_graphfla_landscapes(args.graphfla_root, strict=False)
        selected = _graphfla_targets(targets, args.only)
        prior_training = {
            dataset.name: [item.name for item in _graphfla_training(dataset, training_datasets)]
            for _, dataset in selected
        }
        (args.output / "prior_training.json").write_text(
            json.dumps(prior_training, indent=2) + "\n"
        )
        batches = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(_graphfla_rows)(dataset, training_datasets, index, args)
            for index, dataset in selected
        )
        validation_panel = "graphfla_protein_holdout"
    else:
        target_sources = _target_sources(args.target_source)
        target_candidates = [
            dataset for dataset in training_datasets if dataset.spec.source in target_sources
        ]
        eligibility = pd.DataFrame(
            _target_eligibility(dataset, args.minimum_coverage, args.minimum_candidates)
            for dataset in target_candidates
        )
        eligibility.to_csv(args.output / "target_eligibility.csv", index=False)
        cube_names = set(eligibility.loc[eligibility["included"], "dataset"])
        if not cube_names or (args.only and not cube_names.intersection(args.only)):
            raise SystemExit("no target landscapes meet the selection rules")
        folds = list(iter_protein_holdouts(training_datasets))
        batches = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(_fold_rows)(fold, index, cube_names, args) for index, fold in enumerate(folds)
        )
        validation_panel = f"{args.target_source}_protein_holdout"
    scores = pd.DataFrame([row for batch in batches for row in batch])
    if scores.empty:
        raise SystemExit("no landscapes scored")
    _validate_scores(scores, expected_trials=args.seeds)

    _write_results(scores, args, validation_panel, started)


def _write_results(scores, args, validation_panel, started) -> None:
    """Write trial statistics and run metadata."""
    transfer = _transfer(scores)
    validity_scores = _with_ensemble_arms(scores)
    pairwise = pairwise_winner_transfer(validity_scores, seed=args.seed)
    reliability = measured_rank_reliability(validity_scores, seed=args.seed)
    adjusted = reliability_adjusted_transfer(transfer, reliability)
    scores.to_csv(args.output / "scores.csv", index=False)
    transfer.to_csv(args.output / "transfer.csv", index=False)
    pairwise.to_csv(args.output / "pairwise_transfer.csv", index=False)
    reliability.to_csv(args.output / "measured_reliability.csv", index=False)
    adjusted.to_csv(args.output / "reliability_adjusted_transfer.csv", index=False)

    summary = {
        "budget": args.budget,
        "validation_panel": validation_panel,
        "target_source": "graphfla" if args.graphfla_root is not None else args.target_source,
        "minimum_coverage": args.minimum_coverage,
        "minimum_candidates": args.minimum_candidates,
        "floor_policy": args.floor_policy,
        "global_epistasis": args.global_epistasis,
        "dms_pairwise_coupling": args.dms_pairwise_coupling,
        "seeds": args.seeds,
        "generator_draws": args.draws,
        "posterior_draws": args.draws,
        "acquisition_pool": 20_000,
        "n_landscapes": int(scores["dataset"].nunique()),
        "n_protein_groups": int(scores["protein"].nunique()),
        "landscapes": sorted(scores["dataset"].unique()),
        "n_methods": int(scores["method"].nunique()),
        "wall_seconds": time.perf_counter() - started,
        "median_regret": scores.groupby(["arm", "method"])["regret"]
        .median()
        .unstack()
        .to_dict("index"),
        "median_transfer": transfer.groupby("arm")[["kendall_tau", "spearman", "pearson"]]
        .median()
        .to_dict("index"),
        "pairwise_winner_accuracy": (
            pairwise.groupby("arm")[["correct_pairs", "scored_pairs"]]
            .sum()
            .assign(accuracy=lambda frame: frame["correct_pairs"] / frame["scored_pairs"])[
                "accuracy"
            ]
            .to_dict()
        ),
        "reliability_pass_rate": (
            adjusted.groupby("arm")["meets_reliability_floor"].mean().to_dict()
        ),
        "measured_split_half_tau": {
            "median": float(reliability["split_half_median"].median()),
            "lower": float(reliability["split_half_lower"].median()),
            "upper": float(reliability["split_half_upper"].median()),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["median_transfer"], indent=2))


if __name__ == "__main__":
    main()
