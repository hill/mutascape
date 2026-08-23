"""Does a method's regret on a synthetic landscape predict its regret on the real one?

A benchmark generator exists to rank methods. This scores twelve search methods
on every complete measured cube, then on synthetic cubes built from wild type
and singles alone, and asks whether the two rankings agree.

Example:

    uv run python scripts/run_search_transfer.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym
"""

from __future__ import annotations

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
    load_ssmula_landscapes,
    mutation_order_split,
)
from mutascape.benchmarks.generators import build_generators
from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features, run_method
from mutascape.benchmarks.splits import has_informative_singles
from mutascape.benchmarks.topology import (
    additive_tensor,
    measured_tensor,
    posterior_tensors,
    supported_cube,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path)
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
        action="store_true",
        help="Add NK, Rough Mount Fuji and Potts arms to the audit",
    )
    return parser.parse_args()


def _regret(values: np.ndarray, best: float) -> float:
    """Simple regret normalised by the landscape's own range."""
    top = float(np.max(values))
    bottom = float(np.min(values))
    if not np.isfinite(top - bottom) or top <= bottom:
        return float("nan")
    return (top - best) / (top - bottom)


def _score_landscape(dataset, prior, args, fold_index, dataset_index) -> list[dict[str, object]]:
    """Run every method on every arm of one landscape."""
    split = mutation_order_split(dataset.variants, max_train_depth=1)
    if not has_informative_singles(split.train):
        return []
    cube = supported_cube(dataset, split.train)
    measured, coverage = measured_tensor(dataset, cube)
    mask = np.isfinite(measured)
    if int(mask.sum()) < 4 * args.budget:
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
    for draw, tensor in enumerate(
        posterior_tensors(posterior, cube, n_samples=args.draws, seed=seed)
    ):
        arms[f"posterior_{draw}"] = space.arm_values(tensor)
    if args.generators:
        for name, tensor in build_generators(cube, split.train, seed=seed).items():
            arms[name] = space.arm_values(tensor)

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


def _fold_rows(fold, fold_index, cube_names, args) -> list[dict[str, object]]:
    targets = [dataset for dataset in fold.test if dataset.name in cube_names]
    if not targets:
        return []
    prior = LandscapePrior.fit(list(fold.train), max_order=args.max_order)
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(targets):
        started = time.perf_counter()
        found = _score_landscape(dataset, prior, args, fold_index, dataset_index)
        rows.extend(found)
        print(
            f"  {dataset.name}: {len(found)} runs in {time.perf_counter() - started:.1f}s",
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
        draws = [column for column in pivot.columns if column.startswith("posterior_")]
        if draws:
            # One draw is a benchmark instance. Their mean is what a user who
            # runs the whole ensemble actually sees.
            pivot["posterior_mean"] = pivot[draws].mean(axis=1)
        real = pivot["measured"]
        for arm in pivot.columns:
            if arm == "measured":
                continue
            other = pivot[arm]
            keep = real.notna() & other.notna()
            if int(keep.sum()) < 4:
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
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    datasets = load_ssmula_landscapes(args.ssmula_root, strict=False)
    cube_names = {dataset.name for dataset in datasets}
    if args.combingym_root is not None:
        datasets = datasets + load_combingym_landscapes(args.combingym_root, strict=False)
    if args.only:
        cube_names = {name for name in cube_names if name in set(args.only)}

    args.output.mkdir(parents=True, exist_ok=True)
    folds = list(iter_protein_holdouts(datasets))
    started = time.perf_counter()
    batches = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(_fold_rows)(fold, index, cube_names, args) for index, fold in enumerate(folds)
    )
    scores = pd.DataFrame([row for batch in batches for row in batch])
    if scores.empty:
        raise SystemExit("no landscapes scored")

    transfer = _transfer(scores)
    scores.to_csv(args.output / "scores.csv", index=False)
    transfer.to_csv(args.output / "transfer.csv", index=False)

    summary = {
        "budget": args.budget,
        "floor_policy": args.floor_policy,
        "global_epistasis": args.global_epistasis,
        "seeds": args.seeds,
        "posterior_draws": args.draws,
        "acquisition_pool": 20_000,
        "n_landscapes": int(scores["dataset"].nunique()),
        "n_methods": int(scores["method"].nunique()),
        "wall_seconds": time.perf_counter() - started,
        "median_regret": scores.groupby(["arm", "method"])["regret"]
        .median()
        .unstack()
        .to_dict("index"),
        "median_transfer": transfer.groupby("arm")[["kendall_tau", "spearman", "pearson"]]
        .median()
        .to_dict("index"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["median_transfer"], indent=2))


if __name__ == "__main__":
    main()
