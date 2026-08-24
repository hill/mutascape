"""Compare landscape shape and search behaviour across measured and synthetic cubes.

Only complete factorial libraries qualify, so this runs on SSMuLA. Each held-out
landscape yields three cubes over the same domain: the measured one, an ensemble
of posterior draws, and the additive ridge surrogate.

Example:

    uv run python scripts/run_topology_benchmark.py \
        --ssmula-root /path/to/ssmula \
        --output notebooks/iclr/output_260820_e5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.generators import dms_pairwise_readout_tensor
from mutascape.benchmarks.search import (
    TabularOracle,
    greedy_search,
    random_search,
    ridge_guided_search,
)
from mutascape.benchmarks.spectral import calibrate_order_scale
from mutascape.benchmarks.splits import (
    has_informative_singles,
    iter_protein_holdouts,
    mutation_order_split,
)
from mutascape.benchmarks.topology import (
    additive_tensor,
    measured_tensor,
    navigability_summary,
    posterior_tensors,
    summarise,
    supported_cube,
    walsh_l1,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape

METHODS = {
    "random": random_search,
    "greedy": greedy_search,
    "ridge_guided": ridge_guided_search,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument(
        "--combingym-root",
        type=Path,
        help="Included in prior training only; its libraries are not complete cubes",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--search-budget", type=int, default=96)
    parser.add_argument("--search-seeds", type=int, default=5)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--max-cube", type=int, default=200_000)
    parser.add_argument("--dms-pairwise-coupling", type=float, default=0.1)
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Match the per-order scale ratio to the training landscapes' Walsh spectra",
    )
    return parser.parse_args()


def _oracle(cube, tensor) -> TabularOracle:
    variants, indices = cube.variants()
    return TabularOracle(
        {
            tuple(sorted(variant.items())): float(tensor[index])
            for variant, index in zip(variants, indices)
        }
    )


def _search_scores(oracle, candidates, budget, seeds, base_seed) -> dict[str, float]:
    """Mean best-found fitness per method, averaged over search seeds."""
    scores: dict[str, float] = {}
    for name, method in METHODS.items():
        values = [
            method(oracle, candidates, budget=budget, seed=base_seed + offset).best_fitness
            for offset in range(seeds)
        ]
        scores[name] = float(np.mean(values))
    return scores


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cube_sets = load_ssmula_landscapes(args.ssmula_root, strict=False)
    training_extra = (
        load_combingym_landscapes(args.combingym_root, strict=False)
        if args.combingym_root is not None
        else []
    )
    datasets = cube_sets + training_extra
    cube_names = {dataset.name for dataset in cube_sets}

    topology_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for fold_index, fold in enumerate(iter_protein_holdouts(datasets)):
        targets = [dataset for dataset in fold.test if dataset.name in cube_names]
        if not targets:
            continue
        prior = LandscapePrior.fit(list(fold.train), max_order=args.max_order)
        if args.spectral:
            prior = calibrate_order_scale(prior, list(fold.train), seed=args.seed + fold_index)

        for dataset_index, dataset in enumerate(targets):
            split = mutation_order_split(dataset.variants, max_train_depth=1)
            if not has_informative_singles(split.train):
                skipped.append({"dataset": dataset.name, "reason": "singles carry no variance"})
                continue
            cube = supported_cube(dataset, split.train)
            if cube.size > args.max_cube:
                skipped.append({"dataset": dataset.name, "reason": f"cube {cube.size} too large"})
                continue

            measured, coverage = measured_tensor(dataset, cube)
            if not np.isfinite(measured).any():
                skipped.append({"dataset": dataset.name, "reason": "measured cube empty"})
                continue

            posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)
            seed = args.seed + 1000 * fold_index + dataset_index

            arms: dict[str, list[np.ndarray]] = {
                "measured": [measured],
                "additive": [additive_tensor(split.train, cube)],
                "dms_pairwise": [
                    dms_pairwise_readout_tensor(
                        cube,
                        posterior,
                        coupling=args.dms_pairwise_coupling,
                        seed=seed + 10_000 + draw,
                    )
                    for draw in range(args.samples)
                ],
                "posterior": list(
                    posterior_tensors(posterior, cube, n_samples=args.samples, seed=seed)
                ),
            }

            reference = summarise(measured)
            for arm, tensors in arms.items():
                for draw, tensor in enumerate(tensors):
                    row = {
                        "dataset": dataset.name,
                        "protein": dataset.protein,
                        "arm": arm,
                        "draw": draw,
                        "cube_size": cube.size,
                        "measured_coverage": coverage,
                        **summarise(tensor),
                        **navigability_summary(tensor, cube),
                    }
                    row["walsh_l1_vs_measured"] = walsh_l1(reference, row)
                    topology_rows.append(row)

            if args.skip_search:
                print(f"  {dataset.name}: cube {cube.size}, coverage {coverage:.3f}")
                continue

            # Every arm searches the same candidates, so a ranking difference
            # can only come from the landscape and not from the domain.
            variants, indices = cube.variants()
            measurable = [
                variant for variant, index in zip(variants, indices) if np.isfinite(measured[index])
            ]
            if len(measurable) < args.search_budget:
                continue
            oracles = {
                "measured": _oracle(cube, np.nan_to_num(measured, nan=np.nanmin(measured))),
                "additive": _oracle(cube, arms["additive"][0]),
            }
            for draw, tensor in enumerate(arms["posterior"][:5]):
                oracles[f"posterior_{draw}"] = _oracle(cube, tensor)
            for draw, tensor in enumerate(arms["dms_pairwise"][:5]):
                oracles[f"dms_pairwise_{draw}"] = _oracle(cube, tensor)

            for arm, oracle in oracles.items():
                scores = _search_scores(
                    oracle, measurable, args.search_budget, args.search_seeds, seed
                )
                for method, value in scores.items():
                    search_rows.append(
                        {
                            "dataset": dataset.name,
                            "protein": dataset.protein,
                            "arm": arm,
                            "method": method,
                            "best_fitness": value,
                        }
                    )
            print(f"  {dataset.name}: cube {cube.size}, coverage {coverage:.3f}")

    topology = pd.DataFrame(topology_rows)
    search = pd.DataFrame(search_rows)
    topology.to_csv(args.output / "topology.csv", index=False)
    search.to_csv(args.output / "search.csv", index=False)

    # Rank fidelity: does a method's standing on a synthetic cube match its
    # standing on the measured one?
    rank_rows: list[dict[str, object]] = []
    grouped_search = search.groupby("dataset") if "dataset" in search else []
    for dataset_name, group in grouped_search:
        pivot = group.pivot_table(index="method", columns="arm", values="best_fitness")
        if "measured" not in pivot:
            continue
        for arm in pivot.columns:
            if arm == "measured":
                continue
            tau = kendalltau(pivot["measured"], pivot[arm]).statistic
            rank_rows.append(
                {
                    "dataset": dataset_name,
                    "arm": (
                        "posterior"
                        if arm.startswith("posterior")
                        else "dms_pairwise"
                        if arm.startswith("dms_pairwise")
                        else arm
                    ),
                    "kendall_tau": float(tau) if np.isfinite(tau) else float("nan"),
                }
            )
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(args.output / "rank_transfer.csv", index=False)

    summary = {
        "seed": args.seed,
        "spectral": args.spectral,
        "dms_pairwise_coupling": args.dms_pairwise_coupling,
        "posterior_samples": args.samples,
        "search_budget": args.search_budget,
        "search_seeds": args.search_seeds,
        "n_landscapes": int(topology["dataset"].nunique()) if not topology.empty else 0,
        "skipped": skipped,
        "median_topology": (
            topology.groupby("arm")[
                [
                    "n_optima",
                    "peak_fraction",
                    "walsh_nonadditive",
                    "walsh_l1_vs_measured",
                    "beneficial_fraction",
                    "greedy_peak_percentile",
                    "greedy_normalised_regret",
                    "greedy_path_length",
                ]
            ]
            .median()
            .to_dict("index")
            if not topology.empty
            else {}
        ),
        "median_kendall_tau": (
            ranks.groupby("arm")["kendall_tau"].median().to_dict() if not ranks.empty else {}
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["median_topology"], indent=2))
    print(json.dumps(summary["median_kendall_tau"], indent=2))


if __name__ == "__main__":
    main()
