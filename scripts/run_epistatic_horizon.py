"""Measure the epistatic horizon of measured and generated landscapes.

SLIP defines the horizon as the number of mutations at which stacked adaptive
singles stop being adaptive, and tunes a Potts coupling by inverting it. SLIP
has no measured combinatorial landscape to compare against, so the target
horizon there is a free choice. A complete measured panel turns the same
quantity into a calibration target.

Wild type and every single mutant are exact under the generator, so the mean
adaptive single effect is shared between the measured and generated landscape.
The horizons differ only through the mean pair term, which is what the coupling
scales. A coupling of zero leaves the fitted readout as the only source of
interaction, which separates the two contributions.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.splits import (
    has_informative_singles,
    iter_protein_holdouts,
    mutation_order_split,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape


def _key(mutations: dict[int, str], wild_type_sequence: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        sorted(
            (int(position), amino_acid)
            for position, amino_acid in mutations.items()
            if wild_type_sequence[int(position) - 1] != amino_acid
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--couplings",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.4],
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=20_000,
        help="Cap on enumerated adaptive pairs before uniform subsampling",
    )
    return parser.parse_args()


def _horizon(mean_single: float, mean_epistasis: float) -> float:
    """Solve ``K s + K(K-1)/2 e = 0`` for the non-zero root."""
    if not np.isfinite(mean_epistasis) or abs(mean_epistasis) < 1e-12:
        return float("inf")
    return float((mean_epistasis - 2.0 * mean_single) / mean_epistasis)


def _adaptive_singles(
    wild_type_fitness: float,
    singles: dict[tuple[int, str], float],
) -> dict[tuple[int, str], float]:
    return {
        mutation: fitness - wild_type_fitness
        for mutation, fitness in singles.items()
        if fitness - wild_type_fitness > 0
    }


def _pairs(
    mutations,
    limit: int,
    rng: np.random.Generator,
) -> list[tuple[tuple[int, str], tuple[int, str]]]:
    pairs = [
        (first, second)
        for first, second in combinations(sorted(mutations), 2)
        if first[0] != second[0]
    ]
    if len(pairs) <= limit:
        return pairs
    chosen = rng.choice(len(pairs), size=limit, replace=False)
    return [pairs[index] for index in sorted(chosen.tolist())]


def _mean_epistasis(
    pairs,
    fitness: dict[tuple[int, str], float],
    wild_type: float,
    lookup,
) -> tuple[float, int]:
    """Mean double difference over the supplied pairs."""
    values = []
    for first, second in pairs:
        double = lookup(first, second)
        if double is None or not np.isfinite(double):
            continue
        values.append(double - fitness[first] - fitness[second] + wild_type)
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def _measured_row(dataset, rng: np.random.Generator, limit: int) -> dict[str, object] | None:
    observed: dict[tuple[tuple[int, str], ...], float] = {}
    for mutations, fitness in dataset.variants[["mutations", "fitness"]].itertuples(
        index=False, name=None
    ):
        if np.isfinite(fitness):
            observed[_key(mutations, dataset.wild_type_sequence)] = float(fitness)
    if () not in observed:
        return None
    wild_type = observed[()]
    singles = {key[0]: value for key, value in observed.items() if len(key) == 1}
    adaptive = _adaptive_singles(wild_type, singles)
    if len(adaptive) < 2:
        return None

    def lookup(first, second):
        return observed.get(tuple(sorted((first, second))))

    effects = {mutation: value - wild_type for mutation, value in singles.items()}
    pairs = _pairs(adaptive, limit, rng)
    mean_epistasis, n_pairs = _mean_epistasis(pairs, singles, wild_type, lookup)
    if not n_pairs:
        return None

    # Conditioning on adaptive singles selects upward measurement error, and a
    # double measured independently does not repeat it. That alone makes the
    # mean double difference negative. The unconditioned mean over every pair
    # carries no such selection, so it is the honest calibration target.
    all_pairs = _pairs(singles, limit, rng)
    mean_all, n_all = _mean_epistasis(all_pairs, singles, wild_type, lookup)

    mean_single = float(np.mean(list(adaptive.values())))
    mean_effect = float(np.mean(np.abs(list(effects.values()))))
    return {
        "n_adaptive_singles": len(adaptive),
        "n_pairs": n_pairs,
        "n_all_pairs": n_all,
        "mean_adaptive_single": mean_single,
        "mean_abs_single": mean_effect,
        "mean_adaptive_epistasis": mean_epistasis,
        "mean_all_epistasis": mean_all,
        "epistatic_horizon": _horizon(mean_single, mean_epistasis),
        "pairs": pairs,
        "all_pairs": all_pairs,
        "adaptive": adaptive,
        "singles": singles,
        "wild_type": wild_type,
    }


def _synthetic_row(oracle, measured: dict[str, object]) -> dict[str, float]:
    adaptive: dict[tuple[int, str], float] = measured["adaptive"]
    singles: dict[tuple[int, str], float] = measured["singles"]
    wild_type = float(measured["wild_type"])

    def lookup(first, second):
        return oracle.evaluate({first[0]: first[1], second[0]: second[1]})

    mean_epistasis, _ = _mean_epistasis(measured["pairs"], singles, wild_type, lookup)
    mean_all, _ = _mean_epistasis(measured["all_pairs"], singles, wild_type, lookup)
    mean_single = float(np.mean(list(adaptive.values())))
    return {
        "mean_adaptive_single": mean_single,
        "mean_abs_single": measured["mean_abs_single"],
        "mean_adaptive_epistasis": mean_epistasis,
        "mean_all_epistasis": mean_all,
        "epistatic_horizon": _horizon(mean_single, mean_epistasis),
    }


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    ssmula = load_ssmula_landscapes(args.ssmula_root, strict=False)
    datasets = ssmula
    if args.combingym_root is not None:
        datasets = datasets + load_combingym_landscapes(args.combingym_root, strict=False)
    scored = {dataset.name for dataset in ssmula}

    rows: list[dict[str, object]] = []
    for fold in iter_protein_holdouts(datasets):
        prior = LandscapePrior.fit(list(fold.train), max_order=args.max_order)
        for dataset in fold.test:
            if dataset.name not in scored:
                continue
            split = mutation_order_split(dataset.variants, max_train_depth=1)
            if not has_informative_singles(split.train):
                continue
            measured = _measured_row(dataset, rng, args.max_pairs)
            if measured is None:
                continue

            rows.append(
                {
                    "dataset": dataset.name,
                    "protein": dataset.protein,
                    "source": "measured",
                    "coupling": float("nan"),
                    "n_adaptive_singles": measured["n_adaptive_singles"],
                    "n_pairs": measured["n_pairs"],
                    "n_all_pairs": measured["n_all_pairs"],
                    "mean_adaptive_single": measured["mean_adaptive_single"],
                    "mean_abs_single": measured["mean_abs_single"],
                    "mean_adaptive_epistasis": measured["mean_adaptive_epistasis"],
                    "mean_all_epistasis": measured["mean_all_epistasis"],
                    "epistatic_horizon": measured["epistatic_horizon"],
                }
            )

            posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)
            for coupling in args.couplings:
                oracle = posterior.sample_dms_pairwise(args.seed, coupling=coupling)
                synthetic = _synthetic_row(oracle, measured)
                rows.append(
                    {
                        "dataset": dataset.name,
                        "protein": dataset.protein,
                        "source": "mutascape",
                        "coupling": coupling,
                        "n_adaptive_singles": measured["n_adaptive_singles"],
                        "n_pairs": measured["n_pairs"],
                        "n_all_pairs": measured["n_all_pairs"],
                        **synthetic,
                    }
                )
            horizon = measured["epistatic_horizon"]
            print(f"{dataset.name}: measured horizon {horizon:.2f}", flush=True)

    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)

    # The horizon has a pole where the mean pair term crosses zero, so a single
    # landscape can report a large negative value. The mean pair term itself,
    # scaled by the mean adaptive single effect, stays within fixed limits and carries the
    # sign, which is what separates systematic epistasis from noise.
    table["epistasis_ratio"] = table["mean_adaptive_epistasis"] / table["mean_adaptive_single"]
    table["all_epistasis_ratio"] = table["mean_all_epistasis"] / table["mean_abs_single"]
    table.to_csv(args.output / "epistatic_horizon.csv", index=False)

    finite = table.loc[np.isfinite(table["epistatic_horizon"])]
    measured = finite.loc[finite["source"] == "measured"]
    measured_median = measured["epistatic_horizon"].median()
    print(f"\nmeasured landscapes: {len(measured)}")
    print(f"measured median horizon: {measured_median:.3f}")
    print(f"measured median adaptive-pair ratio: {measured['epistasis_ratio'].median():.3f}")
    print(f"measured median all-pair ratio: {measured['all_epistasis_ratio'].median():.3f}")
    print(f"measured deleterious-mean landscapes: {int((measured['epistasis_ratio'] < 0).sum())}\n")

    synthetic = finite.loc[finite["source"] == "mutascape"]
    summary = (
        synthetic.groupby("coupling")
        .agg(
            median_horizon=("epistatic_horizon", "median"),
            median_ratio=("epistasis_ratio", "median"),
            median_all_ratio=("all_epistasis_ratio", "median"),
            deleterious_mean=("epistasis_ratio", lambda values: int((values < 0).sum())),
            n=("epistatic_horizon", "size"),
        )
        .reset_index()
    )
    summary["horizon_gap"] = (summary["median_horizon"] - measured_median).abs()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
