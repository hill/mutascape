"""Measure the epistatic horizon of measured landscapes and of synthetic arms.

Thomas et al. (2022) define the epistatic horizon as the mutation count at which
adaptive single mutants stop combining into adaptive multi-mutants,

    K s_plus + K (K - 1) / 2 * eps_plus_plus = 0,   K_epi = 1 - 2 s_plus / eps_plus_plus,

where ``s_plus`` averages the adaptive single effects and ``eps_plus_plus``
averages pairwise epistasis over pairs of adaptive singles. They tune Potts
landscapes to a chosen K_epi and report 166 for an untuned model on a protein of
length 118. They never report it for a landscape that was measured, because
their pipeline contains none.

Complete empirical libraries supply that reference value, which turns a free
difficulty dial into a quantity with a target.

Example:

    uv run python scripts/measure_epistatic_horizon.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym \
        --output notebooks/iclr/output_260822_horizon
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.splits import has_informative_singles, mutation_order_split
from mutascape.benchmarks.topology import additive_tensor, supported_cube
from mutascape.completion import LandscapePrior, PosteriorLandscape, _observation_map, _subsets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--max-cube", type=int, default=200_000)
    return parser.parse_args()


def epistatic_horizon(observed: dict, *, adaptive_only: bool = True) -> dict[str, float]:
    """Return the horizon and the two averages behind it.

    ``adaptive_only`` restricts the pair average to pairs whose two singles are
    both beneficial, which is the population a naive design strategy would
    combine.
    """
    wt = observed[()]
    singles = {
        genotype[0]: value - wt for genotype, value in observed.items() if len(genotype) == 1
    }
    adaptive = {key: value for key, value in singles.items() if value > 0}
    if not adaptive:
        return {
            "s_plus": float("nan"),
            "eps_plus": float("nan"),
            "k_epi": float("nan"),
            "n_adaptive_singles": 0.0,
            "n_adaptive_pairs": 0.0,
        }

    epsilons: list[float] = []
    for genotype, value in observed.items():
        if len(genotype) != 2:
            continue
        if any(subset not in observed for subset in _subsets(genotype)):
            continue
        left, right = genotype
        if adaptive_only and not (left in adaptive and right in adaptive):
            continue
        epsilons.append(
            sum((-1.0) ** (2 - len(subset)) * observed[subset] for subset in _subsets(genotype))
        )

    s_plus = float(np.mean(list(adaptive.values())))
    eps_plus = float(np.mean(epsilons)) if epsilons else float("nan")
    horizon = 1.0 - 2.0 * s_plus / eps_plus if epsilons and abs(eps_plus) > 1e-12 else float("nan")
    return {
        "s_plus": s_plus,
        "eps_plus": eps_plus,
        "k_epi": horizon,
        "n_adaptive_singles": float(len(adaptive)),
        "n_adaptive_pairs": float(len(epsilons)),
    }


def _observed_from_tensor(cube, tensor, wild_type_sequence: str) -> dict:
    """Turn a complete cube back into the observation map the horizon needs."""
    variants, indices = cube.variants()
    observed = {}
    for variant, index in zip(variants, indices):
        value = float(tensor[index])
        if not np.isfinite(value):
            continue
        key = tuple(
            sorted(
                (int(position), amino_acid)
                for position, amino_acid in variant.items()
                if wild_type_sequence[int(position) - 1] != amino_acid
            )
        )
        observed[key] = value
    return observed


def main() -> None:
    args = _parse_args()
    cube_sets = load_ssmula_landscapes(args.ssmula_root, strict=False) if args.ssmula_root else []
    extra = (
        load_combingym_landscapes(args.combingym_root, strict=False) if args.combingym_root else []
    )
    datasets = cube_sets + extra
    if not datasets:
        raise SystemExit("No landscapes found. Pass at least one dataset root.")

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        split = mutation_order_split(dataset.variants, max_train_depth=1)
        if not has_informative_singles(split.train):
            continue
        observed = _observation_map(dataset.variants, dataset.wild_type_sequence)
        rows.append(
            {
                "dataset": dataset.name,
                "protein": dataset.protein,
                "arm": "measured",
                "n_sites": len(dataset.mutable_positions),
                **epistatic_horizon(observed),
            }
        )

    # Synthetic arms only make sense where a complete cube is available.
    prior = LandscapePrior.fit(datasets, max_order=2)
    for dataset in cube_sets:
        split = mutation_order_split(dataset.variants, max_train_depth=1)
        if not has_informative_singles(split.train):
            continue
        cube = supported_cube(dataset, split.train)
        if cube.size > args.max_cube:
            continue
        posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)
        variants, indices = cube.variants()
        wt = posterior.observations[()]
        shape = tuple(len(cube.alphabets[p]) for p in cube.positions)

        additive = additive_tensor(split.train, cube)
        rows.append(
            {
                "dataset": dataset.name,
                "protein": dataset.protein,
                "arm": "additive",
                "n_sites": len(dataset.mutable_positions),
                **epistatic_horizon(
                    _observed_from_tensor(cube, additive, dataset.wild_type_sequence)
                ),
            }
        )
        for draw in range(args.samples):
            values = posterior.sample(args.seed + draw).evaluate_batch(variants) - wt
            tensor = np.empty(shape, dtype=np.float64)
            for index, value in zip(indices, values):
                tensor[index] = value
            rows.append(
                {
                    "dataset": dataset.name,
                    "protein": dataset.protein,
                    "arm": "posterior",
                    "draw": draw,
                    "n_sites": len(dataset.mutable_positions),
                    **epistatic_horizon(
                        _observed_from_tensor(cube, tensor, dataset.wild_type_sequence)
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "epistatic_horizon.csv", index=False)

    finite = frame[np.isfinite(frame["k_epi"])]
    summary = {
        "seed": args.seed,
        "n_rows": int(len(frame)),
        "median_k_epi": finite.groupby("arm")["k_epi"].median().to_dict(),
        "negative_k_epi_fraction": (
            finite.assign(neg=finite["k_epi"] < 0).groupby("arm")["neg"].mean().to_dict()
        ),
        "median_eps_plus": finite.groupby("arm")["eps_plus"].median().to_dict(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    measured = finite[finite["arm"] == "measured"]
    print(
        measured[["dataset", "n_sites", "s_plus", "eps_plus", "k_epi", "n_adaptive_pairs"]]
        .round(3)
        .to_string(index=False)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
