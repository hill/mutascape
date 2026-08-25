"""Ask what actually controls how rugged a generated landscape is.

Peak fraction is the ruggedness measure the field already uses, it is what makes
a search problem hard, and a complete panel makes it measurable. If the coupling
controlled it, the coupling could be calibrated against it rather than picked.

Two dials are swept against the same measured target. The coupling scales pair
tables, which are shared by every genotype carrying that pair, so raising it
moves whole neighbourhoods together. A genotype-pinned perturbation moves each
genotype alone. Only the second shape creates peaks, and it is also what assay
error contributes to a measured landscape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.generators import dms_pairwise_readout_tensor
from mutascape.benchmarks.splits import (
    has_informative_singles,
    iter_protein_holdouts,
    mutation_order_split,
)
from mutascape.benchmarks.topology import measured_tensor, summarise, supported_cube
from mutascape.completion import LandscapePrior, PosteriorLandscape


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--couplings",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8],
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0],
        help="Genotype-pinned noise added after the readout, in fitness standard deviations",
    )
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--max-cube", type=int, default=200_000)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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
            cube = supported_cube(dataset, split.train)
            if cube.size > args.max_cube:
                print(f"{dataset.name}: skipped, cube {cube.size}", flush=True)
                continue
            measured, coverage = measured_tensor(dataset, cube)
            if coverage < args.min_coverage:
                print(f"{dataset.name}: skipped, coverage {coverage:.3f}", flush=True)
                continue

            # A measured cube is rarely complete. Peak fraction counts peaks
            # per finite cell, so the two tensors have to carry the same holes
            # or the generated one is scored on a larger, better connected
            # graph than the landscape it is standing in for.
            holes = ~np.isfinite(measured)
            target = summarise(measured)["peak_fraction"]
            rows.append(
                {
                    "dataset": dataset.name,
                    "protein": dataset.protein,
                    "source": "measured",
                    "coupling": float("nan"),
                    "draw": -1,
                    "cube_size": cube.size,
                    "peak_fraction": target,
                }
            )

            posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)
            for coupling in args.couplings:
                for draw in range(args.draws):
                    tensor = dms_pairwise_readout_tensor(
                        cube,
                        posterior,
                        coupling=coupling,
                        seed=args.seed + draw,
                    )
                    tensor = np.where(holes, np.nan, tensor)
                    spread = float(np.nanstd(tensor)) or 1.0
                    # Pair tables are shared across genotypes, so they move
                    # whole neighbourhoods together and add few peaks. A
                    # perturbation pinned to the genotype is the shape that
                    # makes peaks, and it is also what assay error contributes
                    # to a measured landscape.
                    noise_rng = np.random.default_rng(args.seed + 991 * draw)
                    pinned = noise_rng.normal(size=tensor.shape)
                    for noise in args.noise_levels:
                        perturbed = tensor + noise * spread * pinned if noise else tensor
                        rows.append(
                            {
                                "dataset": dataset.name,
                                "protein": dataset.protein,
                                "source": "mutascape",
                                "coupling": coupling,
                                "noise": noise,
                                "draw": draw,
                                "cube_size": cube.size,
                                "peak_fraction": summarise(perturbed)["peak_fraction"],
                            }
                        )
            print(f"{dataset.name}: measured peak fraction {target:.5f}", flush=True)

    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "ruggedness_calibration.csv", index=False)

    measured_rows = table.loc[table["source"] == "measured"].set_index("dataset")["peak_fraction"]
    synthetic = table.loc[table["source"] == "mutascape"]

    def _report(column: str, held: str, held_value: float) -> None:
        subset = synthetic.loc[np.isclose(synthetic[held], held_value)]
        grid = subset.groupby(["dataset", column])["peak_fraction"].mean().unstack()
        grid.insert(0, "measured", measured_rows)
        print(f"\npeak fraction against {column}, {held} held at {held_value:g}\n")
        print(grid.to_string(float_format=lambda value: f"{value:.5f}"))
        ratio = grid.drop(columns="measured").div(grid["measured"], axis=0)
        print(f"\nmedian generated over measured, by {column}")
        print(ratio.median().to_string(float_format=lambda value: f"{value:.3f}"))

    _report("coupling", "noise", 0.0)
    _report("noise", "coupling", 0.1)


if __name__ == "__main__":
    main()
