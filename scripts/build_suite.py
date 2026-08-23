"""Write the shipped benchmark suite as dense tensors.

Each landscape is a complete cube, so the whole suite is a few megabytes and
needs none of the upstream downloads at use time. Difficulty is calibrated
rather than declared: the roughness of each synthetic landscape is chosen so
its local optimum count matches the measured cube it stands in for.

Example:

    uv run python scripts/build_suite.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym \
        --frontier notebooks/iclr/output_frontier/frontier.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks import (
    iter_protein_holdouts,
    load_combingym_landscapes,
    load_ssmula_landscapes,
    mutation_order_split,
)
from mutascape.benchmarks.splits import has_informative_singles
from mutascape.benchmarks.topology import (
    measured_tensor,
    posterior_tensors,
    summarise,
    supported_cube,
    walsh_l1,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape
from mutascape.suite import Benchmark

DEFAULT_ROUGHNESS = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("suite"))
    parser.add_argument("--frontier", type=Path, help="frontier.csv used to calibrate difficulty")
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _calibrated_roughness(frontier: Path | None) -> dict[str, float]:
    """Roughness whose optimum count lands nearest the measured one, per landscape."""
    if frontier is None or not frontier.exists():
        return {}
    frame = pd.read_csv(frontier)
    measured = frame[frame["arm"] == "measured"].set_index("dataset")["n_optima"].to_dict()
    posterior = (
        frame[frame["arm"] == "posterior"]
        .groupby(["dataset", "roughness"])["n_optima"]
        .median()
        .reset_index()
    )
    chosen: dict[str, float] = {}
    for dataset, group in posterior.groupby("dataset"):
        target = measured.get(dataset)
        if target is None or not np.isfinite(target):
            continue
        # Optima counts span orders of magnitude, so match them in log space.
        distance = np.abs(np.log1p(group["n_optima"]) - np.log1p(target))
        chosen[str(dataset)] = float(group.iloc[int(np.argmin(distance))]["roughness"])
    return chosen


def main() -> None:
    args = _parse_args()
    datasets = load_ssmula_landscapes(args.ssmula_root, strict=False)
    cube_names = {dataset.name for dataset in datasets}
    if args.combingym_root is not None:
        datasets = datasets + load_combingym_landscapes(args.combingym_root, strict=False)

    calibration = _calibrated_roughness(args.frontier)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for fold_index, fold in enumerate(iter_protein_holdouts(datasets)):
        targets = [dataset for dataset in fold.test if dataset.name in cube_names]
        if not targets:
            continue
        prior = LandscapePrior.fit(list(fold.train), max_order=args.max_order)
        for dataset_index, dataset in enumerate(targets):
            split = mutation_order_split(dataset.variants, max_train_depth=1)
            if not has_informative_singles(split.train):
                continue
            cube = supported_cube(dataset, split.train)
            measured, coverage = measured_tensor(dataset, cube)
            if not np.isfinite(measured).any():
                continue

            alphabets = tuple(tuple(cube.alphabets[p]) for p in cube.positions)
            reference = summarise(measured)
            roughness = calibration.get(dataset.name, DEFAULT_ROUGHNESS)
            seed = args.seed + 1000 * fold_index + 10 * dataset_index

            bundles = [
                Benchmark(
                    name=f"{dataset.name}_measured",
                    wild_type_sequence=dataset.wild_type_sequence,
                    positions=cube.positions,
                    alphabets=alphabets,
                    values=measured,
                    source="measured",
                    metadata={**reference, "coverage": coverage},
                )
            ]
            posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)
            tensors = posterior_tensors(
                posterior, cube, n_samples=args.draws, seed=seed, roughness=roughness
            )
            for draw, tensor in enumerate(tensors):
                shape = summarise(tensor)
                bundles.append(
                    Benchmark(
                        name=f"{dataset.name}_posterior_{draw}",
                        wild_type_sequence=dataset.wild_type_sequence,
                        positions=cube.positions,
                        alphabets=alphabets,
                        values=tensor,
                        source="posterior",
                        metadata={
                            **shape,
                            "roughness": roughness,
                            "walsh_l1_vs_measured": walsh_l1(reference, shape),
                        },
                    )
                )

            for bundle in bundles:
                path = bundle.save(args.output / f"{bundle.name}.npz")
                manifest.append(
                    {
                        "name": bundle.name,
                        "protein": dataset.protein,
                        "source": bundle.source,
                        "sites": bundle.n_sites,
                        "size": bundle.size,
                        "bytes": path.stat().st_size,
                        **bundle.metadata,
                    }
                )
            print(f"  {dataset.name}: roughness {roughness}, cube {measured.size}", flush=True)

    frame = pd.DataFrame(manifest)
    frame.to_csv(args.output / "manifest.csv", index=False)
    summary = {
        "n_landscapes": int(frame["name"].nunique()),
        "n_proteins": int(frame["protein"].nunique()),
        "total_megabytes": round(float(frame["bytes"].sum()) / 1e6, 2),
        "calibrated": bool(calibration),
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
