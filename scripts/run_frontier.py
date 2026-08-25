"""Trace the frontier between point accuracy and landscape realism.

The most accurate model of a landscape is its conditional mean. A conditional
mean is smooth, holds one local optimum, and cannot rank a search method. One
knob scales every structure the posterior adds beyond the measured singles and
the readout, which turns that tension into a measured curve instead of an
excuse for a lost decimal place.

Example:

    uv run python scripts/run_frontier.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mutascape.benchmarks import (
    iter_protein_holdouts,
    load_combingym_landscapes,
    load_ssmula_landscapes,
    mutation_order_split,
)
from mutascape.benchmarks.baselines import RidgeLandscapeBaseline
from mutascape.benchmarks.splits import has_informative_singles
from mutascape.benchmarks.topology import (
    additive_tensor,
    measured_tensor,
    posterior_tensors,
    summarise,
    supported_cube,
    walsh_l1,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape

ROUGHNESS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("notebooks/iclr/output_frontier"))
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _accuracy(measured, tensor, mask) -> dict[str, float]:
    """Error against every measured genotype above first order."""
    real = measured[mask]
    predicted = tensor[mask]
    keep = np.isfinite(real) & np.isfinite(predicted)
    if int(keep.sum()) < 8:
        return {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
    error = predicted[keep] - real[keep]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(spearmanr(real[keep], predicted[keep]).statistic),
    }


def main() -> None:
    args = _parse_args()
    datasets = load_ssmula_landscapes(args.ssmula_root, strict=False)
    cube_names = {dataset.name for dataset in datasets}
    if args.combingym_root is not None:
        datasets = datasets + load_combingym_landscapes(args.combingym_root, strict=False)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
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

            # Depth of every cube cell, so accuracy is scored where the model
            # was never told the answer.
            grids = np.indices(measured.shape)
            wild_type = [
                cube.alphabets[position].index(cube.wild_type[position])
                for position in cube.positions
            ]
            depth = sum(grids[site] != wild_type[site] for site in range(len(wild_type)))
            held_out = np.isfinite(measured) & (depth >= 2)

            reference = summarise(measured)
            seed = args.seed + 1000 * fold_index + 10 * dataset_index
            posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)

            ridge_tensors = {
                "ridge_additive": additive_tensor(split.train, cube),
            }
            baseline = RidgeLandscapeBaseline(max_order=2).fit(split.train)
            variants, indices = cube.variants()
            wt = float(split.train.loc[split.train["depth"] == 0, "fitness"].iloc[0])
            pairwise = np.empty(measured.shape, dtype=np.float64)
            for index, value in zip(indices, baseline.predict(variants) - wt):
                pairwise[index] = value
            ridge_tensors["ridge_pairwise"] = pairwise

            for name, tensor in ridge_tensors.items():
                shape = summarise(tensor)
                rows.append(
                    {
                        "dataset": dataset.name,
                        "protein": dataset.protein,
                        "arm": name,
                        "roughness": float("nan"),
                        "draw": 0,
                        "coverage": coverage,
                        **shape,
                        "walsh_l1_vs_measured": walsh_l1(reference, shape),
                        **_accuracy(measured, tensor, held_out),
                    }
                )

            rows.append(
                {
                    "dataset": dataset.name,
                    "protein": dataset.protein,
                    "arm": "measured",
                    "roughness": float("nan"),
                    "draw": 0,
                    "coverage": coverage,
                    **reference,
                    "walsh_l1_vs_measured": 0.0,
                    "mae": 0.0,
                    "rmse": 0.0,
                    "spearman": 1.0,
                }
            )

            for roughness in ROUGHNESS:
                tensors = posterior_tensors(
                    posterior, cube, n_samples=args.draws, seed=seed, roughness=roughness
                )
                for draw, tensor in enumerate(tensors):
                    shape = summarise(tensor)
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "protein": dataset.protein,
                            "arm": "posterior",
                            "roughness": roughness,
                            "draw": draw,
                            "coverage": coverage,
                            **shape,
                            "walsh_l1_vs_measured": walsh_l1(reference, shape),
                            **_accuracy(measured, tensor, held_out),
                        }
                    )
            print(f"  {dataset.name}: cube {measured.size}, coverage {coverage:.3f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "frontier.csv", index=False)

    posterior_rows = frame[frame["arm"] == "posterior"]
    curve = (
        posterior_rows.groupby("roughness")[
            ["mae", "spearman", "n_optima", "walsh_l1_vs_measured", "walsh_nonadditive"]
        ]
        .median()
        .reset_index()
    )
    others = (
        frame[frame["arm"] != "posterior"]
        .groupby("arm")[
            ["mae", "spearman", "n_optima", "walsh_l1_vs_measured", "walsh_nonadditive"]
        ]
        .median()
    )
    summary = {
        "n_landscapes": int(frame["dataset"].nunique()),
        "draws": args.draws,
        "curve": curve.to_dict("records"),
        "reference": others.to_dict("index"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(curve.round(3).to_string(index=False))
    print(others.round(3).to_string())


if __name__ == "__main__":
    main()
