"""Count how much of a measurement budget the conditioning rule can actually use.

Exact Mobius inversion fixes an interaction only when the variant and every one
of its subvariants were measured. A budget spent on randomly chosen multi-mutants
rarely completes a subvariant set, so most of it reaches the model as nothing.

Example:

    uv run python scripts/diagnose_budget_conditioning.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym \
        --output notebooks/iclr/output_260820_e3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.splits import (
    budget_split,
    has_informative_singles,
    mutation_order_split,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, nargs="*", default=[96, 384])
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = []
    if args.ssmula_root is not None:
        datasets.extend(load_ssmula_landscapes(args.ssmula_root, strict=False))
    if args.combingym_root is not None:
        datasets.extend(load_combingym_landscapes(args.combingym_root, strict=False))
    if not datasets:
        raise SystemExit("No landscapes found. Pass at least one dataset root.")

    # The prior only supplies max_order here, so one fit over everything is enough.
    prior = LandscapePrior.fit(datasets, max_order=2)
    rows: list[dict[str, object]] = []

    for dataset in datasets:
        base = mutation_order_split(dataset.variants, max_train_depth=1)
        if not has_informative_singles(base.train):
            continue
        for budget in args.budget:
            try:
                split = budget_split(dataset.variants, budget, seed=args.seed)
            except ValueError:
                continue
            posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)
            observed = int((split.train["depth"] > 1).sum())
            fixed = len(posterior.coefficients)
            rows.append(
                {
                    "dataset": dataset.name,
                    "protein": dataset.protein,
                    "max_depth": int(dataset.variants["depth"].max()),
                    "budget": budget,
                    "observed_combinations": observed,
                    "coefficients_fixed": fixed,
                    "usable_fraction": fixed / observed if observed else 0.0,
                }
            )

    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "budget_conditioning.csv", index=False)
    summary = frame.groupby("budget")[
        ["observed_combinations", "coefficients_fixed", "usable_fraction"]
    ].median()
    print(summary.round(3).to_string())
    print(f"\nwrote {args.output / 'budget_conditioning.csv'}")


if __name__ == "__main__":
    main()
