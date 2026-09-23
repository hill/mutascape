"""Run an anchor-preserving readout control with matched coupling and pair seeds."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks.datasets import (
    load_combingym_landscapes,
    load_graphfla_landscapes,
    load_ssmula_landscapes,
)
from mutascape.benchmarks.splits import iter_protein_holdouts
from mutascape.completion import LandscapePrior
from scripts.run_search_transfer import _score_landscape, _target_eligibility


def run_target(dataset, training, fold_index, dataset_index, budget, output):
    path = output / f"b{budget}_{dataset.name}.csv"
    if path.exists():
        return
    prior = LandscapePrior.fit([d for d in training if d.protein != dataset.protein], max_order=3)
    args = SimpleNamespace(
        seed=20260823,
        draws=3,
        seeds=10,
        budget=budget,
        minimum_coverage=0.5,
        floor_policy="censored",
        global_epistasis=True,
        generators="primary",
        dms_pairwise_coupling=0.1,
        methods=None,
        readout_ablation=True,
        score_models=[f"dms_no_readout_{i}" for i in range(3)],
    )
    rows = _score_landscape(dataset, prior, args, fold_index, dataset_index)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Finished {dataset.name} at {budget}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=[48, 96, 192, 384])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    output = Path("results/review_response_20260923/readout_ablation")
    output.mkdir(parents=True, exist_ok=True)
    training = load_ssmula_landscapes(
        ".mutascape/benchmark-data/ssmula", strict=False
    ) + load_combingym_landscapes(".mutascape/benchmark-data/combingym", strict=False)
    targets = []
    # Reproduce original collection-specific indices, including both GB1 assays.
    for source in ("ssmula", "combingym"):
        eligible = {
            d.name
            for d in training
            if d.spec.source == source and _target_eligibility(d, 0.5, 384)["included"]
        }
        for index, fold in enumerate(iter_protein_holdouts(training)):
            selected = [d for d in fold.test if d.name in eligible]
            targets.extend((d, index, j) for j, d in enumerate(selected))
    graph = load_graphfla_landscapes(
        ".mutascape/worktrees/posterior-landscapes-v1/.mutascape/benchmark-data/graphfla",
        strict=True,
    )
    targets.extend((d, index, 0) for index, d in enumerate(graph))
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "budgets": args.budgets,
                "coupling": 0.1,
                "draws": 3,
                "trials": 10,
                "seed": 20260823,
                "targets": [d.name for d, _, _ in targets],
                "control": (
                    "Same unit pair field and coupling. Raw single effects and "
                    "raw single-effect scale. No fitted readout."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    Parallel(n_jobs=args.jobs)(
        delayed(run_target)(d, training, i, j, b, output)
        for b in args.budgets
        for d, i, j in targets
    )


if __name__ == "__main__":
    main()
