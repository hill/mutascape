"""Run the frozen confirmation manifest with checkpoints for each generator."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks.datasets import (
    _graphfla_spec,
    load_combingym_landscapes,
    load_landscape,
    load_ssmula_landscapes,
)
from mutascape.completion import LandscapePrior
from scripts.run_search_transfer import _score_landscape, _validate_scores, _write_results

ROOT = Path("results/review_response_20260923")
GRAPH = Path(".mutascape/worktrees/posterior-landscapes-v1/.mutascape/benchmark-data/graphfla")


def settings(budget, output):
    return SimpleNamespace(
        seed=20260923,
        draws=3,
        seeds=10,
        budget=budget,
        minimum_coverage=1.0,
        minimum_candidates=384,
        floor_policy="censored",
        global_epistasis=True,
        generators="primary",
        dms_pairwise_coupling=0.1,
        methods=None,
        readout_ablation=True,
        graphfla_root=GRAPH,
        target_source="graphfla",
        output=output,
    )


def run_model(dataset, prior, budget, model, index):
    output = ROOT / f"confirmation_b{budget}"
    output.mkdir(exist_ok=True)
    path = output / f"{model}.csv"
    if path.exists():
        return
    args = settings(budget, output)
    args.score_models = [model]
    rows = _score_landscape(dataset, prior, args, index, 0)
    if not rows:
        raise ValueError("Selected confirmation target did not produce trials")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Completed {dataset.name} budget {budget} model {model}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "confirmation_manifest.json").read_text())
    for path, digest in manifest["code_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen model code changed: {path}")
    training = load_ssmula_landscapes(
        ".mutascape/benchmark-data/ssmula", strict=False
    ) + load_combingym_landscapes(".mutascape/benchmark-data/combingym", strict=False)
    tasks = []
    models = [
        "measured",
        "additive",
        *[
            f"{p}_{d}"
            for p in ("dms_pairwise", "dms_no_readout", "potts_low", "rmf_high")
            for d in range(3)
        ],
    ]
    for index, target in enumerate(manifest["targets"]):
        path = GRAPH / "data/BioSequence" / target["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == target["sha256"]
        dataset = load_landscape(
            GRAPH, _graphfla_spec(target["file"], target["protein"], target["reference"])
        )
        prior = LandscapePrior.fit(
            [d for d in training if d.protein != dataset.protein], max_order=3
        )
        (ROOT / "confirmation_prior.json").write_text(json.dumps(prior.to_dict(), indent=2) + "\n")
        tasks.extend((dataset, prior, b, m, index) for b in (384, 192, 96, 48) for m in models)
    Parallel(n_jobs=args.jobs)(delayed(run_model)(*task) for task in tasks)
    for budget in (48, 96, 192, 384):
        output = ROOT / f"confirmation_b{budget}"
        scores = pd.concat(
            [pd.read_csv(output / f"{m}.csv", float_precision="round_trip") for m in models],
            ignore_index=True,
        )
        _validate_scores(scores, expected_trials=10)
        _write_results(
            scores, settings(budget, output), "prospective_graphfla", time.perf_counter()
        )


if __name__ == "__main__":
    main()
