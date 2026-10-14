"""Check controlled search results with 32 new measurements before model fitting."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks import methods
from scripts import test_controlled_benchmarks as controlled
from scripts.search_selection_validation import digest, save_json

OUT = Path("results/controlled_initialisation_audit_20260930")
SOURCE = controlled.OUT


def fresh_initial_rows(space, seed, size):
    """Sample only unknown combinations in this complete binary experiment."""
    unknown = np.flatnonzero(space.codes.sum(axis=1) > 1)
    return np.random.default_rng(seed).choice(unknown, min(size, len(unknown)), replace=False)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    record = json.loads((SOURCE / "manifest.json").read_text())
    sources = record["source_sha256"]
    for path, expected in sources.items():
        if path != "src/mutascape/benchmarks/methods.py":
            assert digest(path) == expected, path
    sources["src/mutascape/benchmarks/methods.py"] = digest("src/mutascape/benchmarks/methods.py")
    sources[str(Path(__file__))] = digest(__file__)
    record["sensitivity"] = {
        "original_manifest_sha256": digest(SOURCE / "manifest.json"),
        "change": "All model methods start with 32 new combinations, excluding WT and singles",
        "interpretation": "Post-audit sensitivity analysis, preserving all original outcomes",
    }
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        save_json(path, record)
    rows = []
    _, _, codes, _ = controlled.inputs("CreiLOV")
    known = np.flatnonzero(codes.sum(axis=1) <= 1)
    for sample in range(5):
        for trial in range(10):
            seed = 930500000 + 137 * (sample * 10 + trial)
            initial = np.random.default_rng(seed).choice(len(codes), 32, replace=False)
            overlap = len(np.intersect1d(initial, known))
            rows.append(dict(sample=sample, trial=trial, known_overlap=overlap))
    pd.DataFrame(rows).to_csv(OUT / "initial_overlap.csv", index=False)


def run(protein, strength, sample):
    controlled.OUT = OUT
    with patch.object(methods, "initial_rows", fresh_initial_rows):
        controlled.run(protein, strength, sample)


def analyse():
    controlled.OUT = OUT
    controlled.analyse()
    table = pd.read_csv(OUT / "trials.csv")
    early = table.loc[table.budget == 32].pivot(
        index=["protein", "strength", "sample", "trial"], columns="method", values="regret"
    )
    for method in ("additive_ridge", "pairwise_ridge", "gp_ei"):
        np.testing.assert_array_equal(early["random"], early[method])
    original = pd.read_csv(SOURCE / "trials.csv")
    keys = ["protein", "strength", "sample", "budget", "trial", "method"]
    paired = original.merge(table, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    random = paired.loc[paired.method == "random"]
    np.testing.assert_array_equal(random.regret_old, random.regret_new)
    paired.to_csv(OUT / "initialisation_comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "analyse"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        controlled.OUT = OUT
        controlled.verify()
        if args.stage == "run":
            Parallel(n_jobs=args.jobs)(
                delayed(run)(p, s, i)
                for p in controlled.PROTEINS
                for s in controlled.STRENGTHS
                for i in range(5)
            )
        else:
            analyse()
