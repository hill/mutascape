"""Replay the affected forest campaigns without replacing the original experiment."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.search_selection_validation import digest, save_json
from scripts.test_usefulness import PROTEINS, SEARCH_MODELS

SOURCE = Path("results/usefulness_20260930")
OUT = Path("results/usefulness_code_audit_20260930")
FORESTS = ("forest_greedy", "forest_thompson")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((SOURCE / "manifest.json").read_text())
    method_path = "src/mutascape/benchmarks/methods.py"
    assert digest(OUT / "methods_before.py") == original["source_sha256"][method_path]
    for path, expected in original["source_sha256"].items():
        if path != method_path:
            assert digest(path) == expected, path
    original["source_sha256"][method_path] = digest(method_path)
    for path in (Path(__file__), Path("scripts/analyse_usefulness.py")):
        original["source_sha256"][str(path)] = digest(path)
    original["audit_correction"] = dict(
        original_manifest_sha256=digest(SOURCE / "manifest.json"),
        forest_fix="Equal observations receive average ranks",
        fidelity_intervals="Crossed resampling retains each shared seed across calibration samples",
        affected_campaigns=2400,
    )
    record = OUT / "manifest.json"
    if record.exists():
        assert json.loads(record.read_text()) == original
    else:
        save_json(record, original)
    for protein in PROTEINS:
        folder = OUT / protein
        folder.mkdir(exist_ok=True)
        for source in (SOURCE / protein).glob("*.npy"):
            target = folder / source.name
            if not target.exists():
                target.symlink_to(source.resolve())
        for filename in ("design.npz", "fidelity.csv", "fits_complete.json"):
            shutil.copyfile(SOURCE / protein / filename, folder / filename)


def run_chunk(protein, model, sample):
    import scripts.test_usefulness as runner

    path = OUT / protein / f"search_{model}_{sample}.csv"
    if path.exists() and len(pd.read_csv(path)) == 120:
        return
    runner.OUT = OUT
    runner.METHODS = {name: runner.METHODS[name] for name in FORESTS}
    runner.search_chunk(protein, model, sample)
    updated = pd.read_csv(path)
    original = pd.read_csv(SOURCE / protein / path.name)
    keep = original.loc[~original.method.isin(FORESTS)]
    combined = pd.concat([keep, updated], ignore_index=True)
    assert len(combined) == 120
    combined.sort_values(["trial", "method"]).to_csv(path, index=False)


def analyse():
    import scripts.analyse_usefulness as statistics
    import scripts.test_usefulness as runner

    runner.OUT = OUT
    runner.verify()
    runner.analyse()
    statistics.OUT = OUT
    statistics.uncertainty()
    statistics.plot()
    old = pd.read_csv(SOURCE / "search_trials.csv")
    new = pd.read_csv(OUT / "search_trials.csv")
    keys = ["protein", "model", "sample", "trial", "method"]
    paired = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(paired) == 14400
    keep = ~paired.method.isin(FORESTS)
    np.testing.assert_array_equal(paired.loc[keep, "regret_old"], paired.loc[keep, "regret_new"])
    paired["difference"] = paired.regret_new - paired.regret_old
    paired["changed"] = paired.difference.abs() > 1e-12
    paired.loc[~keep].groupby(["protein", "method"]).agg(
        changed=("changed", "sum"),
        campaigns=("changed", "size"),
        mean_difference=("difference", "mean"),
    ).to_csv(OUT / "forest_impact.csv")
    (OUT / "analysis_sources.json").write_text(
        json.dumps(
            {
                str(path): digest(path)
                for path in (Path(__file__), Path("scripts/analyse_usefulness.py"))
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "analyse"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        Parallel(n_jobs=args.jobs)(
            delayed(run_chunk)(p, m, s)
            for p in PROTEINS
            for m in ("measured", *SEARCH_MODELS)
            for s in range(3)
        )
    else:
        analyse()
