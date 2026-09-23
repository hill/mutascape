"""Test search rankings under the confirmed distribution-calibrated model."""

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features, run_method
from scripts.confirm_quantile_pilot import OUT, PROTEINS, SEEDS, digest, load


def run_search(name, size, budget, model, seed):
    folder = OUT / name / f"p{size}"
    output = folder / f"search_b{budget}_{model}_{seed}.csv"
    if output.exists():
        return
    frame, _, _, codes, shape, _ = load(name)
    pilots = pd.read_csv(OUT / name / "pilot.csv").row.to_numpy()[:size]
    included = (frame.depth > 1).to_numpy(copy=True)
    included[pilots] = False
    selected = np.flatnonzero(included)
    order = np.argsort(np.ravel_multi_index(codes[selected].T, shape))
    selected = selected[order]
    search_codes = codes[selected]
    space = SearchSpace(
        search_codes,
        shape,
        selected,
        None,
        {tuple(int(v) for v in code): i for i, code in enumerate(search_codes)},
    )
    values = (
        frame.fitness.to_numpy() if model == "measured" else np.load(folder / f"{model}_{seed}.npy")
    )
    values = values[selected]
    features = make_features(space)
    rows = []
    for method in METHODS:
        for trial in range(10):
            start = time.perf_counter()
            best = run_method(
                method, space, values, features, budget=budget, seed=2026100100 + trial * 137
            )
            rows.append(
                dict(
                    protein=name,
                    pilot_size=size,
                    budget=budget,
                    model=model,
                    draw=seed,
                    method=method,
                    trial=trial,
                    best=best,
                    regret=(values.max() - best) / np.ptp(values),
                    space_size=space.size,
                    seconds=time.perf_counter() - start,
                )
            )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Searched {name} pilot {size} budget {budget} {model} {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads((OUT / "protocol.json").read_text())
    for p, checksum in protocol["code_sha256"].items():
        assert digest(p) == checksum, p
    tasks = []
    for size in (48, 192):
        for name in PROTEINS:
            if not (OUT / name / f"p{size}" / "done.json").exists():
                continue
            for budget in (48, 96):
                tasks.append((name, size, budget, "measured", 0))
                tasks.extend(
                    (name, size, budget, model, seed)
                    for model in ("reference", "quantile")
                    for seed in SEEDS
                )
    Parallel(n_jobs=args.jobs)(delayed(run_search)(*t) for t in tasks)
    print(f"Completed {len(tasks)} search groups", flush=True)


if __name__ == "__main__":
    main()
