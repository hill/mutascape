"""Evaluate benchmark efficacy on connected measured protein libraries."""

import argparse
import json
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks.diagnostics import fitness_headroom, library_connectivity
from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, SEARCH_PROTOCOL_VERSION, make_features
from mutascape.model import DMSLandscape
from scripts.search_selection_validation import digest, save_json
from scripts.singles_warm_start import DMSOracle

OUT = Path("results/benchmark_efficacy_20260930")
SOURCE = Path("results/search_selection_20260924")
PROTEINS = ("GB1", "TrpB4", "CreiLOV", "eqFP611")
SETTINGS = {
    "mutascape": ("sigmoid", 0.3),
    "zero_pair": ("sigmoid", 0.0),
    "additive": ("linear", 0.0),
    "linear": ("linear", 1.0),
    "pool": ("pool", 0.3),
}
ENSEMBLE = ("mutascape", "linear", "pool")
SEEDS = tuple(2026093071 + i for i in range(5))
BUDGETS = (48, 96)


def connected_rows(codes, sizes, known, size, seed):
    """Choose a connected domain using only genotype coverage and a fixed seed."""
    space = SearchSpace(
        codes, tuple(sizes), np.arange(len(codes)), None, {tuple(c): i for i, c in enumerate(codes)}
    )
    rng = np.random.default_rng(seed)
    selected = set(map(int, known))
    queue = deque(map(int, rng.permutation(known)))
    while queue and len(selected) < size:
        current = queue.popleft()
        for neighbour in rng.permutation(space.neighbours(current)):
            neighbour = int(neighbour)
            if neighbour not in selected:
                selected.add(neighbour)
                queue.append(neighbour)
                if len(selected) == size:
                    break
    if len(selected) != size:
        raise ValueError("the known observations cannot reach the requested library size")
    selected = np.asarray(sorted(selected))
    return selected[np.lexsort(codes[selected].T[::-1])]


def space_from(codes, shape, known, values):
    return SearchSpace(
        codes,
        tuple(shape),
        np.arange(len(codes)),
        int(known[np.argmax(values[known])]),
        {tuple(c): i for i, c in enumerate(codes)},
    )


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        return verify()
    sources = [Path(__file__), Path("docs/benchmark_efficacy_protocol_2026-09-30.md")]
    sources.extend(Path("src/mutascape").rglob("*.py"))
    sources.extend(
        [Path("scripts/singles_warm_start.py"), Path("scripts/search_selection_validation.py")]
    )
    inventory = []
    for i, protein in enumerate(PROTEINS):
        folder = OUT / protein
        folder.mkdir(exist_ok=True)
        source = SOURCE / protein
        frame = pd.read_pickle(source / "frame.pkl")
        original = np.load(source / "design.npz")
        metadata = json.loads((source / "metadata.json").read_text())
        known = np.flatnonzero(frame.depth.to_numpy() <= 1)
        rows = connected_rows(original["codes"], original["shape"], known, 4096, 2026093080 + i)
        frame = frame.iloc[rows].reset_index(drop=True)
        known = np.flatnonzero(frame.depth.to_numpy() <= 1)
        codes, values = original["codes"][rows], frame.fitness.to_numpy()
        space = space_from(codes, original["shape"], known, values)
        topology = library_connectivity(space)
        assert topology["connected_components"] == 1
        assert topology["isolated_candidates"] == 0
        frame.to_pickle(folder / "frame.pkl")
        np.savez(
            folder / "design.npz", codes=codes, shape=original["shape"], known=known, rows=rows
        )
        np.save(folder / "measured.npy", values)
        save_json(folder / "metadata.json", metadata)
        diagnostics = [dict(model="measured", sample=-1, **fitness_headroom(values, known))]
        for label, (mode, strength) in SETTINGS.items():
            for sample, seed in enumerate(SEEDS):
                model = DMSLandscape.from_singles(
                    metadata["sequence"], frame.iloc[known], mode=mode, strength=strength, seed=seed
                )
                generated = model.evaluate_batch(frame.mutations.tolist())
                np.testing.assert_array_equal(generated[known], values[known])
                np.save(folder / f"{label}_{sample}.npy", generated)
                diagnostics.append(
                    dict(model=label, sample=sample, **fitness_headroom(generated, known))
                )
        pd.DataFrame(diagnostics).to_csv(folder / "headroom.csv", index=False)
        inventory.append(
            dict(
                protein=protein,
                **topology,
                known=len(known),
                depths=frame.depth.value_counts().to_dict(),
            )
        )
        for name in ("frame.pkl", "design.npz", "metadata.json"):
            sources.append(source / name)
        sources.extend(folder.glob("*.npy"))
        sources.extend([folder / "frame.pkl", folder / "design.npz", folder / "metadata.json"])
    save_json(
        OUT / "manifest.json",
        dict(
            protocol=SEARCH_PROTOCOL_VERSION,
            proteins=PROTEINS,
            seeds=SEEDS,
            budgets=BUDGETS,
            settings=SETTINGS,
            ensemble=ENSEMBLE,
            repeats_per_sample=6,
            measured_repeats=30,
            methods=list(METHODS),
            inventory=inventory,
            source_sha256={str(p): digest(p) for p in sources},
        ),
    )
    print("Recorded connected libraries and source hashes", flush=True)


def verify():
    for path, expected in json.loads((OUT / "manifest.json").read_text())["source_sha256"].items():
        assert digest(path) == expected, path


def run(protein, label, sample):
    folder = OUT / protein
    destination = folder / f"trials_{label}_{sample}.csv"
    if destination.exists():
        return
    started = time.monotonic()
    design = np.load(folder / "design.npz")
    known = design["known"]
    values = np.load(folder / ("measured.npy" if label == "measured" else f"{label}_{sample}.npy"))
    space = space_from(design["codes"], design["shape"], known, values)
    features = make_features(space)
    scale = max(float(np.ptp(values)), 1e-12)
    headroom = float(values.max() - values[known].max())
    records = []
    for budget in BUDGETS:
        for trial in range(6):
            seed = (931000000 if label == "measured" else 932000000) + 137 * (sample * 6 + trial)
            for method, function in METHODS.items():
                oracle = DMSOracle(values, budget, known)
                with warnings.catch_warnings(record=True) as caught:
                    function(space, oracle, features, seed)
                assert oracle.spent == budget, (protein, label, method, oracle.spent)
                gap = float(values.max() - oracle.best())
                records.append(
                    dict(
                        protein=protein,
                        model=label,
                        sample=sample,
                        trial=trial,
                        budget=budget,
                        method=method,
                        regret=gap / scale,
                        remaining_regret=gap / headroom if headroom > 0 else np.nan,
                        best=oracle.best(),
                        spent=oracle.spent,
                        fallback_queries=oracle.fallback_queries,
                        warnings=len(caught),
                    )
                )
    pd.DataFrame(records).to_csv(destination, index=False)
    print(
        f"Completed {protein}, {label}, sample {sample}, {time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        verify()
        Parallel(n_jobs=args.jobs)(
            delayed(run)(p, m, s)
            for p in PROTEINS
            for m in ("measured", *SETTINGS)
            for s in range(5)
        )
