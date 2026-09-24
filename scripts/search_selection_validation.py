"""Run the frozen search-selection experiment from 24 September 2026."""

import argparse
import ast
import hashlib
import json
import platform
import re
import time
import urllib.request
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features
from mutascape.model import DMSLandscape
from scripts.investigate_singles_only import CONFIRMATION, DEVELOPMENT, data
from scripts.singles_warm_start import DMSOracle

OUT = Path("results/search_selection_20260924")
SOURCE = Path(".mutascape/benchmark-data/nn4dms-434e354")
REVISION = "434e354828f306cbb6796033ef1169cbe5207350"
BASE_URL = f"https://raw.githubusercontent.com/gitter-lab/nn4dms/{REVISION}/"
NEW = {"Pab1": "pab1", "Ube4b": "ube4b", "Bgl3": "bgl3", "avGFP": "avgfp"}
SAMPLE_SEEDS = tuple(2026092411 + i for i in range(5))
SEED_BASES = {"development": 240910000, "evaluation": 240920000, "synthetic": 240930000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fetch(relative):
    path = SOURCE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(BASE_URL + relative, timeout=60).read())
    return path


def parse_substitutions(variant, sequence, offset):
    """Read archive coordinates and reject unsupported variant types."""
    mutations = {}
    for token in variant.split(","):
        match = re.fullmatch(r"([A-Z])(\d+)([ACDEFGHIKLMNPQRSTVWY])", token)
        if match is None:
            return None
        wild, position, residue = match.groups()
        index = int(position) - offset
        if not 0 <= index < len(sequence) or sequence[index] != wild:
            return None
        if index + 1 in mutations:
            return None
        if wild != residue:
            mutations[index + 1] = residue
    return mutations


def retain_covered_positions(frame, limit=16):
    """Choose positions through genotype coverage, without reading scores."""
    supported = {item for m in frame.loc[frame.depth == 1, "mutations"] for item in m.items()}
    frame = frame.loc[frame.mutations.map(lambda m: set(m.items()) <= supported)].copy()
    counts = Counter(p for m in frame.loc[frame.depth > 1, "mutations"] for p in m)
    positions = sorted(sorted(counts, key=lambda p: (-counts[p], p))[:limit])
    retained = frame.loc[frame.mutations.map(lambda m: set(m) <= set(positions))].copy()
    return retained.reset_index(drop=True), positions, len(frame)


def prepare_new(name, stem, constants):
    source = fetch(f"data/{stem}/{stem}.tsv")
    sequence = constants[stem]["wt_aa"]
    offset = constants[stem]["wt_ofs"]
    raw = pd.read_csv(source, sep="\t")
    frame = raw.rename(columns={"score": "fitness"}).copy()
    frame["mutations"] = frame.variant.map(lambda v: parse_substitutions(v, sequence, offset))
    frame = frame.loc[frame.mutations.notna() & np.isfinite(frame.fitness)].copy()
    frame["depth"] = frame.mutations.map(len)
    frame, positions, supported_count = retain_covered_positions(frame)
    if stem == "avgfp":
        wt = 3.7192121319
        wt_source = "code/parse_source_data.py, parse_avgfp WT subtraction"
    elif stem == "ube4b":
        original = pd.read_excel(fetch("source_data/ube4b/ube4b.xlsx"))
        wt_rows = original.loc[original.seqID.str.strip() == "NA-NA", "nscor_log2_ratio"]
        assert len(wt_rows) == 1
        wt = float(wt_rows.iloc[0])
        wt_source = "source_data/ube4b/ube4b.xlsx, NA-NA reference row"
    else:
        wt = 0.0
        wt_source = (
            "Melamed 2013 WT enrichment equals one, followed by archive log2 transform"
            if stem == "pab1"
            else "code/enrich2.py, ratios with WT normalisation"
        )
    frame = pd.concat(
        [pd.DataFrame([dict(mutations={}, depth=0, fitness=wt)]), frame], ignore_index=True
    )
    keys = frame.mutations.map(lambda m: tuple(sorted(m.items())))
    if keys.duplicated().any():
        raise ValueError(f"Duplicate genotypes in {name}")
    singles = frame.loc[frame.depth <= 1]
    alphabets = {
        p: sorted({sequence[p - 1]} | {m[p] for m in singles.mutations if p in m})
        for p in positions
    }
    codes = np.array(
        [
            [alphabets[p].index(m.get(p, sequence[p - 1])) for p in positions]
            for m in frame.mutations
        ],
        dtype=np.int16,
    )
    shape = tuple(len(alphabets[p]) for p in positions)
    metadata = dict(
        sequence=sequence,
        source=str(source),
        source_sha256=digest(source),
        source_revision=REVISION,
        archive_offset=offset,
        selected_local_positions=positions,
        wild_type_fitness=wt,
        wild_type_source=wt_source,
        raw_rows=len(raw),
        supported_rows=supported_count,
    )
    return frame, {"codes": codes, "shape": shape}, metadata


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise RuntimeError("A frozen manifest already exists")
    constant_file = fetch("code/constants.py")
    constants = ast.literal_eval(ast.parse(constant_file.read_text()).body[1].value)
    for filename in ["code/parse_source_data.py", "code/enrich2.py", "source_data/README.md"]:
        fetch(filename)
    inventory = []
    for name in (*DEVELOPMENT, *CONFIRMATION, *NEW):
        if name in NEW:
            frame, design, metadata = prepare_new(name, NEW[name], constants)
            panel = "new"
        else:
            frame, design, metadata = data(name)
            panel = "development" if name in DEVELOPMENT else "repeat"
        frame = frame[["mutations", "depth", "fitness"]].reset_index(drop=True)
        combinations = int((frame.depth > 1).sum())
        budgets = [b for b in (48, 96) if 10 * b <= combinations]
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        frame.to_pickle(folder / "frame.pkl")
        np.savez(folder / "design.npz", codes=design["codes"], shape=design["shape"])
        metadata.update(panel=panel, budgets=budgets)
        save_json(folder / "metadata.json", metadata)
        np.save(folder / "fitness_measured_0.npy", frame.fitness.to_numpy())
        if panel != "development" and budgets:
            singles = frame.loc[frame.depth <= 1]
            for label in ("mutascape", "additive", "zero_pair"):
                for sample in range(5) if label == "mutascape" else [0]:
                    model = DMSLandscape.from_singles(
                        metadata["sequence"],
                        singles,
                        seed=SAMPLE_SEEDS[sample],
                        strength=0.3 if label == "mutascape" else 0.0,
                        mode="linear" if label == "additive" else "sigmoid",
                        margin=0.05,
                    )
                    values = model.evaluate_batch(frame.mutations.tolist())
                    np.testing.assert_array_equal(values[frame.depth <= 1], singles.fitness)
                    np.save(folder / f"fitness_{label}_{sample}.npy", values)
        inventory.append(
            dict(
                protein=name,
                panel=panel,
                measured=len(frame),
                singles=int((frame.depth == 1).sum()),
                combinations=combinations,
                sites=len(design["shape"]),
                budgets=budgets,
                reason="included" if budgets else "fewer than 480 supported combinations",
            )
        )
        print(inventory[-1], flush=True)
    save_json(OUT / "inventory.json", inventory)


def freeze():
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError("A frozen manifest already exists")
    code_files = [
        Path(__file__).relative_to(Path.cwd()),
        Path("scripts/singles_warm_start.py"),
        Path("docs/search_selection_protocol_2026-09-24.md"),
        *Path("src/mutascape").rglob("*.py"),
    ]
    input_files = [p for p in OUT.rglob("*") if p.is_file()]
    import sklearn

    manifest = dict(
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        source_revision=REVISION,
        python=platform.python_version(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        methods=list(METHODS),
        sample_seeds=SAMPLE_SEEDS,
        search_seed_bases=SEED_BASES,
        measured_repeats=50,
        synthetic_repeats=100,
        synthetic_samples=5,
        source_hashes={str(p): digest(p) for p in sorted(code_files)},
        input_hashes={str(p): digest(p) for p in sorted(input_files)},
        archive_hashes={str(p): digest(p) for p in sorted(SOURCE.rglob("*")) if p.is_file()},
    )
    save_json(manifest_path, manifest)
    print(f"Frozen manifest {digest(manifest_path)}", flush=True)


def run_chunk(name, label, sample, start):
    folder = OUT / name
    trials_dir = folder / "trials"
    trials_dir.mkdir(exist_ok=True)
    path = trials_dir / f"{label}_{sample}_{start:03d}.csv"
    if path.exists():
        return
    metadata = json.loads((folder / "metadata.json").read_text())
    frame = pd.read_pickle(folder / "frame.pkl")
    design = np.load(folder / "design.npz")
    selected = np.lexsort(design["codes"].T[::-1])
    codes = design["codes"][selected]
    values = np.load(folder / f"fitness_{label}_{sample}.npy")[selected]
    depth = frame.depth.to_numpy()[selected]
    known, unknown = np.flatnonzero(depth <= 1), np.flatnonzero(depth > 1)
    start_value = float(values[known].max())
    optimum, scale = float(values.max()), max(float(np.ptp(values)), 1e-12)
    space = SearchSpace(
        codes,
        tuple(map(int, design["shape"])),
        selected,
        int(known[np.argmax(values[known])]),
        {tuple(c): i for i, c in enumerate(codes)},
    )
    features = make_features(space)
    source = metadata["panel"] if metadata["panel"] == "development" else "evaluation"
    if label != "measured":
        source = "synthetic"
    started = time.monotonic()
    rows = []
    for trial in range(start, start + 10):
        repeat = 20 * sample + trial if label == "mutascape" else trial
        seed = SEED_BASES[source] + 137 * repeat
        for budget in metadata["budgets"]:
            for method, function in METHODS.items():
                oracle = DMSOracle(values, budget, known)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    if method == "random":
                        rng = np.random.default_rng(seed)
                        oracle.query(rng.choice(unknown, size=budget, replace=False))
                    else:
                        function(space, oracle, features, seed)
                assert 0 <= oracle.spent <= budget
                best = oracle.best()
                rows.append(
                    dict(
                        protein=name,
                        panel=metadata["panel"],
                        model=label,
                        sample=sample,
                        trial=trial,
                        search_seed=seed,
                        method=method,
                        budget=budget,
                        spent=oracle.spent,
                        regret=(optimum - best) / scale,
                        best_fitness=best,
                        start_fitness=start_value,
                        optimum=optimum,
                        fitness_range=scale,
                        remaining_regret=(optimum - best) / (optimum - start_value)
                        if optimum > start_value
                        else np.nan,
                        warnings=len(caught),
                    )
                )
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)
    print(
        f"Finished {name} {label} sample {sample} repeats {start + 1}-{start + 10} "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )


def run(jobs, proteins):
    manifest = json.loads((OUT / "manifest.json").read_text())
    for section in ("source_hashes", "input_hashes"):
        for path, expected in manifest[section].items():
            if digest(path) != expected:
                raise RuntimeError(f"Frozen file changed {path}")
    tasks = []
    for record in json.loads((OUT / "inventory.json").read_text()):
        name = record["protein"]
        if not record["budgets"] or (proteins and name not in proteins):
            continue
        tasks += [(name, "measured", 0, start) for start in range(0, 50, 10)]
        if record["panel"] == "development":
            continue
        tasks += [
            (name, label, 0, start)
            for label in ("additive", "zero_pair")
            for start in range(0, 100, 10)
        ]
        tasks += [(name, "mutascape", sample, start) for sample in range(5) for start in (0, 10)]
    # Interleave proteins and models so no one target occupies all workers.
    tasks.sort(key=lambda task: (task[3], task[2], task[1], task[0]))
    print(f"Scheduled {len(tasks)} chunks", flush=True)
    Parallel(n_jobs=jobs)(delayed(run_chunk)(*task) for task in tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--proteins", nargs="*")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "freeze":
        freeze()
    else:
        run(args.jobs, args.proteins)


if __name__ == "__main__":
    main()
