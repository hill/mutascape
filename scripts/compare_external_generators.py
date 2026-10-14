"""Run matched search campaigns on Mutascape, SLIP and generic landscapes."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks.diagnostics import fitness_headroom, library_connectivity
from mutascape.benchmarks.methods import METHODS, SEARCH_PROTOCOL_VERSION, make_features
from mutascape.model import DMSLandscape
from scripts.benchmark_efficacy import connected_rows, space_from
from scripts.external_generator_models import nk_values, rmf_values, slip_values
from scripts.fit_slip_comparators import CACHE, MSAS, read_msa
from scripts.singles_warm_start import DMSOracle

OUT = Path("results/external_generators_20261006")
SOURCE = Path("results/search_selection_20260924")
PROTEINS = tuple(MSAS)
SETTINGS = {
    "mutascape": ("sigmoid", 0.3),
    "mutascape_zero_pair": ("sigmoid", 0.0),
    "mutascape_additive": ("linear", 0.0),
    "mutascape_linear": ("linear", 1.0),
    "mutascape_pool": ("pool", 0.3),
}
STOCHASTIC = ("mutascape", "mutascape_linear", "mutascape_pool", "nk0", "nk1", "nk3", "rmf", "hoc")
MODELS = (*SETTINGS, "slip_raw", "slip4", "slip8", "nk0", "nk1", "nk3", "rmf", "hoc")
PRIMARY = ("mutascape", "mutascape_zero_pair", "mutascape_additive", "slip4", "nk1", "rmf")
BUDGETS = (48, 96)
LANDSCAPE_SEEDS = tuple(2026100640 + i for i in range(5))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item()) + "\n")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        return verify()
    inventory, source_paths = [], []
    for protein in PROTEINS:
        source = SOURCE / protein
        frame = pd.read_pickle(source / "frame.pkl")
        metadata = json.loads((source / "metadata.json").read_text())
        msa_path, _, covered, query = read_msa(protein)
        positions = sorted({p for mutations in frame.mutations for p in mutations})
        full_positions = (
            dict(zip(positions, [39, 40, 41, 54]))
            if protein == "GB1"
            else {p: p for p in positions}
        )
        for p in positions:
            assert metadata["sequence"][p - 1] == query[full_positions[p] - 1]
        excluded = [p for p in positions if full_positions[p] not in covered]
        eligible = frame.mutations.map(lambda m: not any(p in m for p in excluded))
        frame = frame.loc[eligible].reset_index(drop=True)
        positions = [p for p in positions if p not in excluded]
        alphabets = {}
        for p in positions:
            wt = metadata["sequence"][p - 1]
            residues = sorted({m[p] for m in frame.mutations if p in m} - {wt})
            alphabets[p] = [wt, *residues]
        codes = np.array(
            [
                [alphabets[p].index(m.get(p, alphabets[p][0])) for p in positions]
                for m in frame.mutations
            ],
            dtype=np.int16,
        )
        sizes = tuple(len(alphabets[p]) for p in positions)
        known = np.flatnonzero(frame.depth.to_numpy() <= 1)
        assert len(known) == 1 + sum(s - 1 for s in sizes)
        assert len(np.unique(codes, axis=0)) == len(codes)
        assert len(frame) >= 2048
        protein_folder = OUT / protein
        protein_folder.mkdir(exist_ok=True)
        values, tuning = {}, {}
        for model in MODELS:
            print("Generating", protein, model, flush=True)
            count = 5 if model in STOCHASTIC else 1
            for sample in range(count):
                seed = LANDSCAPE_SEEDS[sample]
                if model in SETTINGS:
                    mode, strength = SETTINGS[model]
                    fitted = DMSLandscape.from_singles(
                        metadata["sequence"],
                        frame.iloc[known],
                        mode=mode,
                        strength=strength,
                        seed=seed,
                    )
                    v = fitted.evaluate_batch(frame.mutations.tolist())
                    np.testing.assert_array_equal(v[known], frame.fitness.to_numpy()[known])
                elif model.startswith("slip"):
                    horizon = None if model == "slip_raw" else int(model[4:])
                    v, report = slip_values(
                        protein, codes, positions, alphabets, full_positions, horizon
                    )
                    tuning[model] = report
                elif model.startswith("nk"):
                    v = nk_values(codes, sizes, int(model[2:]), seed)
                elif model == "rmf":
                    v = rmf_values(codes, sizes, seed)
                else:
                    v = np.random.default_rng(seed).normal(size=int(np.prod(sizes)))[
                        np.ravel_multi_index(codes.T, sizes)
                    ]
                assert np.isfinite(v).all()
                values[model, sample] = v
        save_json(protein_folder / "slip_tuning.json", tuning)
        target_size = min(4096, len(frame))
        count = 1 if len(frame) == target_size else 3
        domain_sets = []
        for domain in range(count):
            rows = connected_rows(codes, sizes, known, target_size, 2026100600 + domain)
            domain_sets.append(set(rows.tolist()))
            selected = frame.iloc[rows].reset_index(drop=True)
            selected_codes = codes[rows]
            selected_known = np.flatnonzero(selected.depth.to_numpy() <= 1)
            measured = selected.fitness.to_numpy()
            folder = protein_folder / str(domain)
            folder.mkdir(exist_ok=True)
            selected.to_pickle(folder / "frame.pkl")
            np.savez(
                folder / "design.npz",
                codes=selected_codes,
                shape=sizes,
                known=selected_known,
                source_rows=rows,
            )
            np.save(folder / "measured_0.npy", measured)
            diagnostics = [
                dict(model="measured", sample=0, **fitness_headroom(measured, selected_known))
            ]
            for (model, sample), v in values.items():
                np.save(folder / f"{model}_{sample}.npy", v[rows])
                diagnostics.append(
                    dict(model=model, sample=sample, **fitness_headroom(v[rows], selected_known))
                )
            pd.DataFrame(diagnostics).to_csv(folder / "headroom.csv", index=False)
            topology = library_connectivity(
                space_from(selected_codes, sizes, selected_known, measured)
            )
            assert topology["connected_components"] == 1
            inventory.append(
                dict(protein=protein, domain=domain, **topology, known=len(selected_known))
            )
            source_paths.extend(folder.glob("*.npy"))
            source_paths.extend([folder / "design.npz", folder / "frame.pkl"])
        info = dict(
            protein=protein,
            eligible_genotypes=len(frame),
            positions=positions,
            excluded_positions=excluded,
            full_positions=full_positions,
            alphabets=alphabets,
            sequence=metadata["sequence"],
            shared_fraction=[[len(a & b) / target_size for b in domain_sets] for a in domain_sets],
        )
        save_json(protein_folder / "mapping.json", info)
        source_paths.extend(
            [
                source / "frame.pkl",
                source / "metadata.json",
                msa_path,
                CACHE / protein / "mogwai_1000.npz",
                CACHE / protein / "fit.json",
            ]
        )
        print(
            "Prepared",
            protein,
            len(frame),
            "eligible genotypes,",
            count,
            "search spaces",
            flush=True,
        )
    source_paths.extend(Path("src/mutascape").rglob("*.py"))
    source_paths.extend(
        [
            Path(__file__),
            Path("scripts/external_generator_models.py"),
            Path("scripts/fit_slip_comparators.py"),
            Path("scripts/benchmark_efficacy.py"),
            Path("scripts/singles_warm_start.py"),
            Path("docs/external_generators_protocol_2026-10-06.md"),
        ]
    )
    versions = {
        p: subprocess.check_output(
            ["git", "-C", str(CACHE / p), "rev-parse", "HEAD"], text=True
        ).strip()
        for p in ("slip", "mogwai")
    }
    save_json(
        OUT / "manifest.json",
        dict(
            protocol=SEARCH_PROTOCOL_VERSION,
            proteins=PROTEINS,
            models=MODELS,
            primary=PRIMARY,
            budgets=BUDGETS,
            samples=5,
            repeats_per_sample=6,
            fixed_repeats=30,
            landscape_seeds=LANDSCAPE_SEEDS,
            methods=list(METHODS),
            settings=SETTINGS,
            inventory=inventory,
            external_commits=versions,
            python=platform.python_version(),
            source_sha256={str(p): digest(p) for p in source_paths},
        ),
    )


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed at {path}")
    return manifest


def run(protein, domain, model, block, repeats=6, suffix=""):
    folder = OUT / protein / str(domain)
    destination = folder / f"trials_{model}_{block}{suffix}.csv"
    if destination.exists():
        return
    sample = block if model in STOCHASTIC else 0
    design = np.load(folder / "design.npz")
    values = np.load(folder / f"{model}_{sample}.npy")
    known = design["known"]
    space = space_from(design["codes"], design["shape"], known, values)
    features = make_features(space)
    spread = float(np.ptp(values))
    opportunity = float(values.max() - values[known].max())
    records, query_rows, curves = [], [], []
    start = time.monotonic()
    for budget in BUDGETS:
        for trial in range(repeats):
            seed = (2026101000 if model == "measured" else 2026102000) + 137 * (block * 6 + trial)
            for method, function in METHODS.items():
                oracle = DMSOracle(values, budget, known)
                t = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    function(space, oracle, features, seed)
                elapsed = time.monotonic() - t
                assert oracle.spent == budget
                assert len(np.unique(oracle.rows)) == len(known) + budget
                assert set(oracle.rows[: len(known)]) == set(known)
                best = oracle.best()
                gap = float(values.max() - best)
                rows = oracle.rows[len(known) :]
                trace = np.maximum.accumulate(np.r_[values[known].max(), values[rows]])
                query_rows.append(np.pad(rows, (0, 96 - budget), constant_values=-1))
                curves.append(
                    np.pad(
                        (values.max() - trace) / spread, (0, 96 - budget), constant_values=np.nan
                    )
                )
                records.append(
                    dict(
                        protein=protein,
                        domain=domain,
                        model=model,
                        sample=sample,
                        block=block,
                        trial=trial,
                        budget=budget,
                        method=method,
                        regret=gap / spread,
                        remaining_regret=gap / opportunity if opportunity > 0 else np.nan,
                        best=best,
                        percentile=float((values <= best).mean()),
                        spent=oracle.spent,
                        fallback_queries=oracle.fallback_queries,
                        warnings=len(caught),
                        warning_messages=" | ".join(sorted({str(w.message) for w in caught})),
                        elapsed_seconds=elapsed,
                    )
                )
    np.savez_compressed(
        destination.with_suffix(".npz"),
        queries=np.array(query_rows),
        regret_curves=np.array(curves),
    )
    temporary = destination.with_suffix(".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(destination)
    print(
        f"Completed {protein} space {domain}, {model}, block {block}, "
        f"{time.monotonic() - start:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "run"))
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "smoke":
        verify()
        run("GB1", 0, "measured", 0, repeats=1, suffix="_smoke")
    else:
        manifest = verify()
        models = args.models or ["measured", *MODELS]
        Parallel(n_jobs=args.jobs)(
            delayed(run)(d["protein"], d["domain"], m, b)
            for d in manifest["inventory"]
            for m in models
            for b in range(5)
        )


if __name__ == "__main__":
    main()
