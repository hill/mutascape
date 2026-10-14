"""Repeat the leading candidate and a matched-variance control with fresh seeds."""

import argparse
import json
import shutil
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts import compare_external_generators as base
from scripts.check_generator_improvements import OUT as CHECK
from scripts.external_generator_models import nk_values, rmf_values
from scripts.generator_improvement_models import candidate_values
from scripts.run_generator_improvements import REFERENCE

OUT = Path("results/generator_improvements_replication_20261006")
MODELS = (
    "mutascape",
    "mutascape_additive",
    "mutascape_linear",
    "mutascape_pool",
    "effect_pairs",
    "uniform_matched",
    "slip4",
    "nk1",
    "rmf",
)
STOCHASTIC = tuple(m for m in MODELS if m not in ("mutascape_additive", "slip4"))


def matched_strength(singles):
    """Match mean squared pair scales across allowed substitution pairs."""
    wt = float(singles.loc[singles.mutations.map(len) == 0, "fitness"].iloc[0])
    effects = [
        (next(iter(v)), float(f - wt)) for v, f in singles.itertuples(index=False, name=None) if v
    ]
    products = [abs(a * b) for (p, a), (q, b) in combinations(effects, 2) if p != q]
    std = float(np.std([v for _, v in effects]))
    return float(np.sqrt(np.mean(products)) / std) if std > 0 else 0.0


def prepare():
    if (OUT / "manifest.json").exists():
        return verify()
    paths = [
        Path(__file__),
        Path("scripts/generator_improvement_models.py"),
        Path("docs/generator_improvement_replication_2026-10-06.md"),
    ]
    inventory, records = [], []
    for source_root in (REFERENCE, CHECK):
        manifest = json.loads((source_root / "manifest.json").read_text())
        paths.append(source_root / "manifest.json")
        for item in manifest["inventory"]:
            protein, domain = item["protein"], item["domain"]
            source = source_root / protein / str(domain)
            folder = OUT / protein / str(domain)
            folder.mkdir(parents=True, exist_ok=True)
            design = np.load(source / "design.npz")
            frame = pd.read_pickle(source / "frame.pkl")
            sequence = json.loads((source_root / protein / "mapping.json").read_text())["sequence"]
            known = design["known"]
            singles = frame.iloc[known][["mutations", "fitness"]]
            strength = matched_strength(singles)
            for name in ["design.npz", "measured_0.npy", "slip4_0.npy"]:
                shutil.copyfile(source / name, folder / name)
                paths.append(source / name)
            paths.extend([source / "frame.pkl", source_root / protein / "mapping.json"])
            for model in MODELS:
                for sample in range(10 if model in STOCHASTIC else 1):
                    seed = 2026103000 + sample
                    if model in base.SETTINGS:
                        mode, amount = base.SETTINGS[model]
                        fitted = base.DMSLandscape.from_singles(
                            sequence, singles, mode=mode, strength=amount, seed=seed
                        )
                        values = fitted.evaluate_batch(frame.mutations.tolist())
                    elif model == "uniform_matched":
                        fitted = base.DMSLandscape.from_singles(
                            sequence, singles, mode="linear", strength=strength, seed=seed
                        )
                        values = fitted.evaluate_batch(frame.mutations.tolist())
                    elif model == "effect_pairs":
                        values, _ = candidate_values(
                            model,
                            design["codes"],
                            singles,
                            frame.mutations.tolist(),
                            sequence,
                            seed,
                        )
                    elif model == "nk1":
                        values = nk_values(design["codes"], design["shape"], 1, seed)
                    elif model == "rmf":
                        values = rmf_values(design["codes"], design["shape"], seed)
                    elif model == "slip4":
                        values = np.load(source / "slip4_0.npy")
                    else:
                        raise ValueError(model)
                    if model not in ["slip4", "nk1", "rmf"]:
                        np.testing.assert_array_equal(values[known], singles.fitness.to_numpy())
                    assert np.isfinite(values).all()
                    target = folder / f"{model}_{sample}.npy"
                    np.save(target, values)
                    paths.append(target)
            records.append(dict(protein=protein, domain=domain, uniform_matched_strength=strength))
            inventory.append(item)
    pd.DataFrame(records).to_csv(OUT / "fitting.csv", index=False)
    paths.extend(Path("src/mutascape").rglob("*.py"))
    base.save_json(
        OUT / "manifest.json",
        dict(
            models=MODELS,
            stochastic=STOCHASTIC,
            samples=10,
            repeats=6,
            inventory=inventory,
            source_sha256={str(p): base.digest(p) for p in paths},
        ),
    )


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        if base.digest(path) != expected:
            raise ValueError(f"Frozen replication input changed at {path}")
    return manifest


def run(protein, domain, model, block):
    folder = OUT / protein / str(domain)
    destination = folder / f"trials_{model}_{block}.csv"
    if destination.exists():
        return
    sample = block if model in STOCHASTIC else 0
    values = np.load(folder / f"{model}_{sample}.npy")
    design = np.load(folder / "design.npz")
    known = design["known"]
    space = base.space_from(design["codes"], design["shape"], known, values)
    features = base.make_features(space)
    spread = float(np.ptp(values))
    opportunity = float(values.max() - values[known].max())
    rows, queries = [], []
    start = time.monotonic()
    for budget in base.BUDGETS:
        for trial in range(6):
            seed = (2026104000 if model == "measured" else 2026105000) + 137 * (block * 6 + trial)
            for method, function in base.METHODS.items():
                oracle = base.DMSOracle(values, budget, known)
                with warnings.catch_warnings(record=True) as caught:
                    function(space, oracle, features, seed)
                assert oracle.spent == budget
                assert len(np.unique(oracle.rows)) == len(known) + budget
                gap = float(values.max() - oracle.best())
                queries.append(
                    np.pad(oracle.rows[len(known) :], (0, 96 - budget), constant_values=-1)
                )
                rows.append(
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
                        best=oracle.best(),
                        spent=oracle.spent,
                        fallback_queries=oracle.fallback_queries,
                        warnings=len(caught),
                        warning_messages=" | ".join(sorted({str(w.message) for w in caught})),
                    )
                )
    np.savez_compressed(destination.with_suffix(".npz"), queries=np.array(queries))
    pd.DataFrame(rows).to_csv(destination.with_suffix(".tmp"), index=False)
    destination.with_suffix(".tmp").replace(destination)
    print(
        f"Completed {protein} {domain} {model} {block}, {time.monotonic() - start:.1f}s", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        manifest = verify()
        Parallel(n_jobs=args.jobs)(
            delayed(run)(d["protein"], d["domain"], m, b)
            for d in manifest["inventory"]
            for m in ["measured", *MODELS]
            for b in range(10)
        )


if __name__ == "__main__":
    main()
