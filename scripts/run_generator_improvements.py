"""Run the frozen candidate experiments against existing reference campaigns."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts import compare_external_generators as base
from scripts.generator_improvement_models import MODELS, STOCHASTIC, candidate_values

OUT = Path("results/generator_improvements_20261006")
REFERENCE = Path("results/external_generators_20261006")


def prepare():
    if (OUT / "manifest.json").exists():
        return verify()
    reference = base.verify()
    paths = [
        Path(__file__),
        Path("scripts/generator_improvement_models.py"),
        Path("docs/generator_improvement_protocol_2026-10-06.md"),
    ]
    records = []
    for item in reference["inventory"]:
        protein, domain = item["protein"], item["domain"]
        source = REFERENCE / protein / str(domain)
        destination = OUT / protein / str(domain)
        destination.mkdir(parents=True, exist_ok=True)
        design = np.load(source / "design.npz")
        known = design["known"]
        frame = pd.read_pickle(source / "frame.pkl")
        singles = frame.iloc[known][["mutations", "fitness"]].copy()
        sequence = json.loads((REFERENCE / protein / "mapping.json").read_text())["sequence"]
        shutil.copyfile(source / "design.npz", destination / "design.npz")
        paths.extend(
            [source / "design.npz", source / "frame.pkl", REFERENCE / protein / "mapping.json"]
        )
        for model in MODELS:
            prior_name = {
                "msa_regression": "slip_raw",
                "msa_slip4": "slip4",
                "msa_slip8": "slip8",
            }.get(model)
            prior = np.load(source / f"{prior_name}_0.npy") if prior_name else None
            if prior_name:
                paths.append(source / f"{prior_name}_0.npy")
            for sample in range(5 if model in STOCHASTIC else 1):
                values, metadata = candidate_values(
                    model,
                    design["codes"],
                    singles,
                    frame.mutations.tolist(),
                    sequence,
                    base.LANDSCAPE_SEEDS[sample],
                    prior,
                )
                np.testing.assert_array_equal(values[known], singles.fitness.to_numpy())
                assert np.isfinite(values).all()
                target = destination / f"{model}_{sample}.npy"
                np.save(target, values)
                paths.append(target)
                records.append(
                    dict(protein=protein, domain=domain, model=model, sample=sample, **metadata)
                )
    pd.DataFrame(records).to_csv(OUT / "fitting.csv", index=False)
    base.save_json(
        OUT / "manifest.json",
        dict(
            reference_manifest_sha256=base.digest(REFERENCE / "manifest.json"),
            models=MODELS,
            stochastic=STOCHASTIC,
            inventory=reference["inventory"],
            budgets=base.BUDGETS,
            methods=list(base.METHODS),
            samples=5,
            repeats=6,
            source_sha256={str(p): base.digest(p) for p in paths},
        ),
    )


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        if base.digest(path) != expected:
            raise ValueError(f"Frozen candidate input changed at {path}")
    if base.digest(REFERENCE / "manifest.json") != manifest["reference_manifest_sha256"]:
        raise ValueError("The reference manifest changed")
    return manifest


def run(protein, domain, model, block):
    base.OUT = OUT
    base.STOCHASTIC = STOCHASTIC
    base.run(protein, domain, model, block)


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
            for m in MODELS
            for b in range(5)
        )


if __name__ == "__main__":
    main()
