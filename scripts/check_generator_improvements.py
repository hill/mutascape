"""Fit and evaluate all candidate mechanisms on HIV-1 protease."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts import compare_external_generators as base
from scripts import fit_slip_comparators as fitter
from scripts.external_generator_models import nk_values, rmf_values, slip_values
from scripts.generator_improvement_models import MODELS, STOCHASTIC, candidate_values

OUT = Path("results/generator_improvements_check_20261006")
ALL_MODELS = (*base.MODELS, *MODELS)
ALL_STOCHASTIC = (*base.STOCHASTIC, *STOCHASTIC)
PROTEIN = "HIV-1"


def fit():
    fitter.MSAS[PROTEIN] = "HIV-1_b0.5.a2m"
    sys.argv = [__file__, "--protein", PROTEIN]
    fitter.main()


def prepare():
    if (OUT / "manifest.json").exists():
        return verify()
    fitter.MSAS[PROTEIN] = "HIV-1_b0.5.a2m"
    source = Path("results/heldout_panel_20260923") / PROTEIN
    frame = pd.read_pickle(source / "frame.pkl")
    sequence = json.loads((source / "metadata.json").read_text())["sequence"]
    msa_path, _, covered, query = fitter.read_msa(PROTEIN)
    assert query == sequence
    positions = sorted({p for v in frame.mutations for p in v})
    assert set(positions) <= set(covered)
    alphabets = {
        p: [sequence[p - 1], *sorted({v[p] for v in frame.mutations if p in v})] for p in positions
    }
    sizes = tuple(len(alphabets[p]) for p in positions)
    codes = np.asarray(
        [
            [alphabets[p].index(v.get(p, sequence[p - 1])) for p in positions]
            for v in frame.mutations
        ],
        dtype=np.int16,
    )
    known = np.flatnonzero(frame.depth.to_numpy() <= 1)
    assert len(known) == 1 + sum(s - 1 for s in sizes)
    values = frame.fitness.to_numpy()
    topology = base.library_connectivity(base.space_from(codes, sizes, known, values))
    assert topology["connected_components"] == 1
    folder = OUT / PROTEIN / "0"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(folder / "frame.pkl")
    np.savez(folder / "design.npz", codes=codes, shape=sizes, known=known)
    np.save(folder / "measured_0.npy", values)
    singles = frame.iloc[known][["mutations", "fitness"]]
    records, cache = [], {}
    for model in ALL_MODELS:
        for sample in range(5 if model in ALL_STOCHASTIC else 1):
            seed = base.LANDSCAPE_SEEDS[sample]
            metadata = {}
            if model in base.SETTINGS:
                mode, strength = base.SETTINGS[model]
                fitted = base.DMSLandscape.from_singles(
                    sequence, singles, mode=mode, strength=strength, seed=seed
                )
                v = fitted.evaluate_batch(frame.mutations.tolist())
            elif model.startswith("slip"):
                horizon = None if model == "slip_raw" else int(model[4:])
                v, metadata = slip_values(
                    PROTEIN, codes, positions, alphabets, {p: p for p in positions}, horizon
                )
                cache[model] = v
            elif model.startswith("nk"):
                v = nk_values(codes, sizes, int(model[2:]), seed)
            elif model == "rmf":
                v = rmf_values(codes, sizes, seed)
            elif model == "hoc":
                v = np.random.default_rng(seed).normal(size=int(np.prod(sizes)))[
                    np.ravel_multi_index(codes.T, sizes)
                ]
            else:
                prior_name = {
                    "msa_regression": "slip_raw",
                    "msa_slip4": "slip4",
                    "msa_slip8": "slip8",
                }.get(model)
                v, metadata = candidate_values(
                    model,
                    codes,
                    singles,
                    frame.mutations.tolist(),
                    sequence,
                    seed,
                    cache.get(prior_name),
                )
            if model in base.SETTINGS or model in MODELS:
                np.testing.assert_array_equal(v[known], values[known])
            assert np.isfinite(v).all()
            np.save(folder / f"{model}_{sample}.npy", v)
            records.append(
                dict(model=model, sample=sample, **metadata, **base.fitness_headroom(v, known))
            )
    pd.DataFrame(records).to_csv(OUT / "fitting.csv", index=False)
    base.save_json(
        OUT / PROTEIN / "mapping.json",
        dict(sequence=sequence, positions=positions, alphabets=alphabets),
    )
    paths = [
        Path(__file__),
        Path("scripts/generator_improvement_models.py"),
        Path("docs/generator_improvement_check_2026-10-06.md"),
        source / "frame.pkl",
        source / "metadata.json",
        msa_path,
        fitter.CACHE / PROTEIN / "mogwai_1000.npz",
        fitter.CACHE / PROTEIN / "fit.json",
        folder / "design.npz",
        *folder.glob("*.npy"),
    ]
    base.save_json(
        OUT / "manifest.json",
        dict(
            models=ALL_MODELS,
            stochastic=ALL_STOCHASTIC,
            inventory=[dict(protein=PROTEIN, domain=0, known=len(known), **topology)],
            source_sha256={str(p): base.digest(p) for p in paths},
        ),
    )


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        if base.digest(path) != expected:
            raise ValueError(f"Frozen check input changed at {path}")
    return manifest


def run(model, block):
    base.OUT = OUT
    base.STOCHASTIC = ALL_STOCHASTIC
    base.run(PROTEIN, 0, model, block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "prepare", "run"])
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "fit":
        fit()
    elif args.stage == "prepare":
        prepare()
    else:
        verify()
        Parallel(n_jobs=args.jobs)(
            delayed(run)(m, b) for m in ["measured", *ALL_MODELS] for b in range(5)
        )


if __name__ == "__main__":
    main()
