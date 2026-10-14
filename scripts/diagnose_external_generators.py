"""Compare matched local spectra, graph properties and assay-scale distributions."""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance

from mutascape.benchmarks.evaluation import classifier_two_sample_test
from scripts.compare_external_generators import (
    MODELS,
    OUT,
    SETTINGS,
    STOCHASTIC,
    space_from,
    verify,
)


def sample_cubes(codes, sizes, order, seed, count=256):
    """Sample complete subcubes with WT as one state at each variable site."""
    lookup = {tuple(c): i for i, c in enumerate(codes)}
    rng = np.random.default_rng(seed)
    output, seen = [], set()
    for _ in range(200_000):
        base = codes[rng.integers(len(codes))].copy()
        sites = np.sort(rng.choice(len(sizes), order, replace=False))
        alternatives = [int(base[s]) if base[s] else int(rng.integers(1, sizes[s])) for s in sites]
        corners = []
        for mask in range(2**order):
            code = base.copy()
            for i, site in enumerate(sites):
                code[site] = alternatives[i] if mask & (1 << i) else 0
            row = lookup.get(tuple(code))
            if row is None:
                break
            corners.append(row)
        if len(corners) != 2**order:
            continue
        key = tuple(sorted(corners))
        if key in seen:
            continue
        seen.add(key)
        output.append(corners)
        if len(output) == count:
            break
    if len(output) < count:
        raise ValueError(f"Only {len(output)} complete order-{order} cubes found")
    return np.asarray(output)


def walsh_spectrum(values, cubes):
    parity = np.array([[(-1) ** int(i & j).bit_count() for j in range(8)] for i in range(8)])
    coefficients = values[cubes] @ parity / 8
    power = (coefficients[:, 1:] ** 2).sum(axis=0)
    order = np.array([i.bit_count() for i in range(1, 8)])
    return np.array([power[order == k].sum() for k in (1, 2, 3)]) / power.sum()


def diagnose(protein, domain):
    folder = OUT / protein / str(domain)
    design = np.load(folder / "design.npz")
    codes, sizes, known = design["codes"], design["shape"], design["known"]
    measured = np.load(folder / "measured_0.npy")
    space = space_from(codes, sizes, known, measured)
    pairs = np.array([(i, j) for i in range(len(codes)) for j in space.neighbours(i) if i < j])
    degree = np.bincount(pairs.ravel(), minlength=len(codes))
    cubes = sample_cubes(codes, sizes, 3, 2026100660 + domain)
    faces = sample_cubes(codes, sizes, 2, 2026100670 + domain, count=1024)
    np.savez(folder / "topology_design.npz", cubes=cubes, faces=faces, edges=pairs)
    reference = walsh_spectrum(measured, cubes)
    unknown = np.setdiff1d(np.arange(len(codes)), known)
    depths = (codes != 0).sum(axis=1)
    result = []
    for model in ("measured", *MODELS):
        for sample in range(5 if model in STOCHASTIC else 1):
            values = np.load(folder / f"{model}_{sample}.npy")
            spectrum = walsh_spectrum(values, cubes)
            first, second = pairs.T
            bad = np.bincount(
                np.r_[
                    first[values[first] <= values[second]], second[values[second] <= values[first]]
                ],
                minlength=len(values),
            )
            strict_peaks = (bad == 0) & (degree > 0)
            a, b, c, d = values[faces].T
            first_sign = (b - a) * (d - c) < 0
            second_sign = (c - a) * (d - b) < 0
            rec = dict(
                protein=protein,
                domain=domain,
                model=model,
                sample=sample,
                neighbour_correlation=float(
                    np.corrcoef(
                        np.r_[values[first], values[second]], np.r_[values[second], values[first]]
                    )[0, 1]
                ),
                local_optimum_fraction=float(strict_peaks.mean()),
                sign_epistasis=float((first_sign | second_sign).mean()),
                reciprocal_sign_epistasis=float((first_sign & second_sign).mean()),
                walsh1=spectrum[0],
                walsh2=spectrum[1],
                walsh3=spectrum[2],
                walsh_l1=float(abs(spectrum - reference).sum()),
            )
            if model in SETTINGS:
                real, synthetic = measured[unknown], values[unknown]
                rec["w1_real_sd"] = wasserstein_distance(real, synthetic) / real.std()
                # Paired genotypes stay in the same fold. No evaluation subsampling.
                tested = classifier_two_sample_test(
                    np.c_[real, depths[unknown]],
                    np.c_[synthetic, depths[unknown]],
                    real_groups=unknown,
                    synthetic_groups=unknown,
                    classifier="histogram",
                    n_permutations=0,
                    seed=2026100680,
                )
                rec["c2st_auc"] = tested.auc
            result.append(rec)
    pd.DataFrame(result).to_csv(folder / "properties.csv", index=False)
    print("Completed properties", protein, domain, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    manifest = verify()
    Parallel(n_jobs=args.jobs)(
        delayed(diagnose)(d["protein"], d["domain"]) for d in manifest["inventory"]
    )
    table = pd.concat([pd.read_csv(p) for p in OUT.glob("*/*/properties.csv")])
    table.to_csv(OUT / "properties.csv", index=False)


if __name__ == "__main__":
    main()
