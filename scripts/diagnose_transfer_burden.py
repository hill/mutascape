"""Test a protein-balanced burden prior without fitting CreiLOV combinations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from assess_creilov_pilot import fit_pilot, pair_design, pair_noise
from run_creilov_combinatorial_validation import parse_mutant
from scipy.stats import wasserstein_distance

from mutascape.benchmarks import roots
from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features


def main():
    root = roots.data_root()
    datasets = [
        *load_ssmula_landscapes(root / "ssmula", strict=False),
        *load_combingym_landscapes(root / "combingym", strict=False),
    ]
    fits = []
    for dataset in datasets:
        if dataset.protein == "CreiLOV":
            continue
        frame = dataset.variants.copy()
        wt = frame.loc[frame.depth == 0, "fitness"]
        if wt.empty:
            continue
        frame["fitness"] -= wt.iloc[0]
        obs = frame.loc[frame.depth <= 1]
        effects = {
            tuple(m.items())[0]: f
            for m, f in obs.loc[obs.depth == 1, ["mutations", "fitness"]].itertuples(
                index=False, name=None
            )
        }
        scale = np.std(list(effects.values()))
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        frame = frame.loc[
            (frame.depth > 1) & frame.mutations.map(lambda m: all(k in effects for k in m.items()))
        ]
        if len(frame) < 48:
            continue
        pilot = frame.sample(n=min(len(frame), 5000), random_state=42)
        model, noise = fit_pilot(pilot, effects, burden=True)
        coefficients = model[-1].coef_ / model[0].scale_
        fits.append(
            dict(
                protein=dataset.protein,
                name=dataset.name,
                burden=coefficients[0] / scale,
                attenuation=coefficients[1],
                noise=noise / scale,
            )
        )
    fits = pd.DataFrame(fits)
    coefficients = fits.groupby("protein")[["burden", "attenuation", "noise"]].median().median()
    print(fits.to_string(index=False), flush=True)
    print(coefficients, flush=True)
    reference = Path("results/creilov_c2st")
    metadata = json.loads((reference / "metadata.json").read_text())
    raw = pd.read_excel(metadata["raw_path"], nrows=100)
    raw["mutations"] = [{p: aa for p, _, aa in parse_mutant(v)} for v in raw["Unnamed: 0"]]
    raw["depth"] = raw.mutations.map(len)
    raw["fitness"] = raw.mean_log - float(raw.loc[raw.depth == 0, "mean_log"].iloc[0])
    obs = raw.loc[raw.depth <= 1, ["mutations", "depth", "fitness"]]
    effects = {
        tuple(m.items())[0]: f for m, d, f in obs.itertuples(index=False, name=None) if d == 1
    }
    scale = np.std(list(effects.values()))
    sequence = "".join(Path("data/creilov/creilov_wt.fasta").read_text().splitlines()[1:])
    rows = []
    for seed in (42, 43, 44):
        sample = pd.read_csv(reference / f"samples_{seed}.csv")
        sample["mutations"] = [{p: aa for p, _, aa in parse_mutant(v)} for v in sample.variant]
        sample["fitness"] = sample.measured
        additive, design = pair_design(sample.mutations, effects, burden=True)
        real = completion_fidelity_features(obs, sample).to_numpy()
        for draw in (20260826, 20260827, 20260828):
            values = additive + design @ [coefficients.burden * scale, coefficients.attenuation]
            values += coefficients.noise * scale * pair_noise(sample.mutations, sequence, draw)
            synth = completion_fidelity_features(obs, sample, values).to_numpy()
            groups = np.arange(len(sample))
            for classifier in ("logistic", "histogram"):
                result = classifier_two_sample_test(
                    real,
                    synth,
                    real_groups=groups,
                    synthetic_groups=groups,
                    classifier=classifier,
                    seed=seed,
                    n_permutations=0,
                )
                row = dict(
                    seed=seed,
                    draw=draw,
                    classifier=classifier,
                    auc=result.auc,
                    raw_w1=wasserstein_distance(sample.fitness, values),
                )
                print(row, flush=True)
                rows.append(row)
    output = Path(".mutascape/experiments/transfer_burden")
    output.mkdir(parents=True, exist_ok=True)
    fits.to_csv(output / "donors.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
