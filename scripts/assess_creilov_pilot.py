"""Assess a frozen pilot-calibrated prototype without changing the public generator."""

import argparse
import hashlib
import json
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from run_creilov_combinatorial_validation import parse_mutant
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features, run_method
from mutascape.benchmarks.validity import pairwise_winner_transfer
from mutascape.calibrated import PilotCalibratedLandscape
from mutascape.completion import LandscapePrior, PosteriorLandscape, dms_pairwise_unit_coefficient
from mutascape.nonlinearity import GlobalEpistasisSigmoid
from mutascape.validation.fourier import binary_cube_spectral_metrics


def pair_design(mutations, effects, *, burden=False):
    """Return additive values and two corrections that vanish at WT and singles."""
    depth = np.asarray([len(m) for m in mutations])
    additive = np.asarray([sum(effects[k] for k in m.items()) for m in mutations])
    if burden:
        extra = np.maximum(depth - 1, 0)
        return additive, np.column_stack([extra, additive * extra / np.maximum(depth, 1)])
    return additive, np.column_stack([depth * (depth - 1) / 2, additive * (depth - 1)])


def pair_noise(mutations, sequence, seed):
    """Evaluate shared pair coefficients with unit marginal variance."""
    keys = sorted({key for m in mutations for key in m.items()})
    design = np.asarray([[m.get(p) == aa for p, aa in keys] for m in mutations])
    values = np.zeros(len(mutations))
    for i, j in combinations(range(len(keys)), 2):
        if keys[i][0] != keys[j][0]:
            coefficient = dms_pairwise_unit_coefficient(keys[i], keys[j], sequence, seed) / 2
            values += coefficient * (design[:, i] & design[:, j])
    return values


def fit_pilot(pilot, effects, *, burden=False):
    """Fit two mean terms and one residual scale from the pilot alone."""
    additive, design = pair_design(pilot.mutations.tolist(), effects, burden=burden)
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1, fit_intercept=False))
    model.fit(design, pilot.fitness.to_numpy() - additive)
    residual = pilot.fitness.to_numpy() - additive - model.predict(design)
    pairs = pilot.depth.to_numpy() * (pilot.depth.to_numpy() - 1) / 2
    scale = float(np.sqrt(np.sum(residual**2) / np.sum(pairs)))
    return model, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", action="store_true")
    parser.add_argument("--burden", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    reference = Path("results/creilov_c2st")
    metadata = json.loads((reference / "metadata.json").read_text())
    source = Path(metadata["raw_path"])
    raw = pd.read_excel(source).rename(columns={"Unnamed: 0": "variant"})
    raw["mutations"] = [{p: aa for p, _, aa in parse_mutant(v)} for v in raw.variant]
    raw["depth"] = raw.mutations.map(len)
    raw["fitness"] = raw.mean_log - float(raw.loc[raw.depth == 0, "mean_log"].iloc[0])
    raw = raw.loc[np.isfinite(raw.fitness)].reset_index(drop=True)
    singles = raw.loc[raw.depth <= 1, ["mutations", "depth", "fitness"]]
    effects = {
        tuple(m.items())[0]: f for m, d, f in singles.itertuples(index=False, name=None) if d == 1
    }
    sequence = "".join(Path("data/creilov/creilov_wt.fasta").read_text().splitlines()[1:])
    dms = pd.read_csv("data/creilov/creilov.csv")
    original = pd.DataFrame(
        [{"mutations": {}, "depth": 0, "fitness": 0.0}]
        + [
            {"mutations": {r.position: r.mutant}, "depth": 1, "fitness": r.fitness}
            for r in dms.itertuples()
        ]
    )
    original_keys = {tuple(m.items()) for m in original.mutations}
    original = pd.concat(
        [original, singles.loc[[tuple(m.items()) not in original_keys for m in singles.mutations]]],
        ignore_index=True,
    )
    generator = PosteriorLandscape.condition(
        sequence, original, LandscapePrior.from_dict(metadata["prior"])
    )
    reserved = set()
    for seed in (42, 43, 44):
        reserved.update(pd.read_csv(reference / f"samples_{seed}.csv").variant)
    if args.confirm:
        early_pool = raw.loc[(raw.depth > 1) & ~raw.variant.isin(reserved)]
        for seed in (42, 43, 44):
            reserved.update(early_pool.sample(n=192, random_state=seed).variant)
        development = Path("results/creilov_pilot_assessment")
        for seed in (20260909, 20260910, 20260911):
            for prefix in ("pilot", "samples"):
                reserved.update(pd.read_csv(development / f"{prefix}_{seed}.csv").variant)
    pool = raw.loc[(raw.depth > 1) & ~raw.variant.isin(reserved)]
    output = Path(
        "results/creilov_pilot_readout" if args.readout else "results/creilov_pilot_assessment"
    )
    if args.burden:
        output = output.with_name(output.name + "_burden")
    if args.confirm:
        output = output.with_name(output.name + "_confirmation")
    output.mkdir(parents=True, exist_ok=True)
    rows, searches, fitted, spectra = [], [], [], []
    all_mutations = raw.mutations.tolist()
    base, design = pair_design(all_mutations, effects)
    positions = sorted({p for p, _ in effects})
    alphabets = {
        p: [sequence[p - 1]] + sorted({aa for pos, aa in effects if pos == p}) for p in positions
    }
    codes = np.asarray(
        [[alphabets[p].index(m.get(p, sequence[p - 1])) for p in positions] for m in all_mutations],
        dtype=np.int16,
    )
    # Use new genotype samples and pilot seeds after the development experiments.
    split_seeds = (20260920, 20260921, 20260922) if args.confirm else (20260909, 20260910, 20260911)
    draw_seeds = (20260923, 20260924, 20260925) if args.confirm else (20260912, 20260913, 20260914)
    methods = (
        tuple(METHODS) if args.confirm else ("random", "greedy", "ridge_greedy", "forest_greedy")
    )
    trials = 20 if args.confirm else 10
    binary_indices = np.asarray((codes != 0) @ (1 << np.arange(len(positions))), dtype=np.int64)
    for split_seed in split_seeds:
        chosen = pool.sample(n=2048, random_state=split_seed)
        pilot, test = chosen.iloc[:48], chosen.iloc[48:]
        assert set(pilot.variant).isdisjoint(test.variant)
        assert set(chosen.variant).isdisjoint(reserved)
        readout = None
        fit_data = pilot
        fit_effects = effects
        if args.readout:
            observed_values = np.r_[pilot.fitness, list(effects.values()), 0.0]
            margin = 0.05 * np.ptp(observed_values)
            readout = GlobalEpistasisSigmoid(
                floor=float(observed_values.min() - margin),
                ceiling=float(observed_values.max() + margin),
                scale=1.0,
            )
            fit_data = pilot.copy()
            fit_data["fitness"] = readout.inverse(pilot.fitness)
            fit_effects = {key: float(readout.inverse(value)) for key, value in effects.items()}
        base, design = pair_design(all_mutations, fit_effects, burden=args.burden)
        model, scale = fit_pilot(fit_data, fit_effects, burden=args.burden)
        pilot.to_csv(output / f"pilot_{split_seed}.csv", index=False)
        fitted.append(
            dict(
                seed=split_seed,
                coefficients=(model[-1].coef_ / model[0].scale_).tolist(),
                pair_scale=scale,
                floor=readout.floor if readout else None,
                ceiling=readout.ceiling if readout else None,
            )
        )
        # Every arm uses the original audit feature definition and fixed classifiers.
        real_features = completion_fidelity_features(original, test).to_numpy()
        groups = np.arange(len(test))
        features_saved = {"groups": groups, "measured": real_features}
        samples = (
            test[["variant", "depth", "fitness"]].rename(columns={"fitness": "measured"}).copy()
        )
        keep = ~raw.index.isin(pilot.index)
        search_codes = codes[keep]
        space = SearchSpace(
            search_codes,
            tuple(map(len, alphabets.values())),
            np.flatnonzero(keep),
            None,
            {tuple(map(int, code)): i for i, code in enumerate(search_codes)},
        )
        wt_index = np.flatnonzero((raw.depth.to_numpy() == 0)[keep])
        object.__setattr__(space, "wild_type_row", int(wt_index[0]))
        search_features = make_features(space)
        full_arms = {"measured": raw.fitness.to_numpy()}
        calibrated = (
            PilotCalibratedLandscape.fit(sequence, singles, pilot)
            if args.burden and args.readout
            else None
        )
        if calibrated is not None:
            (output / f"model_{split_seed}.json").write_text(
                json.dumps(calibrated.to_dict(), indent=2) + "\n"
            )
        for draw in draw_seeds:
            values = (
                base + model.predict(design) + scale * pair_noise(all_mutations, sequence, draw)
            )
            if readout is not None:
                values = readout.forward(values)
            if calibrated is not None:
                values = calibrated.draw(draw, reference_size=None).evaluate_batch(all_mutations)
            full_arms[f"pilot_{draw}"] = values
            spectra.append(
                dict(
                    split_seed=split_seed,
                    draw=draw,
                    **binary_cube_spectral_metrics(
                        raw.fitness.to_numpy(),
                        values,
                        binary_indices,
                        len(positions),
                        max_order=None,
                    ),
                )
            )
            original_values = generator.draw(draw).evaluate_batch(test.mutations.tolist())
            for arm, sample_values in (
                (f"pilot_{draw}", values[test.index]),
                (f"original_{draw}", original_values),
            ):
                synth = completion_fidelity_features(original, test, sample_values).to_numpy()
                features_saved[arm] = synth
                samples[arm] = sample_values
                for classifier in ("logistic", "histogram"):
                    result = classifier_two_sample_test(
                        real_features,
                        synth,
                        real_groups=groups,
                        synthetic_groups=groups,
                        classifier=classifier,
                        seed=split_seed,
                        n_permutations=0,
                    )
                    row = dict(
                        split_seed=split_seed,
                        arm=arm,
                        classifier=classifier,
                        auc=result.auc,
                        balanced_accuracy=result.accuracy,
                        raw_w1=wasserstein_distance(test.fitness, sample_values),
                        raw_w1_real_sd=wasserstein_distance(test.fitness, sample_values)
                        / np.std(test.fitness),
                        fold_aucs=json.dumps(result.fold_aucs),
                        fold_accuracies=json.dumps(result.fold_accuracies),
                    )
                    rows.append(row)
                    print(row, flush=True)
        np.savez_compressed(output / f"features_{split_seed}.npz", **features_saved)
        samples.to_csv(output / f"samples_{split_seed}.csv", index=False)
        for arm, values in full_arms.items():
            for method in methods:
                for search_seed in range(trials):
                    best = run_method(
                        method, space, values[keep], search_features, budget=48, seed=search_seed
                    )
                    searches.append(
                        dict(
                            split_seed=split_seed,
                            arm=arm,
                            method=method,
                            search_seed=search_seed,
                            best=best,
                        )
                    )
                print(f"search {split_seed} {arm} {method}", flush=True)
            print(f"search finished {split_seed} {arm}", flush=True)
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(searches).to_csv(output / "search.csv", index=False)
        pd.DataFrame(spectra).to_csv(output / "spectra.csv", index=False)
    scores = pd.DataFrame(searches).rename(
        columns={"split_seed": "dataset", "search_seed": "trial"}
    )
    scores["dataset"] = scores.dataset.astype(str)
    scores["regret"] = -scores.best
    pairwise_winner_transfer(scores, seed=42).to_csv(output / "method_pairs.csv", index=False)
    info = dict(
        protocol="creilov-pilot-check-v1",
        pilot_budget=48,
        test_samples=2000,
        claim="Prototype with target combinations. This does not establish singles-only transfer.",
        fitted=fitted,
        prior=metadata["prior"],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        code_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "src/mutascape/completion.py",
                "src/mutascape/nonlinearity.py",
                "src/mutascape/benchmarks/evaluation.py",
                "src/mutascape/benchmarks/methods.py",
                "src/mutascape/benchmarks/validity.py",
            )
        },
        dms_sha256=hashlib.sha256(Path("data/creilov/creilov.csv").read_bytes()).hexdigest(),
        excluded_development_variants=len(reserved),
        features="Original DMS audit features shared by every arm",
        permutations=0,
        development=(
            "Three older samples informed the model family. "
            "New samples exclude those genotypes. The readout follow-up reuses these samples."
        ),
        readout=args.readout,
        burden=args.burden,
        confirmation=args.confirm,
        split_seeds=split_seeds,
        draw_seeds=draw_seeds,
        methods=methods,
        trials=trials,
        software={name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        search="48 queries. Pilot genotypes excluded from candidates.",
        spectra="Full measured library, including 48 calibration rows. Descriptive fit check.",
    )
    (output / "metadata.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
