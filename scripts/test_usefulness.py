"""Run the fixed exploratory usefulness checks recorded on 30 September 2026."""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, rankdata, spearmanr, wasserstein_distance

from mutascape.benchmarks.evaluation import (
    classifier_two_sample_test,
    completion_fidelity_features,
)
from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features
from mutascape.calibrated import PilotCalibratedLandscape, empirical_quantiles
from mutascape.model import DMSLandscape
from scripts.analyse_search_selection import choice_weights
from scripts.investigate_singles_only import local_geometry
from scripts.research_pair_posterior import ResearchPairPosterior, latent_mean, pair_field
from scripts.search_selection_validation import digest, save_json
from scripts.singles_warm_start import DMSOracle

OUT = Path("results/usefulness_20260930")
SOURCE = Path("results/search_selection_20260924")
PROTEINS = ("CreiLOV", "eqFP611", "TrpB4", "Pab1", "Ube4b")
SEEDS = (2026093011, 2026093012, 2026093013)
SETTINGS = {
    "default": ("sigmoid", 0.3),
    "additive": ("linear", 0.0),
    "linear_pairs": ("linear", 0.3),
    "strong_pairs": ("sigmoid", 3.0),
}
SEARCH_MODELS = (*SETTINGS, "quantile_192", "conditioned_192", "mean_192")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists():
        verify()
        return
    sources = [
        Path(__file__),
        Path("scripts/research_pair_posterior.py"),
        Path("docs/usefulness_protocol_2026-09-30.md"),
        *Path("src/mutascape").rglob("*.py"),
    ]
    for index, protein in enumerate(PROTEINS):
        folder = OUT / protein
        folder.mkdir(exist_ok=True)
        frame = pd.read_pickle(SOURCE / protein / "frame.pkl")
        source_design = np.load(SOURCE / protein / "design.npz")
        rng = np.random.default_rng(2026093000 + index)
        cube_file = Path("results/singles_only_restart_20260923") / protein / "design.npz"
        cubes = np.load(cube_file)["cubes"] if cube_file.exists() else np.empty((0, 8), int)
        cube_rows = np.unique(cubes)
        known = np.flatnonzero(frame.depth.to_numpy() <= 1)
        combinations = np.flatnonzero(frame.depth.to_numpy() > 1)
        test = np.sort(rng.choice(np.setdiff1d(combinations, cube_rows), 1000, replace=False))
        available = np.setdiff1d(combinations, np.union1d(test, cube_rows))
        pilots = np.array([rng.choice(available, 192, replace=False) for _ in range(3)])
        reference = np.sort(rng.choice(combinations, min(8192, len(combinations)), replace=False))
        requested = np.unique(np.r_[known, test, cube_rows, reference, pilots.ravel()])
        lookup = {int(row): i for i, row in enumerate(requested)}

        def remap(rows):
            return np.array([lookup[int(r)] for r in np.ravel(rows)], dtype=int).reshape(
                np.shape(rows)
            )

        np.savez(
            folder / "design.npz",
            requested=requested,
            test=remap(test),
            known=remap(known),
            cubes=remap(cubes),
            pilots=remap(pilots),
            reference=remap(reference),
            codes=source_design["codes"][requested],
            shape=source_design["shape"],
        )
        for filename in ("frame.pkl", "design.npz", "metadata.json"):
            sources.append(SOURCE / protein / filename)
        if cube_file.exists():
            sources.append(cube_file)
        sources.append(folder / "design.npz")
    record = dict(
        proteins=PROTEINS,
        seeds=SEEDS,
        settings=SETTINGS,
        pilot_sizes=[48, 192],
        search_models=SEARCH_MODELS,
        budget=96,
        measured_repeats=30,
        synthetic_repeats_per_sample=10,
        samples=3,
        test_combinations=1000,
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        source_sha256={str(p): digest(p) for p in sources},
    )
    save_json(path, record)
    print("Recorded protocol and input hashes", flush=True)


def verify():
    record = json.loads((OUT / "manifest.json").read_text())
    for path, expected in record["source_sha256"].items():
        assert digest(path) == expected, f"Frozen input changed: {path}"


def load(protein):
    design = np.load(OUT / protein / "design.npz")
    frame = (
        pd.read_pickle(SOURCE / protein / "frame.pkl")
        .iloc[design["requested"]]
        .reset_index(drop=True)
    )
    metadata = json.loads((SOURCE / protein / "metadata.json").read_text())
    return frame, design, metadata


def quantile_map(scores, reference, target):
    unique, start, counts = np.unique(
        np.sort(scores[reference]), return_index=True, return_counts=True
    )
    probabilities = (start + counts / 2) / len(reference)
    return empirical_quantiles(target, np.interp(scores, unique, probabilities))


def fit_protein(protein):
    folder = OUT / protein
    if (folder / "fits_complete.json").exists():
        return
    frame, design, metadata = load(protein)
    variants = frame.mutations.tolist()
    singles = frame.iloc[design["known"]]
    keys = [tuple(sorted(m.items())) for m in variants]
    records = []

    def save(label, pilot, sample, values, extra=None):
        values = np.array(values, copy=True)
        values[design["known"]] = singles.fitness
        assert np.isfinite(values).all()
        record = dict(protein=protein, model=label, pilot=pilot, sample=sample)
        if extra:
            record.update(extra)
        name = f"{label}_p{pilot}_s{sample}"
        np.save(folder / f"{name}.npy", values)
        record["file"] = name + ".npy"
        records.append(record)

    for label, (mode, strength) in SETTINGS.items():
        for sample, seed in enumerate(SEEDS):
            model = DMSLandscape.from_singles(
                metadata["sequence"],
                singles,
                seed=seed,
                mode=mode,
                strength=strength,
            )
            save(label, -1, sample, model.evaluate_batch(variants))
    for pilot_index in range(3):
        for size in (48, 192):
            pilot_rows = design["pilots"][pilot_index, :size]
            assert not np.intersect1d(
                pilot_rows, np.r_[design["test"], design["cubes"].ravel()]
            ).size
            pilot = frame.iloc[pilot_rows]
            model = PilotCalibratedLandscape.fit(metadata["sequence"], singles, pilot)
            posterior = ResearchPairPosterior.fit(model, pilot)
            save_json(
                folder / f"fit_p{pilot_index}_n{size}.json",
                {
                    "base": model.to_dict(),
                    "noise_ratio": posterior.noise_ratio,
                    "noise_variance": posterior.noise_variance,
                    "pilot_rows": design["requested"][pilot_rows].tolist(),
                },
            )
            base = latent_mean(model, variants)
            for sample, seed in enumerate(SEEDS):
                independent = base + pair_field(model, variants, seed)
                conditioned = posterior.latent(variants, seed)
                for label, latent in (("quantile", independent), ("conditioned", conditioned)):
                    scores = model.anchors[()] + model.readout.forward(latent)
                    values = quantile_map(scores, design["reference"], pilot.fitness.to_numpy())
                    save(
                        f"{label}_{size}",
                        pilot_index,
                        sample,
                        values,
                        {"noise_ratio": posterior.noise_ratio},
                    )
                if sample == 0:
                    # A query's posterior score cannot depend on its batch neighbours.
                    check_rows = design["test"][:7]
                    check = posterior.latent([variants[i] for i in check_rows], seed)
                    np.testing.assert_allclose(check, conditioned[check_rows], atol=1e-10)
            if size == 192:
                scores = model.anchors[()] + model.readout.forward(posterior.latent(variants, None))
                save(
                    "mean_192",
                    pilot_index,
                    0,
                    quantile_map(scores, design["reference"], pilot.fitness.to_numpy()),
                )
    assert len(keys) == len(frame)
    save_json(folder / "fits_complete.json", records)
    print(f"Fitted {protein}, {len(records)} conditions", flush=True)


def metrics_protein(protein):
    folder = OUT / protein
    if (folder / "fidelity.csv").exists():
        return
    frame, design, _ = load(protein)
    singles, test = frame.iloc[design["known"]], frame.iloc[design["test"]]
    y = test.fitness.to_numpy()
    real = completion_fidelity_features(singles, test).to_numpy()
    codes = design["codes"][design["test"]]
    indicators = np.column_stack(
        [codes[:, j] == a for j, n in enumerate(design["shape"]) for a in range(n)]
    )
    real = np.column_stack([real, indicators])
    measured_spectrum = (
        local_geometry(frame.fitness.to_numpy(), design["cubes"])[0]
        if len(design["cubes"])
        else None
    )
    records = []
    for record in json.loads((folder / "fits_complete.json").read_text()):
        values = np.load(folder / record["file"])
        np.testing.assert_array_equal(values[design["known"]], singles.fitness)
        fake = completion_fidelity_features(singles, test, values[design["test"]]).to_numpy()
        result = classifier_two_sample_test(
            real,
            np.column_stack([fake, indicators]),
            real_groups=design["test"],
            synthetic_groups=design["test"],
            n_permutations=0,
            classifier="histogram",
            seed=2026093000,
        )
        spectrum = (
            local_geometry(values, design["cubes"])[0] if measured_spectrum is not None else None
        )
        records.append(
            dict(
                **record,
                auc=result.auc,
                w1=wasserstein_distance(y, values[design["test"]]) / np.std(y),
                spearman=spearmanr(y, values[design["test"]]).statistic,
                walsh_l1=np.abs(spectrum - measured_spectrum).sum()
                if spectrum is not None
                else np.nan,
            )
        )
    pd.DataFrame(records).to_csv(folder / "fidelity.csv", index=False)
    print(f"Scored fidelity for {protein}", flush=True)


def search_chunk(protein, label, sample):
    folder = OUT / protein
    path = folder / f"search_{label}_{sample}.csv"
    if path.exists():
        return
    frame, design, _ = load(protein)
    selected = np.r_[design["known"], design["test"]]
    selected = selected[np.lexsort(design["codes"][selected].T[::-1])]
    codes = design["codes"][selected]
    if label == "measured":
        values = frame.fitness.to_numpy()[selected]
    else:
        pilot, landscape_sample = (-1, sample) if label in SETTINGS else (sample, sample)
        if label == "mean_192":
            landscape_sample = 0
        values = np.load(folder / f"{label}_p{pilot}_s{landscape_sample}.npy")[selected]
    known = np.flatnonzero(frame.depth.to_numpy()[selected] <= 1)
    unknown = np.flatnonzero(frame.depth.to_numpy()[selected] > 1)
    np.testing.assert_array_equal(values[known], frame.fitness.to_numpy()[selected][known])
    space = SearchSpace(
        codes,
        tuple(design["shape"]),
        selected,
        int(known[np.argmax(values[known])]),
        {tuple(c): i for i, c in enumerate(codes)},
    )
    features = make_features(space)
    scale = max(float(np.ptp(values)), 1e-12)
    started = time.monotonic()
    rows = []
    for trial in range(10):
        seed = (930100000 if label == "measured" else 930200000) + 137 * (10 * sample + trial)
        for method, function in METHODS.items():
            oracle = DMSOracle(values, 96, known)
            with warnings.catch_warnings(record=True) as caught:
                if method == "random":
                    oracle.query(np.random.default_rng(seed).choice(unknown, 96, replace=False))
                else:
                    function(space, oracle, features, seed)
            fallback_queries = oracle.remaining
            if fallback_queries:
                available = np.array([row for row in unknown if row not in oracle.seen()])
                fallback = np.random.default_rng(seed + 79001).choice(
                    available, fallback_queries, replace=False
                )
                oracle.query(fallback)
            assert oracle.spent == 96, (protein, label, method, oracle.spent)
            rows.append(
                dict(
                    protein=protein,
                    model=label,
                    sample=sample,
                    trial=trial,
                    method=method,
                    regret=(values.max() - oracle.best()) / scale,
                    spent=oracle.spent,
                    warnings=len(caught),
                    fallback_queries=fallback_queries,
                )
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(
        f"Searched {protein} {label} sample {sample} in {time.monotonic() - started:.1f}s",
        flush=True,
    )


def bootstrap_tau(first, second):
    return float(kendalltau(np.round(first, 12), np.round(second, 12)).statistic)


def analyse():
    fidelity = pd.concat([pd.read_csv(OUT / p / "fidelity.csv") for p in PROTEINS])
    fidelity.to_csv(OUT / "fidelity.csv", index=False)
    summary = fidelity.groupby(["protein", "model"])[["auc", "w1", "spearman", "walsh_l1"]].median()
    summary.to_csv(OUT / "fidelity_summary.csv")
    files = [p for name in PROTEINS for p in (OUT / name).glob("search_*.csv")]
    table = pd.concat([pd.read_csv(p) for p in files])
    assert len(table) == len(PROTEINS) * 8 * 3 * 10 * len(METHODS)
    assert not table.duplicated(["protein", "model", "sample", "trial", "method"]).any()
    assert (table.spent == 96).all()
    table.to_csv(OUT / "search_trials.csv", index=False)
    means = table.groupby(["protein", "model", "method"]).regret.mean()
    means.to_csv(OUT / "method_means.csv")
    rows = []
    for protein in PROTEINS:
        actual = means.loc[protein, "measured"].reindex(METHODS).to_numpy()
        predictions = {m: means.loc[protein, m].reindex(METHODS).to_numpy() for m in SEARCH_MODELS}
        predictions["scenario_ensemble"] = np.mean([predictions[m] for m in SETTINGS], axis=0)
        other = [
            means.loc[p, "measured"].reindex(METHODS).to_numpy() for p in PROTEINS if p != protein
        ]
        predictions["fixed_other_proteins"] = np.mean([rankdata(a) for a in other], axis=0)
        for model, prediction in predictions.items():
            weights = choice_weights(prediction)
            rows.append(
                dict(
                    protein=protein,
                    model=model,
                    tau=bootstrap_tau(prediction, actual),
                    selected_regret=float(weights @ actual),
                    excess_regret=float(weights @ actual - actual.min()),
                    best_measured_regret=actual.min(),
                    selected_methods=",".join(np.array(list(METHODS))[weights > 0]),
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "search_summary.csv", index=False)
    save_json(
        OUT / "audit.json",
        dict(
            trials=len(table),
            warning_count=int(table.warnings.sum()),
            fallback_runs=int((table.fallback_queries > 0).sum()),
            fallback_queries=int(table.fallback_queries.sum()),
            full_budget=True,
            source_sha256={str(p): digest(p) for p in files},
            limits="Exploratory results on existing proteins and a restricted candidate library",
        ),
    )
    print(summary.round(3).to_string())
    print(
        pd.DataFrame(rows)
        .groupby("model")[["tau", "selected_regret", "excess_regret"]]
        .mean()
        .round(4)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "metrics", "search", "analyse"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    verify()
    if args.stage in {"fit", "metrics"}:
        function = fit_protein if args.stage == "fit" else metrics_protein
        Parallel(n_jobs=args.jobs)(delayed(function)(p) for p in PROTEINS)
    elif args.stage == "search":
        Parallel(n_jobs=args.jobs)(
            delayed(search_chunk)(p, m, s)
            for p in PROTEINS
            for m in ("measured", *SEARCH_MODELS)
            for s in range(3)
        )
    else:
        analyse()


if __name__ == "__main__":
    main()
