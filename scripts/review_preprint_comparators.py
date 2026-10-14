"""Check current preprint values and compare additive search benchmarks."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kendalltau, rankdata, spearmanr, wasserstein_distance

from mutascape import QuantileCalibratedLandscape, from_dms
from mutascape.benchmarks.evaluation import (
    classifier_two_sample_test,
    completion_fidelity_features,
)
from scripts.benchmark_efficacy import ENSEMBLE, OUT, PROTEINS, SETTINGS, verify
from scripts.build_singles_manuscript import PILOT, RESULTS, ROOT, SEEDS, VALIDATION


def spectral_fractions(values, cubes):
    """Calculate parity coefficients without the production transform."""
    masks = range(8)
    parity = np.array([[(-1) ** (i & j).bit_count() for j in masks] for i in masks])
    coefficients = np.asarray(values)[cubes] @ parity / 8
    energy = np.square(coefficients[:, 1:]).sum(axis=0)
    orders = np.array([i.bit_count() for i in range(1, 8)])
    return np.array([energy[orders == k].sum() for k in (1, 2, 3)]) / energy.sum()


def selected_regret(predicted, measured):
    rounded = np.round(predicted, 12)
    chosen = rounded == rounded.min(axis=-1, keepdims=True)
    return (chosen * measured).sum(axis=-1) / chosen.sum(axis=-1)


def row_spearman(a, b):
    a, b = rankdata(np.round(a, 12), axis=-1), rankdata(np.round(b, 12), axis=-1)
    a -= a.mean(axis=-1, keepdims=True)
    b -= b.mean(axis=-1, keepdims=True)
    scale = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.divide((a * b).sum(axis=-1), scale, where=scale > 0, out=np.full_like(scale, np.nan))


def audit_search(destination):
    verify()
    audit = json.loads((OUT / "audit.json").read_text())
    for filename, expected in audit["source_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    trials = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("*/trials_*.csv"))])
    assert len(trials) == 17280
    assert (trials.spent == trials.budget).all()
    assert (trials[["warnings", "fallback_queries"]] == 0).all().all()
    methods = sorted(trials.method.unique())
    for (protein, model, sample), frame in trials.groupby(["protein", "model", "sample"]):
        path = OUT / protein / ("measured.npy" if model == "measured" else f"{model}_{sample}.npy")
        fitness = np.load(path)
        known = np.load(OUT / protein / "design.npz")["known"]
        np.testing.assert_allclose(
            frame.regret, (fitness.max() - frame.best) / np.ptp(fitness), atol=1e-12
        )
        np.testing.assert_allclose(
            frame.remaining_regret,
            (fitness.max() - frame.best) / (fitness.max() - fitness[known].max()),
            atol=1e-12,
        )
    records, contrasts = [], []
    rng = np.random.default_rng(2026093017)
    n_bootstrap = 20000
    for budget in (48, 96):
        vectors, boot = {}, {}
        for protein in PROTEINS:
            group = trials.query("protein == @protein and budget == @budget")
            vectors[protein], boot[protein] = {}, {}
            sample_index = rng.integers(5, size=(n_bootstrap, 5))
            repeat_index = rng.integers(6, size=(n_bootstrap, 5, 6))
            measured_index = rng.integers(30, size=(n_bootstrap, 30))
            for model in ("measured", *SETTINGS):
                values = (
                    group.query("model == @model")
                    .pivot(index=["sample", "trial"], columns="method", values="regret")
                    .sort_index()
                    .reindex(columns=methods)
                    .to_numpy()
                )
                assert values.shape == (30, 12)
                vectors[protein][model] = values.mean(axis=0)
                if model == "measured":
                    boot[protein][model] = values[measured_index].mean(axis=1)
                else:
                    boot[protein][model] = values.reshape(5, 6, 12)[
                        sample_index[:, :, None], repeat_index
                    ].mean(axis=(1, 2))
            vectors[protein]["ensemble"] = np.mean([vectors[protein][m] for m in ENSEMBLE], axis=0)
        for protein in PROTEINS:
            measured = vectors[protein]["measured"]
            vectors[protein]["fixed"] = np.mean(
                [vectors[p]["measured"] for p in PROTEINS if p != protein], axis=0
            )
            for model in (*SETTINGS, "ensemble", "fixed"):
                prediction = vectors[protein][model]
                a, b = np.round(prediction, 12), np.round(measured, 12)
                records.append(
                    dict(
                        protein=protein,
                        budget=budget,
                        model=model,
                        spearman=spearmanr(a, b).statistic,
                        tau=kendalltau(a, b).statistic,
                        mspd=np.mean((prediction - measured) ** 2),
                        selected_regret=selected_regret(prediction, measured),
                    )
                )
        protein_index = rng.integers(4, size=(n_bootstrap, 4))
        for target, control in (("additive", "mutascape"), ("zero_pair", "mutascape")):
            for metric in ("spearman", "selected_regret"):
                function = row_spearman if metric == "spearman" else selected_regret
                delta = np.stack(
                    [
                        function(boot[p][target], boot[p]["measured"])
                        - function(boot[p][control], boot[p]["measured"])
                        for p in PROTEINS
                    ]
                )
                sampled = np.nanmean(delta[protein_index, np.arange(n_bootstrap)[:, None]], axis=1)
                point = np.mean(
                    [
                        function(vectors[p][target], vectors[p]["measured"])
                        - function(vectors[p][control], vectors[p]["measured"])
                        for p in PROTEINS
                    ]
                )
                low, high = np.nanquantile(sampled, [0.025, 0.975])
                contrasts.append(
                    dict(
                        budget=budget,
                        target=target,
                        control=control,
                        metric=metric,
                        difference=point,
                        low=low,
                        high=high,
                    )
                )
    derived = pd.DataFrame(records).set_index(["protein", "budget", "model"]).sort_index()
    saved = pd.read_csv(OUT / "per_protein.csv").set_index(derived.index.names).sort_index()
    np.testing.assert_allclose(derived, saved[derived.columns], rtol=1e-10, atol=1e-12)
    derived.to_csv(destination / "search_checked.csv")
    pd.DataFrame(contrasts).to_csv(destination / "additive_contrasts.csv", index=False)
    return dict(
        campaigns=len(trials),
        verified_search_rows=len(derived),
        bootstrap_replicates=n_bootstrap,
        bootstrap_seed=2026093017,
    )


def audit_fitness(destination, rerun_c2st):
    study = json.loads((VALIDATION / "singles-only-research-v1.json").read_text())
    extension = json.loads((VALIDATION / "singles-pilot-extension-v1.json").read_text())
    inventory = pd.DataFrame(study["inventory"]).set_index("protein")
    base = pd.DataFrame(study["fidelity"]).set_index(["protein", "model", "seed"])
    calibrated = pd.DataFrame(extension["metrics"]).set_index(["protein", "seed"])
    records = []
    for protein in inventory.index:
        folder = RESULTS / protein
        frame = pd.read_pickle(folder / "frame.pkl")
        design = np.load(folder / "design.npz")
        meta = json.loads((folder / "metadata.json").read_text())
        assert (
            hashlib.sha256((ROOT / meta["source"]).read_bytes()).hexdigest()
            == meta["source_sha256"]
        )
        known = np.flatnonzero(frame.depth <= 1)
        assert len(frame) == inventory.loc[protein, "measured"]
        assert len(known) - 1 == inventory.loc[protein, "singles"]
        assert np.prod(design["shape"]) == inventory.loc[protein, "declared"]
        test, cubes = design["test"], design["cubes"]
        assert len(test) == len(np.unique(test)) == 1000
        assert cubes.shape == (256, 8)
        assert not np.intersect1d(known, test).size
        for cube in cubes:
            codes = design["codes"][cube]
            assert len(np.unique(codes, axis=0)) == 8
            sizes = [len(np.unique(codes[:, j])) for j in range(codes.shape[1])]
            assert sizes.count(2) == 3 and max(sizes) == 2
        excluded = np.unique(np.r_[test, cubes.ravel(), known])
        pilot_rows = pd.read_csv(PILOT / protein / "pilot.csv")["row"].to_numpy()
        assert len(pilot_rows) == len(np.unique(pilot_rows)) == 192
        assert not np.intersect1d(pilot_rows, excluded).size
        real = frame.fitness.to_numpy()
        measured_spectrum = spectral_fractions(real, cubes)
        for model in ("sigmoid", "additive", "calibrated"):
            for seed in SEEDS[:1] if model == "additive" else SEEDS:
                if model == "calibrated":
                    values = np.load(PILOT / protein / f"evaluation_{seed}.npy")
                    saved = calibrated.loc[protein, seed]
                else:
                    values = np.load(folder / f"fitness_{model}_{seed}.npy")
                    saved = base.loc[protein, model, seed]
                np.testing.assert_array_equal(values[known], real[known])
                w1 = wasserstein_distance(real[test], values[test]) / real[test].std()
                walsh = np.abs(measured_spectrum - spectral_fractions(values, cubes)).sum()
                np.testing.assert_allclose([w1, walsh], saved[["w1", "local_walsh_l1"]], atol=1e-12)
                auc = float(saved.auc_sequence)
                if rerun_c2st and protein in ("GB1", "TrpB4", "T7", "TEV"):
                    real_features = completion_fidelity_features(
                        frame.iloc[known], frame.iloc[test]
                    ).to_numpy()
                    fake_features = completion_fidelity_features(
                        frame.iloc[known], frame.iloc[test], values[test]
                    ).to_numpy()
                    codes = design["codes"][test]
                    indicators = np.column_stack(
                        [codes[:, j] == a for j, n in enumerate(design["shape"]) for a in range(n)]
                    )
                    score = classifier_two_sample_test(
                        np.column_stack([real_features, indicators]),
                        np.column_stack([fake_features, indicators]),
                        real_groups=test,
                        synthetic_groups=test,
                        classifier="histogram",
                        n_permutations=0,
                        seed=2026092300,
                    )
                    np.testing.assert_allclose(score.auc, auc, atol=1e-12)
                records.append(
                    dict(
                        protein=protein,
                        model=model,
                        seed=seed,
                        w1=w1,
                        local_walsh_l1=walsh,
                        auc_sequence=auc,
                    )
                )
        print(f"Verified saved fitness and spectra for {protein}", flush=True)
    pd.DataFrame(records).to_csv(destination / "fitness_checked.csv", index=False)
    return dict(
        proteins=len(inventory),
        fitness_metric_rows=len(records),
        rerun_sequence_classifiers=28 if rerun_c2st else 0,
    )


def audit_sample_size(destination: Path) -> dict[str, dict[str, float]]:
    """Compare sampled and full-data Wasserstein distances for single-site generators."""
    study = json.loads((VALIDATION / "singles-only-research-v1.json").read_text())
    records = []
    for protein in study["protocol"]["confirmation"]:
        folder = RESULTS / protein
        frame = pd.read_pickle(folder / "frame.pkl")
        with np.load(folder / "design.npz") as design:
            test = design["test"]
        measured = frame.fitness.to_numpy()
        eligible = np.flatnonzero(frame.depth.to_numpy() > 1)
        assert len(np.unique(test)) == 1000
        assert np.isin(test, eligible).all()
        for model in ("sigmoid", "additive"):
            for seed in SEEDS if model == "sigmoid" else SEEDS[:1]:
                values = np.load(folder / f"fitness_{model}_{seed}.npy")
                assert np.isfinite(values[eligible]).all()
                row = dict(protein=protein, model=model, seed=seed, n_full=len(eligible))
                for label, indices in (("sample", test), ("full", eligible)):
                    scale = np.std(measured[indices])
                    assert scale > 0
                    row[f"{label}_w1"] = (
                        wasserstein_distance(measured[indices], values[indices]) / scale
                    )
                records.append(row)
    results = pd.DataFrame(records)
    results.to_csv(destination / "sample_size_wasserstein.csv", index=False)
    summary = (
        results.groupby(["protein", "model"])[["sample_w1", "full_w1"]]
        .median()
        .groupby("model")
        .mean()
        .to_dict(orient="index")
    )
    (destination / "sample_size_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def audit_assets(destination):
    """Rebuild tables and plot pixels in a temporary directory."""
    from scripts import build_efficacy_manuscript as search
    from scripts import build_singles_manuscript as single

    manuscript = ROOT.parent / "mutascape_manu_v1"

    def png_only(figure, folder, name):
        figure.savefig(folder / f"{name}.png", bbox_inches="tight", dpi=240)
        plt.close(figure)

    original_saves = single.save, search.save
    checks = {}
    try:
        single.save = search.save = png_only
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for folder in ("tables", "figures"):
                (output / folder).mkdir()
            study = json.loads((VALIDATION / "singles-only-research-v1.json").read_text())
            extension = json.loads((VALIDATION / "singles-pilot-extension-v1.json").read_text())
            single.tables(study, extension, output / "tables")
            search.build(output)
            single.creilov_figure(output / "figures")
            for path in output.glob("*/*"):
                saved = manuscript / path.relative_to(output)
                if path.suffix == ".png":
                    actual = np.array(Image.open(path))
                    expected = np.array(Image.open(saved))
                    checks[path.name] = np.array_equal(actual, expected)
                else:
                    checks[path.name] = saved.read_bytes() == path.read_bytes()
    finally:
        single.save, search.save = original_saves
    assert all(checks.values()), checks
    (destination / "asset_checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    frame = pd.read_pickle(RESULTS / "CreiLOV/frame.pkl")
    design = np.load(RESULTS / "CreiLOV/design.npz")
    meta = json.loads((RESULTS / "CreiLOV/metadata.json").read_text())
    known = np.flatnonzero(frame.depth <= 1)
    requested = np.unique(np.r_[design["test"], design["cubes"].ravel(), known])
    generator = from_dms(frame.iloc[known][["mutations", "fitness"]], wt_sequence=meta["sequence"])
    calibrated = QuantileCalibratedLandscape.from_dict(
        json.loads((PILOT / "CreiLOV/model.json").read_text())
    )
    for seed in SEEDS:
        for model, path in (
            (generator, RESULTS / "CreiLOV" / f"fitness_sigmoid_{seed}.npy"),
            (calibrated, PILOT / "CreiLOV" / f"evaluation_{seed}.npy"),
        ):
            current = model.draw(seed=seed).evaluate_batch(frame.iloc[requested].mutations.tolist())
            np.testing.assert_array_equal(current, np.load(path)[requested])
    fitness = np.array([[2.0, 2.0], [4.0, 8.0]])
    parity = np.array([[1, 1], [1, -1]])
    coefficients = parity @ fitness @ parity / 4
    np.testing.assert_array_equal(coefficients, [[4, -1], [-2, 1]])
    energy = np.array([coefficients[0, 1] ** 2 + coefficients[1, 0] ** 2, coefficients[1, 1] ** 2])
    np.testing.assert_allclose(energy / np.var(fitness), [5 / 6, 1 / 6])
    return dict(
        identical_assets=len(checks),
        creilov_genotypes=len(requested),
        public_api_conditions=6,
        walsh_example_verified=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/preprint_comparator_review_20260930")
    )
    parser.add_argument("--rerun-c2st", action="store_true")
    parser.add_argument("--sample-size-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.sample_size_only:
        print(json.dumps(audit_sample_size(args.output), indent=2))
        return
    report = dict(
        search=audit_search(args.output),
        fitness=audit_fitness(args.output, args.rerun_c2st),
        assets=audit_assets(args.output),
    )
    report["sources"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            ROOT.parent / "mutascape_manu_v1/main.typ",
            ROOT.parent / "mutascape_manu_v1/main.pdf",
            OUT / "audit.json",
            VALIDATION / "singles-only-research-v1.json",
            VALIDATION / "singles-pilot-extension-v1.json",
        )
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
