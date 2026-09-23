"""Compare three fixed pilot-model alternatives on fresh CreiLOV genotype splits."""

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.completion import LandscapePrior, PosteriorLandscape, dms_pairwise_unit_coefficient
from mutascape.experimental import PilotCalibratedLandscape
from mutascape.validation.fourier import binary_cube_spectral_metrics
from scripts.run_creilov_combinatorial_validation import parse_mutant

OUT = Path("results/c2st_alternatives_20260923")
DRAW_SEEDS = (2026092401, 2026092402, 2026092403)
MODELS = ("singles_only", "pilot_reference", "loo_noise", "raw_quadratic", "learned_pairs")


def features(mutations, effects, quadratic=False):
    depth = np.array([len(m) for m in mutations], dtype=float)
    additive = np.array([sum(effects[k] for k in m.items()) for m in mutations])
    extra = np.maximum(depth - 1, 0)
    columns = [extra, additive * extra / np.maximum(depth, 1)]
    if quadratic:
        columns += [extra**2, additive**2 * extra / np.maximum(depth, 1) ** 2]
    return additive, np.column_stack(columns)


def basis(mutations, keys):
    singles = np.array([[m.get(p) == aa for p, aa in keys] for m in mutations], dtype=bool)
    pairs = [(i, j) for i, j in combinations(range(len(keys)), 2) if keys[i][0] != keys[j][0]]
    design = np.column_stack([singles[:, i] & singles[:, j] for i, j in pairs])
    return singles, design, pairs


def field(keys, pairs, sequence, seed):
    return np.array(
        [dms_pairwise_unit_coefficient(keys[i], keys[j], sequence, seed) / 2 for i, j in pairs]
    )


def ridge_fit(design, target, alpha):
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=alpha, fit_intercept=False))
    model.fit(design, target)
    return model[-1].coef_ / model[0].scale_


@dataclass
class AlternativeFits:
    sequence: str
    singles: pd.DataFrame
    keys: list
    reference: PilotCalibratedLandscape
    loo_scale: float
    raw_beta: np.ndarray
    raw_scale: float
    pair_mean: np.ndarray
    pair_root: np.ndarray
    bayes_alpha: float
    bayes_lambda: float

    @classmethod
    def fit(cls, sequence, singles, pilot):
        reference = PilotCalibratedLandscape.fit(sequence, singles, pilot)
        keys = sorted(reference.effects)
        additive, design = features(pilot.mutations, reference.effects)
        latent = reference.readout.inverse(pilot.fitness.to_numpy() - reference.anchors[()])
        residual = latent - additive - design @ reference.coefficients
        _, pair_design, _ = basis(pilot.mutations, keys)
        pairs_per_row = pair_design.sum(axis=1)
        loo = []
        raw_effects = {k[0]: v - reference.anchors[()] for k, v in reference.anchors.items() if k}
        raw_additive, raw_design = features(pilot.mutations, raw_effects, quadratic=True)
        target = pilot.fitness.to_numpy() - reference.anchors[()] - raw_additive
        raw_beta = ridge_fit(raw_design, target, alpha=10)
        raw_errors = []
        for index in range(len(pilot)):
            keep = np.arange(len(pilot)) != index
            fitted = PilotCalibratedLandscape.fit(sequence, singles, pilot.iloc[keep])
            a, x = features(pilot.mutations.iloc[[index]], fitted.effects)
            mean = fitted.readout.forward(a + x @ fitted.coefficients)[0]
            loo.append(latent[index] - reference.readout.inverse(mean))
            beta = ridge_fit(raw_design[keep], target[keep], alpha=10)
            raw_errors.append(target[index] - raw_design[index] @ beta)
        bayes = BayesianRidge(fit_intercept=False).fit(pair_design.astype(float), residual)
        eigenvalues, eigenvectors = np.linalg.eigh(bayes.sigma_)
        root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        return cls(
            sequence,
            singles.copy(),
            keys,
            reference,
            float(np.sqrt(np.sum(np.square(loo)) / pairs_per_row.sum())),
            raw_beta,
            float(np.sqrt(np.sum(np.square(raw_errors)) / pairs_per_row.sum())),
            bayes.coef_,
            root,
            float(bayes.alpha_),
            float(bayes.lambda_),
        )

    def predict(self, mutations, seed, model):
        if model not in MODELS[1:]:
            raise ValueError(f"Unknown pilot alternative: {model}")
        g = self.reference
        _, pair_design, pairs = basis(mutations, self.keys)
        noise = pair_design @ field(self.keys, pairs, self.sequence, seed)
        if model == "raw_quadratic":
            effects = {k[0]: v - g.anchors[()] for k, v in g.anchors.items() if k}
            a, x = features(mutations, effects, quadratic=True)
            values = g.anchors[()] + a + x @ self.raw_beta + self.raw_scale * noise
        else:
            a, x = features(mutations, g.effects)
            if model == "learned_pairs":
                rng = np.random.default_rng(seed)
                coefficients = self.pair_mean + self.pair_root @ rng.standard_normal(len(pairs))
                interaction = pair_design @ coefficients
            else:
                scale = self.loo_scale if model == "loo_noise" else g.pair_scale
                interaction = scale * noise
            values = g.anchors[()] + g.readout.forward(a + x @ g.coefficients + interaction)
        for i, mutations_at_row in enumerate(mutations):
            key = tuple(sorted(mutations_at_row.items()))
            if key in g.anchors:
                values[i] = g.anchors[key]
        return values


def load_data():
    meta = json.loads(Path("results/creilov_c2st/metadata.json").read_text())
    raw = pd.read_excel(meta["raw_path"]).rename(columns={"Unnamed: 0": "variant"})
    raw["mutations"] = [{p: aa for p, _, aa in parse_mutant(v)} for v in raw.variant]
    raw["depth"] = raw.mutations.map(len)
    raw["fitness"] = raw.mean_log - float(raw.loc[raw.depth == 0, "mean_log"].iloc[0])
    raw = raw.loc[np.isfinite(raw.fitness)].reset_index(drop=True)
    singles = raw.loc[raw.depth <= 1, ["mutations", "depth", "fitness"]]
    sequence = "".join(Path("data/creilov/creilov_wt.fasta").read_text().splitlines()[1:])
    original = pd.read_csv("data/creilov/creilov.csv")
    original = pd.DataFrame(
        [{"mutations": {}, "depth": 0, "fitness": 0.0}]
        + [
            {"mutations": {r.position: r.mutant}, "depth": 1, "fitness": r.fitness}
            for r in original.itertuples()
        ]
    )
    known = {tuple(m.items()) for m in original.mutations}
    original = pd.concat(
        [original, singles.loc[[tuple(m.items()) not in known for m in singles.mutations]]],
        ignore_index=True,
    )
    return raw, singles, original, sequence, meta


def main():
    OUT.mkdir(exist_ok=True)
    raw, singles, original, sequence, meta = load_data()
    excluded = set()
    files = sorted(Path("results").glob("creilov*/samples_*.csv"))
    files += sorted(Path("results").glob("creilov*/pilot_*.csv"))
    for path in files:
        frame = pd.read_csv(path)
        if "variant" in frame:
            excluded.update(frame.variant)
    early_excluded = set()
    for seed in (42, 43, 44):
        early_excluded.update(pd.read_csv(f"results/creilov_c2st/samples_{seed}.csv").variant)
    early_pool = raw.loc[(raw.depth > 1) & ~raw.variant.isin(early_excluded)]
    for seed in (42, 43, 44):
        excluded.update(early_pool.sample(n=192, random_state=seed).variant)
    pool = raw.loc[(raw.depth > 1) & ~raw.variant.isin(excluded)]
    selected = pool.sample(n=3 * 2048, random_state=2026092300)
    protocol = {
        "models": list(MODELS),
        "pilot_size": 48,
        "test_size": 2000,
        "splits": 3,
        "draw_seeds": DRAW_SEEDS,
        "selection_seed": 2026092300,
        "classifier_features": "Original audit features, shared by every model",
        "classifier_settings": "Logistic and histogram, five paired folds, no permutations",
        "secondary_classifier": "Histogram with genotype indicators added to audit features",
        "loo_noise": (
            "Refit all pilot parameters without each calibration row. Estimate latent "
            "residual scale in the full pilot coordinate system."
        ),
        "raw_quadratic": (
            "Raw additive fitness plus four depth/additive features. Ridge penalty 10. "
            "Leave-one-out residual pair scale."
        ),
        "learned_pairs": (
            "Fit Bayesian ridge to pilot latent residuals after the reference mean. "
            "Sample shared pair coefficients from their posterior. No added observation "
            "noise."
        ),
        "spectra": (
            "Descriptive full-library binary projection. Includes calibration rows and "
            "prior evaluation genotypes."
        ),
        "limits": (
            "Exploratory within-protein study. Fresh genotype splits do not provide "
            "independent-protein confirmation."
        ),
        "raw_sha256": hashlib.sha256(Path(meta["raw_path"]).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "software": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")},
    }
    # Record every choice and genotype identity before fitting or scoring.
    (OUT / "experiment_source.py.txt").write_bytes(Path(__file__).read_bytes())
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for split in range(3):
        group = selected.iloc[split * 2048 : (split + 1) * 2048]
        group.iloc[:48].to_csv(OUT / f"pilot_{split}.csv", index=False)
        group.iloc[48:].to_csv(OUT / f"test_{split}.csv", index=False)
    posterior = PosteriorLandscape.condition(
        sequence, singles, LandscapePrior.from_dict(meta["prior"])
    )
    keys = sorted(posterior.single_effects)
    indicators, full_pairs, pairs = basis(raw.mutations, keys)
    positions = sorted({p for p, _ in keys})
    binary_index = np.array(
        [sum(1 << j for j, p in enumerate(positions) if p in m) for m in raw.mutations]
    )
    all_mutations = raw.mutations.tolist()
    default_additive = np.array(
        [sum(posterior.preimage_effects[k] for k in m.items()) for m in all_mutations]
    )
    rows, spectra, fits = [], [], []
    default_values = {}
    for draw in DRAW_SEEDS:
        noise = full_pairs @ field(keys, pairs, sequence, draw)
        values = posterior.observations[()] + posterior.nonlinearity.forward(
            default_additive + 0.2 * posterior.preimage_scale * noise
        )
        np.testing.assert_allclose(
            values[:20], posterior.draw(draw).evaluate_batch(all_mutations[:20]), atol=1e-12
        )
        default_values[draw] = values
    for split in range(3):
        group = selected.iloc[split * 2048 : (split + 1) * 2048]
        pilot, test = group.iloc[:48], group.iloc[48:]
        assert set(pilot.variant).isdisjoint(test.variant)
        assert set(group.variant).isdisjoint(excluded)
        fitted = AlternativeFits.fit(sequence, singles, pilot)
        fits.append(
            dict(
                split=split,
                reference_scale=fitted.reference.pair_scale,
                loo_scale=fitted.loo_scale,
                raw_scale=fitted.raw_scale,
                raw_beta=fitted.raw_beta.tolist(),
                bayes_alpha=fitted.bayes_alpha,
                bayes_lambda=fitted.bayes_lambda,
            )
        )
        real_features = completion_fidelity_features(original, test).to_numpy()
        groups = np.arange(len(test))
        predictions = {}
        for draw in DRAW_SEEDS:
            for model in MODELS:
                values = (
                    default_values[draw]
                    if model == "singles_only"
                    else fitted.predict(all_mutations, draw, model)
                )
                if model != "singles_only":
                    np.testing.assert_allclose(
                        fitted.predict(singles.mutations.tolist(), draw, model),
                        singles.fitness,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        values[test.index[:20]],
                        fitted.predict(test.mutations.tolist()[:20], draw, model),
                        atol=1e-12,
                    )
                predictions[f"{model}_{draw}"] = values
                test_values = values[test.index]
                features_synthetic = completion_fidelity_features(
                    original, test, test_values
                ).to_numpy()
                basic = dict(
                    split=split,
                    draw=draw,
                    model=model,
                    raw_w1=wasserstein_distance(test.fitness, test_values),
                    raw_w1_real_sd=wasserstein_distance(test.fitness, test_values)
                    / test.fitness.std(ddof=0),
                    spearman=float(spearmanr(test.fitness, test_values).statistic),
                )
                for classifier, feature_set in (
                    ("logistic", "audit"),
                    ("histogram", "audit"),
                    ("histogram", "with_genotype"),
                ):
                    real_x, synthetic_x = real_features, features_synthetic
                    if feature_set == "with_genotype":
                        real_x = np.column_stack([real_x, indicators[test.index]])
                        synthetic_x = np.column_stack([synthetic_x, indicators[test.index]])
                    result = classifier_two_sample_test(
                        real_x,
                        synthetic_x,
                        real_groups=groups,
                        synthetic_groups=groups,
                        classifier=classifier,
                        seed=2026092301 + split,
                        n_permutations=0,
                    )
                    rows.append(
                        dict(
                            **basic,
                            classifier=classifier,
                            feature_set=feature_set,
                            auc=result.auc,
                            balanced_accuracy=result.accuracy,
                            fold_aucs=json.dumps(result.fold_aucs),
                        )
                    )
                spectra.append(
                    dict(
                        split=split,
                        draw=draw,
                        model=model,
                        **binary_cube_spectral_metrics(
                            raw.fitness.to_numpy(), values, binary_index, len(positions)
                        ),
                    )
                )
                pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
                pd.DataFrame(spectra).to_csv(OUT / "spectra.csv", index=False)
                print(f"Scored split {split}, draw {draw}, model {model}", flush=True)
        np.savez_compressed(OUT / f"full_predictions_{split}.npz", **predictions)
    (OUT / "fitted_parameters.json").write_text(json.dumps(fits, indent=2) + "\n")
    summary = (
        pd.DataFrame(rows)
        .groupby(["model", "feature_set", "classifier"])[
            ["auc", "balanced_accuracy", "raw_w1", "spearman"]
        ]
        .median()
    )
    summary.to_csv(OUT / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
