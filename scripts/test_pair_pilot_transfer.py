"""Test the fixed pair-learning model on three further protein libraries."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import BayesianRidge

from mutascape.benchmarks.datasets import (
    COMBINGYM_MANIFEST,
    SSMULA_MANIFEST,
    _graphfla_spec,
    load_landscape,
)
from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.benchmarks.topology import measured_tensor, supported_cube
from mutascape.experimental import PilotCalibratedLandscape
from mutascape.validation.fourier import walsh_l1_distance
from scripts.audit_preprint_results import _spectrum
from scripts.compare_c2st_alternatives import basis, features, field
from scripts.run_review_confirmation import GRAPH

OUT = Path("results/pair_pilot_transfer_20260923")
PROTEINS = ("DHFR", "eqFP611", "ParB-Noc")
DRAW_SEEDS = (2026092501, 2026092502, 2026092503)
PILOT_SIZES = (48, 192)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protein(name):
    if name == "DHFR":
        root = Path(".mutascape/benchmark-data/ssmula")
        spec = next(s for s in SSMULA_MANIFEST if s.name == "DHFR")
    elif name == "eqFP611":
        root = Path(".mutascape/benchmark-data/combingym")
        spec = next(s for s in COMBINGYM_MANIFEST if s.name == "eqFP611_red")
    elif name == "ParB-Noc":
        root = GRAPH
        spec = _graphfla_spec("Jalal2020_NBS.csv", "ParB-Noc", "RTAG")
    else:
        raise ValueError(f"Unknown protein: {name}")
    return load_landscape(root, spec), root / spec.relative_path


@dataclass
class PairFit:
    reference: PilotCalibratedLandscape
    keys: list
    mean: np.ndarray
    root: np.ndarray
    noise_precision: float
    coefficient_precision: float

    @classmethod
    def fit(cls, sequence, singles, pilot):
        reference = PilotCalibratedLandscape.fit(sequence, singles, pilot)
        keys = sorted(reference.effects)
        additive, design = features(pilot.mutations, reference.effects)
        target = reference.readout.inverse(pilot.fitness.to_numpy() - reference.anchors[()])
        residual = target - additive - design @ reference.coefficients
        _, pair_design, _ = basis(pilot.mutations, keys)
        bayes = BayesianRidge(fit_intercept=False).fit(pair_design.astype(float), residual)
        eigenvalues, eigenvectors = np.linalg.eigh(bayes.sigma_)
        root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        return cls(reference, keys, bayes.coef_, root, float(bayes.alpha_), float(bayes.lambda_))

    def predict(self, mutations, seed, model):
        if model not in {"pilot_reference", "learned_pairs", "learned_mean", "additive"}:
            raise ValueError(f"Unknown model: {model}")
        g = self.reference
        if model == "additive":
            effects = {key[0]: value - g.anchors[()] for key, value in g.anchors.items() if key}
            values = g.anchors[()] + features(mutations, effects)[0]
        else:
            _, pair_design, pairs = basis(mutations, self.keys)
            additive, design = features(mutations, g.effects)
            if model == "pilot_reference":
                interaction = g.pair_scale * (
                    pair_design @ field(self.keys, pairs, g.domain.wild_type_sequence, seed)
                )
            else:
                coefficients = self.mean
                if model == "learned_pairs":
                    coefficients = coefficients + self.root @ np.random.default_rng(
                        seed
                    ).standard_normal(len(pairs))
                interaction = pair_design @ coefficients
            values = g.anchors[()] + g.readout.forward(
                additive + design @ g.coefficients + interaction
            )
        for i, mutation in enumerate(mutations):
            key = tuple(sorted(mutation.items()))
            if key in g.anchors:
                values[i] = g.anchors[key]
        return values


def freeze():
    OUT.mkdir(exist_ok=True)
    code = [
        Path(__file__),
        Path("scripts/compare_c2st_alternatives.py"),
        Path("scripts/run_pair_pilot_search.py"),
        Path("src/mutascape/experimental/pilot.py"),
        Path("src/mutascape/benchmarks/methods.py"),
        Path("src/mutascape/benchmarks/evaluation.py"),
        Path("src/mutascape/validation/fourier.py"),
    ]
    protocol = {
        "proteins": PROTEINS,
        "pilot_sizes": PILOT_SIZES,
        "splits": 3,
        "test_size": 2000,
        "selection_seed": 2026092500,
        "draw_seeds": DRAW_SEEDS,
        "models": ["pilot_reference", "learned_pairs", "learned_mean", "additive"],
        "primary": "48 calibration combinations. Existing pilot versus learned pair draws.",
        "secondary": "192 calibration combinations. The first 48 form the smaller pilot.",
        "features": "Same-assay singles define shared features. Paired genotype folds, five folds.",
        "classifiers": ["logistic", "histogram", "histogram_with_sequence"],
        "tuning": "No model or classifier settings selected using these test outcomes.",
        "search": {
            "pilot_size": 48,
            "budgets": [48, 96],
            "methods": "All 12 existing methods",
            "seeds": 10,
            "models": ["measured", "pilot_reference", "learned_pairs"],
            "domain": "All measured combinations except the calibration genotypes for that split",
            "draws": 3,
            "splits": 3,
            "cost": "48 calibration combinations plus search queries, beyond WT and singles",
            "limits": "Calibration fits the generator. Search methods do not receive these data.",
        },
        "spectra": "Full complete libraries, including calibration data. Descriptive only.",
        "limits": (
            "These proteins have appeared in earlier project audits. "
            "This model was developed on CreiLOV."
        ),
        "software": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
        "code_sha256": {str(p): digest(p) for p in code},
        "datasets": [],
    }
    for name in PROTEINS:
        dataset, source = load_protein(name)
        frame = dataset.variants.reset_index(drop=True)
        singles = frame.loc[frame.depth <= 1]
        cube = supported_cube(dataset, singles)
        _, coverage = measured_tensor(dataset, cube)
        assert coverage == 1 and len(frame) == cube.size
        selected = frame.loc[frame.depth > 1].sample(n=3 * 2192, random_state=2026092500)
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        for split in range(3):
            group = selected.iloc[split * 2192 : (split + 1) * 2192]
            pd.DataFrame({"row": group.index[:192]}).to_csv(
                directory / f"pilot_{split}.csv", index=False
            )
            pd.DataFrame({"row": group.index[192:]}).to_csv(
                directory / f"test_{split}.csv", index=False
            )
        frame.to_csv(directory / "genotypes.csv", index=False)
        protocol["datasets"].append(
            {
                "protein": name,
                "source": str(source),
                "source_sha256": digest(source),
                "rows": len(frame),
                "singles_with_wt": len(singles),
                "cube_shape": [len(cube.alphabets[p]) for p in cube.positions],
                "source_url": dataset.spec.source_url,
                "source_revision": dataset.spec.source_revision,
                "genotypes_sha256": digest(directory / "genotypes.csv"),
            }
        )
    for path in code:
        (OUT / (str(path).replace("/", "__") + ".txt")).write_bytes(path.read_bytes())
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def run_split(name, split, size):
    directory = OUT / name / f"p{size}_s{split}"
    if (directory / "done.json").exists():
        return
    directory.mkdir(exist_ok=True)
    dataset, _ = load_protein(name)
    frame = dataset.variants.reset_index(drop=True)
    singles = frame.loc[frame.depth <= 1]
    pilot_ids = pd.read_csv(OUT / name / f"pilot_{split}.csv").row.to_numpy()[:size]
    test_ids = pd.read_csv(OUT / name / f"test_{split}.csv").row.to_numpy()
    assert set(pilot_ids).isdisjoint(test_ids)
    pilot, test = frame.iloc[pilot_ids], frame.iloc[test_ids]
    fitted = PairFit.fit(dataset.wild_type_sequence, singles, pilot)
    (directory / "reference_fit.json").write_text(
        json.dumps(fitted.reference.to_dict(), indent=2) + "\n"
    )
    np.savez_compressed(directory / "pair_fit.npz", mean=fitted.mean, root=fitted.root)
    cube = supported_cube(dataset, singles)
    shape = tuple(len(cube.alphabets[p]) for p in cube.positions)
    codes = np.array(
        [
            [cube.alphabets[p].index(m.get(p, cube.wild_type[p])) for p in cube.positions]
            for m in frame.mutations
        ]
    )
    flat_indices = np.ravel_multi_index(codes.T, shape)
    assert len(set(flat_indices)) == len(frame)

    def tensor(values):
        out = np.empty(len(frame))
        out[flat_indices] = values
        return out.reshape(shape)

    measured_spectrum = _spectrum(tensor(frame.fitness.to_numpy()))
    indicators = basis(test.mutations, fitted.keys)[0]
    real_features = completion_fidelity_features(singles, test).to_numpy()
    groups = np.arange(len(test))
    rows, spectra = [], []
    model_draws = [(m, seed) for m in ("pilot_reference", "learned_pairs") for seed in DRAW_SEEDS]
    model_draws += [("learned_mean", DRAW_SEEDS[0]), ("additive", DRAW_SEEDS[0])]
    for model, seed in model_draws:
        values = fitted.predict(frame.mutations.tolist(), seed, model)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values[singles.index], singles.fitness)
        np.testing.assert_allclose(
            values[test_ids[:20]],
            fitted.predict(test.mutations.tolist()[:20], seed, model),
            atol=1e-12,
        )
        np.save(directory / f"{model}_{seed}.npy", values)
        prediction = values[test_ids]
        synthetic_features = completion_fidelity_features(singles, test, prediction).to_numpy()
        basic = dict(
            protein=name,
            split=split,
            pilot_size=size,
            model=model,
            draw=seed,
            raw_w1=wasserstein_distance(test.fitness, prediction),
            scaled_w1=wasserstein_distance(test.fitness, prediction) / test.fitness.std(ddof=0),
            spearman=float(spearmanr(test.fitness, prediction).statistic),
            rmse=float(np.sqrt(np.mean((prediction - test.fitness.to_numpy()) ** 2)))
            / test.fitness.std(ddof=0),
        )
        for classifier, feature_set in (
            ("logistic", "audit"),
            ("histogram", "audit"),
            ("histogram", "with_genotype"),
        ):
            real_x, synthetic_x = real_features, synthetic_features
            if feature_set == "with_genotype":
                real_x = np.column_stack([real_x, indicators])
                synthetic_x = np.column_stack([synthetic_x, indicators])
            result = classifier_two_sample_test(
                real_x,
                synthetic_x,
                real_groups=groups,
                synthetic_groups=groups,
                classifier=classifier,
                seed=2026092500 + split,
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
        synthetic_spectrum = _spectrum(tensor(values))
        spectra.append(
            dict(
                **basic,
                walsh_l1=walsh_l1_distance(measured_spectrum, synthetic_spectrum),
                measured_spectrum=json.dumps(measured_spectrum),
                synthetic_spectrum=json.dumps(synthetic_spectrum),
            )
        )
        pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
        pd.DataFrame(spectra).to_csv(directory / "spectra.csv", index=False)
        print(f"Scored {name} pilot {size} split {split} {model} {seed}", flush=True)
    (directory / "done.json").write_text(
        json.dumps(
            {
                "noise_precision": fitted.noise_precision,
                "coefficient_precision": fitted.coefficient_precision,
            }
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.freeze:
        if (OUT / "protocol.json").exists():
            raise ValueError("Protocol already exists")
        freeze()
    protocol = json.loads((OUT / "protocol.json").read_text())
    checksums = dict(protocol["code_sha256"])
    amendment = OUT / "implementation_amendment.json"
    if amendment.exists():
        checksums.update(json.loads(amendment.read_text())["code_sha256"])
    for path, checksum in checksums.items():
        assert digest(path) == checksum, path
    for dataset in protocol["datasets"]:
        assert digest(dataset["source"]) == dataset["source_sha256"]
    Parallel(n_jobs=args.jobs)(
        delayed(run_split)(name, split, size)
        for size in PILOT_SIZES
        for name in PROTEINS
        for split in range(3)
    )


if __name__ == "__main__":
    main()
