"""Develop distribution-calibrated fixed landscapes using measured combinations."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata, spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.benchmarks.topology import supported_cube
from scripts.test_pair_pilot_transfer import OUT as PREVIOUS
from scripts.test_pair_pilot_transfer import PROTEINS, PairFit, load_protein

OUT = Path("results/copula_development_20260923")


def empirical_quantile(sample, probabilities):
    """Interpolate order statistics, retaining intervals occupied by repeated values."""
    values = np.sort(np.asarray(sample))
    knots = (np.arange(len(values)) + 0.5) / len(values)
    return np.interp(probabilities, knots, values)


def remap(values, pilot_fitness, eligible):
    ranks = (rankdata(values[eligible]) - 0.5) / eligible.sum()
    mapped = np.empty(len(values))
    mapped[eligible] = empirical_quantile(pilot_fitness, ranks)
    mapped[~eligible] = values[~eligible]
    return mapped


def cube_codes(dataset):
    frame = dataset.variants.reset_index(drop=True)
    singles = frame.loc[frame.depth <= 1]
    cube = supported_cube(dataset, singles)
    codes = np.array(
        [
            [cube.alphabets[p].index(m.get(p, cube.wild_type[p])) for p in cube.positions]
            for m in frame.mutations
        ],
        dtype=np.int16,
    )
    shape = tuple(len(cube.alphabets[p]) for p in cube.positions)
    return codes, shape


def distance(left, right):
    out = np.zeros((len(left), len(right)), dtype=float)
    for j in range(left.shape[1]):
        out += left[:, j, None] != right[None, :, j]
    return out


def prior_field(shape, rho, seed):
    rng = np.random.default_rng(seed)
    result = rng.standard_normal(shape)
    for axis, size in enumerate(shape):
        covariance = (1 - rho) * np.eye(size) + rho * np.ones((size, size))
        root = np.linalg.cholesky(covariance)
        result = np.moveaxis(result, axis, 0)
        result = (root @ result.reshape(size, -1)).reshape(result.shape)
        result = np.moveaxis(result, 0, axis)
    return result.ravel()


def summary_features(frame, singles):
    wt = float(singles.loc[singles.depth == 0, "fitness"].iloc[0])
    effects = {k: v - wt for m, v in zip(singles.mutations, singles.fitness) for k in m.items()}
    additive = np.array([sum(effects[k] for k in m.items()) for m in frame.mutations])
    depth = frame.depth.to_numpy(dtype=float)
    minimum = np.array([min([effects[k] for k in m.items()], default=0) for m in frame.mutations])
    squared = np.array([sum(effects[k] ** 2 for k in m.items()) for m in frame.mutations])
    return np.column_stack([additive, depth, minimum, squared])


@dataclass
class CopulaFit:
    codes: np.ndarray
    shape: tuple
    pilot_ids: np.ndarray
    pilot_fitness: np.ndarray
    z: np.ndarray
    mean: np.ndarray
    rho: float
    noise: float
    amplitude: float
    factor: tuple
    training_mean: np.ndarray
    objective: float

    @classmethod
    def fit(cls, frame, codes, shape, pilot_ids, singles, trend=True):
        y = frame.fitness.to_numpy()[pilot_ids]
        z = ndtri((rankdata(y) - 0.5) / len(y))
        # Use fixed summaries from measured singles. All scaling uses calibration rows.
        design = summary_features(frame, singles)
        centre = design[pilot_ids].mean(0)
        scale = design[pilot_ids].std(0)
        scale[scale < 1e-10] = 1
        design = (design - centre) / scale
        mean = (
            Ridge(alpha=10).fit(design[pilot_ids], z).predict(design)
            if trend
            else np.full(len(frame), z.mean())
        )
        residual = z - mean[pilot_ids]
        dist = distance(codes[pilot_ids], codes[pilot_ids])

        def objective(parameters, output=False):
            rho = 1 / (1 + np.exp(-parameters[0]))
            noise = np.exp(parameters[1])
            amplitude = np.exp(parameters[2])
            kernel = amplitude * rho**dist + noise * np.eye(len(y))
            factor = cho_factor(kernel)
            value = 0.5 * residual @ cho_solve(factor, residual) + np.log(np.diag(factor[0])).sum()
            if output:
                return rho, noise, amplitude, factor, float(value)
            return value

        fits = [
            minimize(
                objective,
                [r, np.log(0.2), np.log(max(residual.var(), 0.1))],
                method="L-BFGS-B",
                bounds=[(-4, 5), (-6, 2), (-6, 3)],
            )
            for r in (-1, 1, 3)
        ]
        best = min(fits, key=lambda f: f.fun)
        rho, noise, amplitude, factor, value = objective(best.x, True)
        return cls(
            codes,
            shape,
            pilot_ids,
            y,
            z,
            mean,
            rho,
            noise,
            amplitude,
            factor,
            mean[pilot_ids],
            value,
        )

    def draw(self, seed, field_noise=True, marginal=True):
        flat_ids = np.ravel_multi_index(self.codes.T, self.shape)
        latent_prior = np.sqrt(self.amplitude) * prior_field(self.shape, self.rho, seed)[flat_ids]
        noise = (
            np.sqrt(self.noise)
            * np.random.default_rng(seed + 913).standard_normal(np.prod(self.shape))[flat_ids]
        )
        observed_prior = latent_prior + noise
        weights = cho_solve(
            self.factor, self.z - self.training_mean - observed_prior[self.pilot_ids]
        )
        result = self.mean + latent_prior + (noise if field_noise else 0)
        for start in range(0, len(result), 8192):
            kernel = self.amplitude * self.rho ** distance(
                self.codes[start : start + 8192], self.codes[self.pilot_ids]
            )
            result[start : start + 8192] += kernel @ weights
        if field_noise:
            # The independent component is a fixed genotype effect, shared across repeated queries.
            result[self.pilot_ids] = self.z
        return result if not marginal else empirical_quantile(self.pilot_fitness, ndtr(result))


def score(frame, singles, test_ids, values, codes, meta):
    test = frame.iloc[test_ids]
    real = completion_fidelity_features(singles, test).to_numpy()
    synthetic = completion_fidelity_features(singles, test, values[test_ids]).to_numpy()
    indicators = np.column_stack(
        [codes[test_ids, j] == a for j in range(codes.shape[1]) for a in np.unique(codes[:, j])]
    )
    rows = []
    for classifier, seq in (("histogram", False), ("histogram", True), ("logistic", False)):
        x = np.column_stack([real, indicators]) if seq else real
        y = np.column_stack([synthetic, indicators]) if seq else synthetic
        result = classifier_two_sample_test(
            x,
            y,
            real_groups=np.arange(len(test)),
            synthetic_groups=np.arange(len(test)),
            classifier=classifier,
            seed=2026092900,
            n_permutations=0,
        )
        rows.append(
            dict(
                **meta,
                classifier=classifier,
                sequence=seq,
                auc=result.auc,
                w1=wasserstein_distance(test.fitness, values[test_ids]) / test.fitness.std(ddof=0),
                spearman=float(spearmanr(test.fitness, values[test_ids]).statistic),
            )
        )
    return rows


def run(name, size):
    dataset, _ = load_protein(name)
    frame = dataset.variants.reset_index(drop=True)
    singles = frame.loc[frame.depth <= 1]
    codes, shape = cube_codes(dataset)
    pilot_ids = pd.read_csv(PREVIOUS / name / "pilot_0.csv").row.to_numpy()[:size]
    test_ids = pd.read_csv(PREVIOUS / name / "test_0.csv").row.to_numpy()
    eligible = (frame.depth > 1).to_numpy()
    pair = PairFit.fit(dataset.wild_type_sequence, singles, frame.iloc[pilot_ids])
    copula = CopulaFit.fit(frame, codes, shape, pilot_ids, singles)
    rows = []
    seed = 2026092901
    models = {
        "reference": pair.predict(frame.mutations.tolist(), seed, "pilot_reference"),
        "pairs": pair.predict(frame.mutations.tolist(), seed, "learned_pairs"),
    }
    models["reference_quantile"] = remap(
        models["reference"], frame.fitness.to_numpy()[pilot_ids], eligible
    )
    models["pairs_quantile"] = remap(models["pairs"], frame.fitness.to_numpy()[pilot_ids], eligible)
    latent = copula.draw(seed, marginal=False)
    models["copula_gp"] = empirical_quantile(copula.pilot_fitness, ndtr(latent))
    models["copula_gp_quantile"] = remap(latent, copula.pilot_fitness, eligible)
    models["copula_smooth_quantile"] = remap(
        copula.draw(seed, field_noise=False, marginal=False), copula.pilot_fitness, eligible
    )
    folder = OUT / name / f"p{size}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "fit.json").write_text(
        json.dumps(
            {k: getattr(copula, k) for k in ("rho", "noise", "amplitude", "objective")}, indent=2
        )
        + "\n"
    )
    for model, values in models.items():
        values[singles.index] = singles.fitness
        np.save(folder / f"{model}.npy", values)
        rows += score(
            frame,
            singles,
            test_ids,
            values,
            codes,
            dict(protein=name, pilot_size=size, model=model),
        )
        pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
        print(
            name, size, model, [(r["auc"], r["w1"], r["spearman"]) for r in rows[-3:]], flush=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    from joblib import Parallel, delayed

    OUT.mkdir(exist_ok=True)
    (OUT / "development_protocol.json").write_text(
        json.dumps(
            {
                "status": (
                    "Development on previously evaluated calibration split zero. "
                    "These outcomes cannot confirm a selected model."
                ),
                "models": [
                    "reference_quantile",
                    "pairs_quantile",
                    "copula_gp",
                    "copula_gp_quantile",
                    "copula_smooth_quantile",
                ],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    Parallel(n_jobs=args.jobs)(delayed(run)(name, size) for size in (48, 192) for name in PROTEINS)


if __name__ == "__main__":
    main()
