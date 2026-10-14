"""Condition a random pair field on combination residuals for a research check."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from mutascape.calibrated import PilotCalibratedLandscape, _design
from mutascape.model import DMSLandscape


def pair_kernel(first: list[dict], second: list[dict]) -> np.ndarray:
    """Return the covariance of the WT-corrected, unit-variance pair field."""
    total = np.zeros((len(first), len(second)))
    squares = np.zeros_like(total)
    positions = sorted({p for m in first for p in m} & {p for m in second for p in m})
    for position in positions:
        a = np.asarray([m.get(position, "") for m in first])
        b = np.asarray([m.get(position, "") for m in second])
        present = (a[:, None] != "") & (b[None, :] != "")
        term = present * (1.0 + (a[:, None] == b[None, :])) / 2
        total += term
        squares += term**2
    return (total**2 - squares) / 2


def latent_mean(model: PilotCalibratedLandscape, variants: list[dict]) -> np.ndarray:
    keys = [tuple(sorted(m.items())) for m in variants]
    additive, design = _design(keys, model.effects)
    return additive + design @ model.coefficients


def pair_field(model: PilotCalibratedLandscape, variants: list[dict], seed: int) -> np.ndarray:
    """Evaluate the same random field used by the combination model."""
    oracle = DMSLandscape(
        model.domain,
        model.anchors,
        model.effects,
        model.pair_scale,
        seed,
        1.0,
        "linear",
        None,
        None,
    )
    values = oracle.evaluate_batch(variants) - model.anchors[()]
    additive = np.asarray([sum(model.effects[k] for k in m.items()) for m in variants])
    values -= additive
    values[np.asarray([len(m) <= 1 for m in variants])] = 0
    return values


@dataclass
class ResearchPairPosterior:
    """Keep the existing mean model and learn directions for its random pairs."""

    model: PilotCalibratedLandscape
    variants: list[dict]
    residual: np.ndarray
    factor: tuple
    noise_variance: float
    noise_ratio: float

    @classmethod
    def fit(cls, model: PilotCalibratedLandscape, pilot):
        variants = pilot.mutations.tolist()
        residual = model.readout.inverse(pilot.fitness.to_numpy() - model.anchors[()])
        residual -= latent_mean(model, variants)
        covariance = model.pair_scale**2 * pair_kernel(variants, variants)
        scale = max(float(np.diag(covariance).mean()), 1e-8)
        candidates = []
        for ratio in (0.01, 0.1, 1.0, 10.0):
            variance = ratio * scale
            factor = cho_factor(covariance + variance * np.eye(len(variants)), lower=True)
            nll = residual @ cho_solve(factor, residual) / 2
            nll += np.log(np.diag(factor[0])).sum()
            candidates.append((nll, ratio, variance, factor))
        _, ratio, variance, factor = min(candidates, key=lambda row: row[0])
        return cls(model, variants, residual, factor, variance, ratio)

    def latent(self, variants: list[dict], seed: int | None) -> np.ndarray:
        covariance = self.model.pair_scale**2 * pair_kernel(variants, self.variants)
        mean = latent_mean(self.model, variants)
        if seed is None:
            return mean + covariance @ cho_solve(self.factor, self.residual)
        prior = pair_field(self.model, variants, seed)
        observed_prior = pair_field(self.model, self.variants, seed)
        noise = np.random.default_rng(seed + 1943).normal(
            scale=np.sqrt(self.noise_variance), size=len(self.variants)
        )
        adjustment = cho_solve(self.factor, self.residual - observed_prior - noise)
        return mean + prior + covariance @ adjustment
