"""Calibrate a fixed pilot landscape to the measured combination distribution."""

from dataclasses import dataclass
from itertools import combinations, islice, product
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.completion import dms_pairwise_unit_coefficient
from mutascape.experimental.pilot import PilotCalibratedLandscape, _design
from mutascape.mutant import VariantLike


def empirical_quantiles(sample: NDArray, probabilities: NDArray) -> NDArray[np.float64]:
    """Interpolate measured quantiles with linear tails and repeated-value intervals."""
    values = np.sort(np.asarray(sample, dtype=float))
    probabilities = np.asarray(probabilities, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("quantile calibration needs at least two finite measurements")
    unique, start, counts = np.unique(values, return_index=True, return_counts=True)
    knots, ordinates = [], []
    for value, first, count in zip(unique, start, counts):
        positions = (
            [(first + 0.5) / len(values)]
            if count == 1
            else [first / len(values), (first + count) / len(values)]
        )
        knots.extend(positions)
        ordinates.extend([value] * len(positions))
    knots, ordinates = np.asarray(knots), np.asarray(ordinates)
    if len(knots) == 1:
        return np.full_like(probabilities, ordinates[0])
    result = np.interp(probabilities, knots, ordinates)
    low = probabilities < knots[0]
    high = probabilities > knots[-1]
    result[low] = ordinates[0] + (probabilities[low] - knots[0]) * (ordinates[1] - ordinates[0]) / (
        knots[1] - knots[0]
    )
    result[high] = ordinates[-1] + (probabilities[high] - knots[-1]) * (
        ordinates[-1] - ordinates[-2]
    ) / (knots[-1] - knots[-2])
    return result


@dataclass(frozen=True)
class QuantileCalibratedLandscape:
    """Fit a pilot model and calibrate its combination-fitness distribution.

    Calibration combinations must sample the intended library uniformly.
    Each draw enumerates the supported library to fix its score distribution.
    Measured wild type and singles retain their original values.
    """

    reference: PilotCalibratedLandscape
    pilot_fitness: NDArray[np.float64]

    @classmethod
    def fit(cls, wild_type_sequence: str, singles: pd.DataFrame, pilot: pd.DataFrame) -> Self:
        """Fit using measured wild type, singles, and a uniform combination pilot."""
        return cls(
            PilotCalibratedLandscape.fit(wild_type_sequence, singles, pilot),
            np.sort(pilot.fitness.to_numpy(dtype=float)),
        )

    def to_dict(self) -> dict:
        """Return the fitted model without its generated score distributions."""
        return {
            "model": "quantile-pilot-v1",
            "reference": self.reference.to_dict(),
            "pilot_fitness": self.pilot_fitness.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Self:
        """Restore a fitted model without calibration refits."""
        if payload.get("model") != "quantile-pilot-v1":
            raise ValueError("unsupported quantile model version")
        values = np.asarray(payload["pilot_fitness"], dtype=float)
        if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
            raise ValueError("invalid calibration fitness")
        return cls(PilotCalibratedLandscape.from_dict(payload["reference"]), np.sort(values))

    def draw(self, seed: int, *, maximum_variants: int = 1_000_000) -> QuantileLandscapeDraw:
        """Construct one fixed oracle after enumerating its supported library."""
        g = self.reference
        if g.domain.count() > maximum_variants:
            raise ValueError("library exceeds maximum_variants for exact distribution calibration")
        keys = sorted(g.effects)
        weights = np.zeros((len(keys), len(keys)))
        for i, j in combinations(range(len(keys)), 2):
            if keys[i][0] != keys[j][0]:
                value = (
                    dms_pairwise_unit_coefficient(
                        keys[i], keys[j], g.domain.wild_type_sequence, int(seed)
                    )
                    / 4
                )
                weights[i, j] = weights[j, i] = value
        oracle = QuantileLandscapeDraw(self, keys, weights, np.empty(0), np.empty(0))
        positions = g.domain.positions
        alphabets = [
            (g.domain.wild_type_sequence[p - 1], *g.domain.substitutions_at(p)) for p in positions
        ]
        variants = (
            tuple(
                (p, aa)
                for p, aa in zip(positions, residues)
                if aa != g.domain.wild_type_sequence[p - 1]
            )
            for residues in product(*alphabets)
        )
        scores = []
        while batch := list(islice(variants, 8192)):
            batch = [key for key in batch if len(key) > 1]
            if batch:
                scores.append(oracle.base_values(batch))
        all_scores = np.sort(np.concatenate(scores))
        unique, start, counts = np.unique(all_scores, return_index=True, return_counts=True)
        probabilities = (start + counts / 2) / len(all_scores)
        return QuantileLandscapeDraw(self, keys, weights, unique, probabilities)


@dataclass(frozen=True)
class QuantileLandscapeDraw:
    """One fixed landscape with a calibration-derived fitness distribution."""

    generator: QuantileCalibratedLandscape
    keys: list
    weights: NDArray[np.float64]
    score_knots: NDArray[np.float64]
    probabilities: NDArray[np.float64]

    def base_values(self, variants: list[tuple]) -> NDArray[np.float64]:
        g = self.generator.reference
        additive, design = _design(variants, g.effects)
        lookup = {key: i for i, key in enumerate(self.keys)}
        indicators = np.zeros((len(variants), len(self.keys)))
        for row, variant in enumerate(variants):
            for key in variant:
                indicators[row, lookup[key]] = 1
        interactions = np.sum((indicators @ self.weights) * indicators, axis=1)
        return g.anchors[()] + g.readout.forward(
            additive + design @ g.coefficients + g.pair_scale * interactions
        )

    def evaluate_batch(self, variants: list[VariantLike]) -> NDArray[np.float64]:
        """Evaluate variants without changing the draw or its calibration map."""
        g = self.generator.reference
        keys = [tuple(sorted(g.domain.parse(v).items())) for v in variants]
        if not keys:
            return np.empty(0)
        values = self.base_values(keys)
        probabilities = np.interp(values, self.score_knots, self.probabilities)
        result = empirical_quantiles(self.generator.pilot_fitness, probabilities)
        for i, key in enumerate(keys):
            if key in g.anchors:
                result[i] = g.anchors[key]
        return result

    def evaluate(self, variant: VariantLike) -> float:
        """Evaluate one supported variant."""
        return float(self.evaluate_batch([variant])[0])

    def __getitem__(self, variant: VariantLike) -> float:
        return self.evaluate(variant)
