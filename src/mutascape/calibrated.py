"""Landscapes calibrated to measured combinations as well as singles.

These models need a pilot of measured multi-mutants. The quantile model maps
each draw onto the measured combination-fitness distribution. The burden model
underneath it fits depth terms to the same pilot.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations, islice, product
from typing import Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mutascape.completion import dms_pairwise_unit_coefficient
from mutascape.domain import MutationDomain
from mutascape.mutant import VariantLike, parse_variant
from mutascape.nonlinearity import GlobalEpistasisSigmoid

DEFAULT_REFERENCE_SIZE = 100_000


def _sample_reference_batches(
    domain: MutationDomain, size: int, seed: int, batch_size: int = 8192
) -> Iterator[list[tuple[tuple[int, str], ...]]]:
    """Sample uniformly with replacement from supported multi-mutant genotypes.

    Independent uniform residue choices give uniform genotypes even when sites
    have different alphabet sizes. Reject WT and single-site variants without
    balancing mutation depths or using experimental combination fitness.
    """
    rng = np.random.default_rng(seed)
    positions = domain.positions
    alternatives = [domain.substitutions_at(p) for p in positions]
    sizes = np.asarray([len(aa) + 1 for aa in alternatives])
    remaining = size
    while remaining:
        codes = rng.integers(sizes, size=(min(batch_size, remaining), len(positions)))
        codes = codes[np.count_nonzero(codes, axis=1) > 1]
        batch = [
            tuple(
                (position, residues[code - 1])
                for position, residues, code in zip(positions, alternatives, row)
                if code
            )
            for row in codes
        ]
        if batch:
            remaining -= len(batch)
            yield batch


def _design(keys, effects):
    depth = np.asarray([len(key) for key in keys])
    additive = np.asarray([sum(effects[k] for k in key) for key in keys])
    extra = np.maximum(depth - 1, 0)
    return additive, np.column_stack([extra, additive * extra / np.maximum(depth, 1)])


@dataclass(frozen=True)
class PilotCalibratedLandscape:
    """Fit a sigmoid burden model from same-assay singles and a combination pilot.

    Pilot measurements fit aggregate terms and are not imposed as exact
    anchors.
    """

    domain: MutationDomain
    anchors: dict
    effects: dict
    coefficients: NDArray[np.float64]
    pair_scale: float
    readout: GlobalEpistasisSigmoid
    calibration_keys: tuple

    @classmethod
    def fit(cls, wild_type_sequence: str, singles: pd.DataFrame, pilot: pd.DataFrame) -> Self:
        """Fit using only the supplied wild type, singles, and calibration variants."""
        if singles.empty or pilot.empty:
            raise ValueError("singles and pilot must contain measurements")
        anchors = {}
        for mutations, fitness in singles[["mutations", "fitness"]].itertuples(
            index=False, name=None
        ):
            key = tuple(
                sorted(parse_variant(mutations, wild_type_sequence=wild_type_sequence).items())
            )
            if len(key) > 1:
                raise ValueError("single-mutant input contains a combination")
            if not np.isfinite(fitness):
                raise ValueError("anchor fitness must be finite")
            if key in anchors:
                raise ValueError("duplicate anchor genotype")
            anchors[key] = float(fitness)
        if () not in anchors:
            raise ValueError("single-mutant input must include wild type")
        substitutions = tuple(key[0] for key in anchors if key)
        domain = MutationDomain(wild_type_sequence, substitutions)
        keys = tuple(tuple(sorted(domain.parse(m).items())) for m in pilot.mutations)
        if any(len(k) < 2 for k in keys):
            raise ValueError("pilot must contain only multi-mutants")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate pilot genotype")
        y = pilot.fitness.to_numpy(dtype=float) - anchors[()]
        if not np.isfinite(y).all():
            raise ValueError("pilot fitness must be finite")
        single_effects = {k[0]: v - anchors[()] for k, v in anchors.items() if k}
        observed = np.r_[y, list(single_effects.values()), 0.0]
        span = float(np.ptp(observed))
        if span <= 1e-12:
            raise ValueError("calibration fitness has no variation")
        readout = GlobalEpistasisSigmoid(
            floor=float(observed.min() - 0.05 * span),
            ceiling=float(observed.max() + 0.05 * span),
            scale=1.0,
        )
        effects = {k: float(readout.inverse(v)) for k, v in single_effects.items()}
        additive, design = _design(keys, effects)
        if len(keys) < 3 or np.linalg.matrix_rank(design) < 2:
            raise ValueError("pilot must identify both burden terms")
        model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1, fit_intercept=False))
        model.fit(design, readout.inverse(y) - additive)
        residual = readout.inverse(y) - additive - model.predict(design)
        n_pairs = np.asarray([len(k) * (len(k) - 1) / 2 for k in keys])
        scale = float(np.sqrt(np.sum(residual**2) / np.sum(n_pairs)))
        return cls(
            domain, anchors, effects, model[-1].coef_ / model[0].scale_, scale, readout, keys
        )

    def to_dict(self) -> dict:
        """Return fitted parameters and input identities for model restoration."""
        return {
            "model": "pilot-burden-v1",
            "wild_type_sequence": self.domain.wild_type_sequence,
            "anchors": [
                {"mutations": list(key), "fitness": value} for key, value in self.anchors.items()
            ],
            "effects": [
                {"position": p, "residue": aa, "effect": value}
                for (p, aa), value in self.effects.items()
            ],
            "coefficients": self.coefficients.tolist(),
            "pair_scale": self.pair_scale,
            "readout": {
                "floor": self.readout.floor,
                "ceiling": self.readout.ceiling,
                "scale": self.readout.scale,
            },
            "calibration_keys": [list(key) for key in self.calibration_keys],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Self:
        """Restore a fitted model without refitting calibration data."""
        if payload.get("model") != "pilot-burden-v1":
            raise ValueError("unsupported pilot model version")
        anchors = {
            tuple((int(p), aa) for p, aa in row["mutations"]): float(row["fitness"])
            for row in payload["anchors"]
        }
        effects = {
            (int(row["position"]), row["residue"]): float(row["effect"])
            for row in payload["effects"]
        }
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        pair_scale = float(payload["pair_scale"])
        if (
            coefficients.shape != (2,)
            or not np.isfinite(coefficients).all()
            or not np.isfinite(pair_scale)
            or pair_scale < 0
            or () not in anchors
        ):
            raise ValueError("invalid pilot model parameters")
        domain = MutationDomain(payload["wild_type_sequence"], tuple(effects))
        keys = tuple(tuple((int(p), aa) for p, aa in key) for key in payload["calibration_keys"])
        return cls(
            domain,
            anchors,
            effects,
            coefficients,
            pair_scale,
            GlobalEpistasisSigmoid(**payload["readout"]),
            keys,
        )

    def draw(self, seed: int) -> PilotLandscapeDraw:
        """Return a reproducible oracle with one shared pair field."""
        return PilotLandscapeDraw(self, int(seed))


@dataclass(frozen=True)
class PilotLandscapeDraw:
    """One fixed draw from the pilot-calibrated burden model."""

    generator: PilotCalibratedLandscape
    seed: int

    @property
    def domain(self) -> MutationDomain:
        return self.generator.domain

    def evaluate_batch(self, variants: list[VariantLike]) -> NDArray[np.float64]:
        """Evaluate supported variants in input order, preserving measured singles."""
        g = self.generator
        keys = [tuple(sorted(g.domain.parse(v).items())) for v in variants]
        if not keys:
            return np.empty(0, dtype=float)
        additive, design = _design(keys, g.effects)
        noise = np.asarray(
            [
                sum(
                    dms_pairwise_unit_coefficient(a, b, g.domain.wild_type_sequence, self.seed) / 2
                    for a, b in combinations(key, 2)
                )
                for key in keys
            ]
        )
        values = g.anchors[()] + g.readout.forward(
            additive + design @ g.coefficients + g.pair_scale * noise
        )
        for index, key in enumerate(keys):
            if key in g.anchors:
                values[index] = g.anchors[key]
        return np.asarray(values, dtype=float)

    def evaluate(self, variant: VariantLike) -> float:
        """Evaluate one supported genotype."""
        return float(self.evaluate_batch([variant])[0])

    def __getitem__(self, variant: VariantLike) -> float:
        return self.evaluate(variant)


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
    A fixed uniform reference sample estimates the generated score distribution.
    Small search spaces use every supported multi-mutant instead.
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

    def draw(
        self,
        seed: int,
        *,
        reference_size: int | None = DEFAULT_REFERENCE_SIZE,
        reference_seed: int = 0,
        maximum_variants: int | None = None,
    ) -> QuantileLandscapeDraw:
        """Construct a fixed oracle using a uniform score reference sample.

        At most ``reference_size`` multi-mutant scores define the percentile map.
        Larger spaces use that many uniform samples with replacement. Smaller
        spaces use all multi-mutants exactly. Set ``reference_size=None`` to
        request exact enumeration explicitly, including for archived results.

        ``reference_seed`` fixes reference genotypes independently of the pair
        interaction seed. No further experimental measurements are needed.
        Keep both seeds and the reference size to reproduce a sampled landscape.
        There is no default search-space limit. ``maximum_variants`` remains an
        optional caller-specified limit for compatibility with earlier code.
        """
        for name, value, minimum in (
            ("reference_size", reference_size, 2),
            ("reference_seed", reference_seed, 0),
            ("maximum_variants", maximum_variants, 1),
        ):
            if value is None and name != "reference_seed":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
        g = self.reference
        total = g.domain.count()
        if maximum_variants is not None and total > maximum_variants:
            raise ValueError("search space exceeds the requested maximum_variants")
        multi_count = total - len(g.effects) - 1
        exact = reference_size is None or multi_count <= reference_size
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
        if exact:
            variants = (
                tuple(
                    (p, aa)
                    for p, aa in zip(positions, residues)
                    if aa != g.domain.wild_type_sequence[p - 1]
                )
                for residues in product(*alphabets)
            )
            batches = iter(lambda: list(islice(variants, 8192)), [])
        else:
            batches = _sample_reference_batches(g.domain, reference_size, reference_seed)
        scores = []
        for batch in batches:
            batch = [key for key in batch if len(key) > 1]
            if batch:
                scores.append(oracle.base_values(batch))
        all_scores = np.sort(np.concatenate(scores))
        unique, start, counts = np.unique(all_scores, return_index=True, return_counts=True)
        probabilities = (start + counts / 2) / len(all_scores)
        return QuantileLandscapeDraw(
            self,
            keys,
            weights,
            unique,
            probabilities,
            reference_mode="exact" if exact else "sampled",
            reference_size=len(all_scores),
            reference_seed=None if exact else int(reference_seed),
        )


@dataclass(frozen=True)
class QuantileLandscapeDraw:
    """One fixed landscape with a calibration-derived fitness distribution."""

    generator: QuantileCalibratedLandscape
    keys: list
    weights: NDArray[np.float64]
    score_knots: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    reference_mode: Literal["exact", "sampled"] = "exact"
    reference_size: int = 0
    reference_seed: int | None = None

    @property
    def domain(self) -> MutationDomain:
        return self.generator.reference.domain

    def is_observed(self, variant: VariantLike) -> bool:
        """Return whether this genotype is measured wild type or a measured single."""
        return tuple(sorted(self.domain.parse(variant).items())) in self.generator.reference.anchors

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
        probabilities = np.interp(values, self.score_knots, self.probabilities, left=0.0, right=1.0)
        result = empirical_quantiles(self.generator.pilot_fitness, probabilities)
        for i, key in enumerate(keys):
            if key in g.anchors:
                result[i] = g.anchors[key]
        return result

    def evaluate(self, variant: VariantLike) -> float:
        """Evaluate one supported variant."""
        return float(self.evaluate_batch([variant])[0])

    def score(self, variant: VariantLike) -> float:
        """Score one variant supplied as a label, mapping, or ``Mutant``."""
        return self.evaluate(variant)

    def score_many(self, variants: list[VariantLike]) -> pd.Series:
        """Score variants and return values indexed by canonical mutation labels."""
        mutants = [self.domain.parse(variant) for variant in variants]
        return pd.Series(
            self.evaluate_batch(mutants),
            index=[mutant.label for mutant in mutants],
            name="fitness",
        )

    def __getitem__(self, variant: VariantLike) -> float:
        return self.evaluate(variant)
