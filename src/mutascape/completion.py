"""Probabilistic completion of whole-protein combinatorial landscapes.

Sparse observations determine the wild-type and single-mutant effects. An
empirical prior learned from other combinatorial landscapes supplies missing
interaction coefficients. Each posterior sample is a stable oracle defined
for every genotype in the target mutation domain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mutascape.benchmarks.datasets import LandscapeDataset

MutationKey = tuple[tuple[int, str], ...]


def _key(mutations: dict[int, str], wild_type_sequence: str) -> MutationKey:
    return tuple(
        sorted(
            (int(position), amino_acid)
            for position, amino_acid in mutations.items()
            if wild_type_sequence[int(position) - 1] != amino_acid
        )
    )


def _subsets(key: MutationKey):
    for order in range(len(key) + 1):
        yield from combinations(key, order)


def _observation_map(variants: pd.DataFrame, wild_type_sequence: str) -> dict[MutationKey, float]:
    required = {"mutations", "fitness"}
    missing = sorted(required - set(variants.columns))
    if missing:
        raise ValueError(f"observations are missing columns: {', '.join(missing)}")
    observed: dict[MutationKey, float] = {}
    for mutations, fitness in variants[["mutations", "fitness"]].itertuples(index=False, name=None):
        if not np.isfinite(fitness):
            continue
        observed[_key(mutations, wild_type_sequence)] = float(fitness)
    if () not in observed:
        raise ValueError("observations must include the wild type")
    return observed


def interaction_coefficients(
    variants: pd.DataFrame,
    wild_type_sequence: str,
    *,
    max_order: int = 3,
) -> dict[int, list[float]]:
    """Calculate Möbius interaction coefficients with observed subvariants.

    A coefficient is returned only when every lower-order subvariant required
    by the exact decomposition was measured. This prevents hidden imputation
    from entering the empirical prior.
    """
    if max_order < 1:
        raise ValueError("max_order must be positive")
    observed = _observation_map(variants, wild_type_sequence)
    coefficients: dict[int, list[float]] = {order: [] for order in range(1, max_order + 1)}

    for genotype, fitness in observed.items():
        order = len(genotype)
        if order == 0 or order > max_order:
            continue
        subsets = list(_subsets(genotype))
        if any(subset not in observed for subset in subsets):
            continue
        coefficient = 0.0
        for subset in subsets:
            coefficient += (-1.0) ** (order - len(subset)) * observed[subset]
        coefficients[order].append(float(coefficient))
    return coefficients


def _robust_scale(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    lower, upper = np.quantile(values, [0.25, 0.75])
    scale = float((upper - lower) / 1.349)
    if scale <= 1e-12:
        scale = float(np.std(values))
    return scale


@dataclass(frozen=True)
class LandscapePrior:
    """Transferable prior over normalised interaction coefficients."""

    order_location: dict[int, float]
    order_scale: dict[int, float]
    depth_location: dict[int, float]
    depth_scale: dict[int, float]
    lower_bound: float
    upper_bound: float
    degrees_of_freedom: float = 5.0
    n_landscapes: int = 0
    n_proteins: int = 0

    @property
    def max_order(self) -> int:
        return max(self.order_scale, default=1)

    @classmethod
    def fit(
        cls,
        datasets: list[LandscapeDataset] | tuple[LandscapeDataset, ...],
        *,
        max_order: int = 3,
    ) -> Self:
        """Learn interaction and assay-range priors from complete landscapes."""
        if not datasets:
            raise ValueError("at least one training landscape is required")
        if max_order < 2:
            raise ValueError("max_order must be at least two")

        by_order: dict[int, list[float]] = {order: [] for order in range(2, max_order + 1)}
        by_depth_location: dict[int, list[float]] = {}
        by_depth_scale: dict[int, list[float]] = {}
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []

        for dataset in datasets:
            coefficients = interaction_coefficients(
                dataset.variants,
                dataset.wild_type_sequence,
                max_order=max_order,
            )
            single = np.asarray(coefficients[1], dtype=np.float64)
            target_scale = _robust_scale(single)
            if target_scale <= 1e-12:
                target_scale = _robust_scale(dataset.variants["fitness"].to_numpy(dtype=float))
            if target_scale <= 1e-12:
                continue

            wt = float(dataset.variants.loc[dataset.variants["depth"] == 0, "fitness"].iloc[0])
            finite_fitness = dataset.variants["fitness"].to_numpy(dtype=float)
            finite_fitness = finite_fitness[np.isfinite(finite_fitness)]
            lower_bounds.append(float((np.min(finite_fitness) - wt) / target_scale))
            upper_bounds.append(float((np.max(finite_fitness) - wt) / target_scale))
            for order in by_order:
                by_order[order].extend(value / target_scale for value in coefficients[order])

            single_effects = {
                tuple(mutations.items())[0]: float(fitness) - wt
                for mutations, fitness in dataset.variants.loc[
                    dataset.variants["depth"] == 1, ["mutations", "fitness"]
                ].itertuples(index=False, name=None)
            }
            residuals_by_depth: dict[int, list[float]] = {}
            for mutations, fitness, depth in dataset.variants[
                ["mutations", "fitness", "depth"]
            ].itertuples(index=False, name=None):
                if depth < 2 or not all(item in single_effects for item in mutations.items()):
                    continue
                additive = wt + sum(single_effects[item] for item in mutations.items())
                residuals_by_depth.setdefault(int(depth), []).append(
                    (float(fitness) - additive) / target_scale
                )
            for depth, residual_values in residuals_by_depth.items():
                residuals = np.asarray(residual_values, dtype=np.float64)
                by_depth_location.setdefault(depth, []).append(float(np.median(residuals)))
                by_depth_scale.setdefault(depth, []).append(_robust_scale(residuals))

        locations: dict[int, float] = {}
        scales: dict[int, float] = {}
        previous_scale = 0.25
        for order in range(2, max_order + 1):
            values = np.asarray(by_order[order], dtype=np.float64)
            if values.size:
                locations[order] = float(np.median(values))
                scale = _robust_scale(values)
                scales[order] = max(scale, 1e-6)
                previous_scale = scales[order]
            else:
                locations[order] = 0.0
                scales[order] = previous_scale * 0.5

        depth_locations = {
            depth: float(np.median(values)) for depth, values in by_depth_location.items()
        }
        depth_scales = {}
        for depth, within_scales in by_depth_scale.items():
            locations_at_depth = np.asarray(by_depth_location[depth], dtype=np.float64)
            within = np.asarray(within_scales, dtype=np.float64)
            between_scale = (
                float(np.std(locations_at_depth, ddof=1)) if locations_at_depth.size > 1 else 0.0
            )
            posterior_predictive_scale = sqrt(float(np.mean(within**2)) + between_scale**2)
            depth_scales[depth] = max(posterior_predictive_scale, 1e-6)

        lower_bound = float(np.median(lower_bounds)) if lower_bounds else -5.0
        upper_bound = float(np.median(upper_bounds)) if upper_bounds else 5.0
        if lower_bound >= upper_bound:
            lower_bound, upper_bound = -5.0, 5.0
        return cls(
            order_location=locations,
            order_scale=scales,
            depth_location=depth_locations,
            depth_scale=depth_scales,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_landscapes=len(datasets),
            n_proteins=len({dataset.protein for dataset in datasets}),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible prior parameters."""
        return {
            "order_location": {str(k): v for k, v in self.order_location.items()},
            "order_scale": {str(k): v for k, v in self.order_scale.items()},
            "depth_location": {str(k): v for k, v in self.depth_location.items()},
            "depth_scale": {str(k): v for k, v in self.depth_scale.items()},
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "degrees_of_freedom": self.degrees_of_freedom,
            "n_landscapes": self.n_landscapes,
            "n_proteins": self.n_proteins,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> Self:
        """Restore a prior written by :meth:`to_dict`."""
        locations = dict(values["order_location"])
        scales = dict(values["order_scale"])
        depth_locations = dict(values.get("depth_location", {}))
        depth_scales = dict(values.get("depth_scale", {}))
        return cls(
            order_location={int(k): float(v) for k, v in locations.items()},
            order_scale={int(k): float(v) for k, v in scales.items()},
            depth_location={int(k): float(v) for k, v in depth_locations.items()},
            depth_scale={int(k): float(v) for k, v in depth_scales.items()},
            lower_bound=float(values["lower_bound"]),
            upper_bound=float(values["upper_bound"]),
            degrees_of_freedom=float(values.get("degrees_of_freedom", 5.0)),
            n_landscapes=int(values.get("n_landscapes", 0)),
            n_proteins=int(values.get("n_proteins", 0)),
        )


@dataclass(frozen=True)
class PosteriorLandscape:
    """A sparse DMS dataset conditioned on a transferable landscape prior."""

    wild_type_sequence: str
    observations: dict[MutationKey, float]
    coefficients: dict[MutationKey, float]
    single_effects: dict[tuple[int, str], float]
    prior: LandscapePrior
    target_scale: float

    @classmethod
    def condition(
        cls,
        wild_type_sequence: str,
        observations: pd.DataFrame,
        prior: LandscapePrior,
    ) -> Self:
        """Condition the prior on WT, singles, and any supplied combinations."""
        observed = _observation_map(observations, wild_type_sequence)
        observed_coefficients = interaction_coefficients(
            observations,
            wild_type_sequence,
            max_order=prior.max_order,
        )
        coefficient_map: dict[MutationKey, float] = {}
        for genotype in observed:
            order = len(genotype)
            if order < 2 or order > prior.max_order:
                continue
            subsets = list(_subsets(genotype))
            if any(subset not in observed for subset in subsets):
                continue
            coefficient_map[genotype] = sum(
                (-1.0) ** (order - len(subset)) * observed[subset] for subset in subsets
            )

        wt = observed[()]
        single_effects = {
            genotype[0]: fitness - wt
            for genotype, fitness in observed.items()
            if len(genotype) == 1
        }
        target_scale = _robust_scale(np.asarray(list(single_effects.values()), dtype=np.float64))
        if target_scale <= 1e-12:
            target_scale = 1.0
        if not any(observed_coefficients[order] for order in observed_coefficients):
            raise ValueError("observations must contain at least one single mutant")
        return cls(
            wild_type_sequence=wild_type_sequence,
            observations=observed,
            coefficients=coefficient_map,
            single_effects=single_effects,
            prior=prior,
            target_scale=target_scale,
        )

    def sample(self, seed: int, *, clip_to_assay_range: bool = False) -> CompletedLandscape:
        """Return one fixed oracle realisation.

        Clipping is optional because the latent fitness oracle and a censored
        assay readout represent different objects.
        """
        return CompletedLandscape(
            self,
            seed=int(seed),
            clip_to_assay_range=clip_to_assay_range,
        )

    def predict(
        self,
        variants: list[dict[int, str]],
        *,
        n_samples: int = 100,
        seed: int = 0,
        interval: float = 0.8,
    ) -> pd.DataFrame:
        """Return posterior mean, spread, and interval for each genotype."""
        if n_samples < 2:
            raise ValueError("n_samples must be at least two")
        if not 0.0 < interval < 1.0:
            raise ValueError("interval must lie between zero and one")
        values = np.vstack(
            [self.sample(seed + index).evaluate_batch(variants) for index in range(n_samples)]
        )
        tail = (1.0 - interval) / 2.0
        return pd.DataFrame(
            {
                "mean": np.mean(values, axis=0),
                "std": np.std(values, axis=0, ddof=1),
                "lower": np.quantile(values, tail, axis=0),
                "upper": np.quantile(values, 1.0 - tail, axis=0),
            }
        )


@dataclass(frozen=True)
class CompletedLandscape:
    """One stable, queryable sample from a posterior landscape."""

    posterior: PosteriorLandscape
    seed: int
    clip_to_assay_range: bool = False

    @staticmethod
    def _at_depth(values: dict[int, float], depth: int, fallback: float) -> float:
        if not values:
            return fallback
        depths = np.asarray(sorted(values), dtype=np.float64)
        parameters = np.asarray([values[int(item)] for item in depths], dtype=np.float64)
        return float(np.interp(depth, depths, parameters))

    def _draw_coefficient(self, interaction: MutationKey) -> float:
        observed = self.posterior.coefficients.get(interaction)
        if observed is not None:
            return observed
        order = len(interaction)
        prior = self.posterior.prior
        payload = f"{self.seed}|{interaction!r}".encode()
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        draw_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(draw_seed)
        df = prior.degrees_of_freedom
        unit_variance_t = float(rng.standard_t(df) / sqrt(df / (df - 2.0)))
        # The depth prior carries systematic residual location. Centring each
        # coefficient avoids multiplying an order median by every subset.
        return self.posterior.target_scale * prior.order_scale[order] * unit_variance_t

    def evaluate(self, mutations: dict[int, str]) -> float:
        """Evaluate any genotype in the whole-protein mutation domain."""
        posterior = self.posterior
        genotype = _key(mutations, posterior.wild_type_sequence)
        if genotype in posterior.observations:
            return posterior.observations[genotype]

        missing_singles = [
            mutation for mutation in genotype if mutation not in posterior.single_effects
        ]
        if missing_singles:
            labels = ", ".join(
                f"{position}{amino_acid}" for position, amino_acid in missing_singles
            )
            raise ValueError(f"missing single-mutant measurements for: {labels}")

        value = posterior.observations[()] + sum(
            posterior.single_effects[mutation] for mutation in genotype
        )
        depth = len(genotype)
        depth_location = self._at_depth(posterior.prior.depth_location, depth, 0.0)
        value += posterior.target_scale * depth_location

        observed_interaction = 0.0
        sampled_interaction = 0.0
        sampled_variance = 0.0
        for order in range(2, min(len(genotype), posterior.prior.max_order) + 1):
            for group in combinations(genotype, order):
                interaction = tuple(group)
                coefficient = posterior.coefficients.get(interaction)
                if coefficient is None:
                    sampled_interaction += self._draw_coefficient(interaction)
                    sampled_variance += posterior.prior.order_scale[order] ** 2
                else:
                    observed_interaction += coefficient

        desired_scale = self._at_depth(
            posterior.prior.depth_scale,
            depth,
            sqrt(sampled_variance),
        )
        if sampled_variance > 0:
            sampled_interaction *= desired_scale / sqrt(sampled_variance)
        value += observed_interaction + sampled_interaction

        if self.clip_to_assay_range:
            lower = min(
                min(posterior.observations.values()),
                posterior.observations[()] + posterior.prior.lower_bound * posterior.target_scale,
            )
            upper = max(
                max(posterior.observations.values()),
                posterior.observations[()] + posterior.prior.upper_bound * posterior.target_scale,
            )
            if lower < upper:
                value = float(np.clip(value, lower, upper))
        return float(value)

    def evaluate_batch(self, variants: list[dict[int, str]]) -> NDArray[np.float64]:
        """Evaluate a sequence of genotypes."""
        return np.asarray([self.evaluate(variant) for variant in variants], dtype=np.float64)
