"""Probabilistic completion of whole-protein combinatorial landscapes.

Sparse observations determine the wild-type and single-mutant effects. An
empirical prior learned from other combinatorial landscapes supplies missing
interaction coefficients. Each posterior sample is a stable oracle defined
for every genotype in the target mutation domain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import sqrt
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.nonlinearity import GlobalEpistasisSigmoid, NonLinearity

if TYPE_CHECKING:
    from mutascape.benchmarks.datasets import LandscapeDataset

MutationKey = tuple[tuple[int, str], ...]

# Below this many contributing landscapes a depth bin cannot support a transferred shift.
_MIN_DEPTH_LANDSCAPES = 3

# Width of the band, as a fraction of the assay range, counted as sitting on the floor.
_FLOOR_BAND = 0.02


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


def _hash_normal(key: object, seed: int) -> float:
    """Standard normal draw pinned to a key, so a genotype always gets the same one."""
    payload = f"{seed!r}|{key!r}".encode()
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return float(np.random.default_rng(int.from_bytes(digest[:8], "little")).normal())


def _single_effect_scale(values: NDArray[np.float64]) -> float:
    """Normalise one landscape's coefficients onto a transferable scale.

    Standard deviation beats the interquartile range because most single
    mutations are neutral. The middle half of a DMS single-effect distribution
    sits near zero, so dividing by its width inflates the transferred
    coefficients of the narrowest assays by an order of magnitude, and the
    inflated landscapes then dominate every pooled prior.
    """
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _robust_scale(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    lower, upper = np.quantile(values, [0.25, 0.75])
    scale = float((upper - lower) / 1.349)
    if scale <= 1e-12:
        scale = float(np.std(values))
    return scale


def _training_readout(
    single_effects: dict[tuple[int, str], float],
    centred_fitness: NDArray[np.float64],
    *,
    margin: float = 0.05,
) -> NonLinearity | None:
    """Fit a readout for a training landscape from its full observed range.

    A training landscape is measured everywhere, so its own combinations give
    the assay floor and ceiling directly. A target holds those out and takes
    the pooled range from the prior instead.
    """
    if len(single_effects) < 2 or centred_fitness.size == 0:
        return None
    low = float(np.min(centred_fitness))
    high = float(np.max(centred_fitness))
    span = high - low
    if not np.isfinite(span) or span <= 1e-12:
        return None
    by_position: dict[int, dict[str, float]] = {}
    for (position, amino_acid), effect in single_effects.items():
        by_position.setdefault(int(position), {})[amino_acid] = float(effect)
    try:
        return GlobalEpistasisSigmoid.fit_from_singles(
            by_position,
            margin=margin,
            floor=min(low - margin * span, -margin * span),
            ceiling=max(high + margin * span, margin * span),
        )
    except (ValueError, RuntimeError):
        return None


def _latent_coefficients(
    variants: pd.DataFrame,
    wild_type_sequence: str,
    wild_type_fitness: float,
    transform,
    orders: tuple[int, ...],
) -> dict[int, list[float]]:
    """Mobius coefficients per order, taken after the inverse readout.

    One pass over the observations covers every requested order. The largest
    training landscapes hold over a hundred thousand variants, so rebuilding the
    map per order dominates the fit.
    """
    observed = _observation_map(variants, wild_type_sequence)
    keys = list(observed)
    centred = np.asarray([observed[key] for key in keys], dtype=np.float64) - wild_type_fitness
    latent = dict(zip(keys, np.asarray(transform(centred), dtype=np.float64)))
    wanted = set(orders)
    values: dict[int, list[float]] = {order: [] for order in orders}
    for genotype in keys:
        order = len(genotype)
        if order not in wanted:
            continue
        subsets = list(_subsets(genotype))
        if any(subset not in latent for subset in subsets):
            continue
        values[order].append(
            float(sum((-1.0) ** (order - len(subset)) * latent[subset] for subset in subsets))
        )
    return values


@dataclass(frozen=True)
class LandscapePrior:
    """Transferable prior over normalised interaction coefficients."""

    order_location: dict[int, float]
    order_scale: dict[int, float]
    depth_location: dict[int, float]
    depth_scale: dict[int, float]
    lower_bound: float
    upper_bound: float
    depth_between: dict[int, float] = field(default_factory=dict)
    depth_shrinkage: dict[int, float] = field(default_factory=dict)
    texture_scale: float = 0.0
    texture_order: int = 3
    texture_iid_fraction: float = 0.3
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
        shrink_depth: bool = True,
        global_epistasis: bool = True,
    ) -> Self:
        """Learn interaction and assay-range priors from complete landscapes.

        ``shrink_depth`` exists so the depth term can be ablated. Turning it off
        applies the pooled residual at full strength to every held-out protein.
        ``global_epistasis`` must match the setting used when conditioning a
        target, because it selects the space the coefficients are measured in.
        """
        if not datasets:
            raise ValueError("at least one training landscape is required")
        if max_order < 2:
            raise ValueError("max_order must be at least two")

        by_order: dict[int, list[float]] = {order: [] for order in range(2, max_order + 1)}
        by_depth_location: dict[int, list[float]] = {}
        by_depth_scale: dict[int, list[float]] = {}
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        texture_scales: list[float] = []

        for dataset in datasets:
            coefficients = interaction_coefficients(
                dataset.variants,
                dataset.wild_type_sequence,
                max_order=max_order,
            )
            single = np.asarray(coefficients[1], dtype=np.float64)
            target_scale = _single_effect_scale(single)
            if target_scale <= 1e-12:
                target_scale = _single_effect_scale(
                    dataset.variants["fitness"].to_numpy(dtype=float)
                )
            if target_scale <= 1e-12:
                continue

            wt = float(dataset.variants.loc[dataset.variants["depth"] == 0, "fitness"].iloc[0])
            finite_fitness = dataset.variants["fitness"].to_numpy(dtype=float)
            finite_fitness = finite_fitness[np.isfinite(finite_fitness)]
            lower_bounds.append(float((np.min(finite_fitness) - wt) / target_scale))
            upper_bounds.append(float((np.max(finite_fitness) - wt) / target_scale))

            single_effects = {
                tuple(mutations.items())[0]: float(fitness) - wt
                for mutations, fitness in dataset.variants.loc[
                    dataset.variants["depth"] == 1, ["mutations", "fitness"]
                ].itertuples(index=False, name=None)
            }

            # Coefficients must be learned in the space they are later applied.
            # A readout compresses hard near its floor, so a spread measured on
            # raw fitness understates the latent variation that produced it.
            readout = None
            if global_epistasis:
                readout = _training_readout(single_effects, finite_fitness - wt)
            if readout is not None:
                transform = readout.inverse
                effects = {key: float(transform(value)) for key, value in single_effects.items()}
                coefficient_scale = _single_effect_scale(
                    np.asarray(list(effects.values()), dtype=np.float64)
                )
            else:
                transform = None
                effects = single_effects
                coefficient_scale = target_scale
            if coefficient_scale <= 1e-12:
                continue

            if transform is None:
                for order in by_order:
                    by_order[order].extend(
                        value / coefficient_scale for value in coefficients[order]
                    )
            else:
                latent = _latent_coefficients(
                    dataset.variants,
                    dataset.wild_type_sequence,
                    wt,
                    transform,
                    tuple(by_order),
                )
                for order in by_order:
                    by_order[order].extend(
                        value / coefficient_scale for value in latent.get(order, ())
                    )

            # A censored assay parks most of its library on the detection floor.
            # The plateau is not flat, and its fine structure is what creates
            # local optima, so the prior carries its spread.
            centred = finite_fitness - wt
            floor = float(np.min(centred))
            band = _FLOOR_BAND * float(np.max(centred) - floor)
            on_floor = centred[centred <= floor + band]
            if on_floor.size >= 8:
                texture_scales.append(_robust_scale(on_floor) / target_scale)

            supported_depths: list[int] = []
            supported_fitness: list[float] = []
            supported_additive: list[float] = []
            for mutations, fitness, depth in dataset.variants[
                ["mutations", "fitness", "depth"]
            ].itertuples(index=False, name=None):
                if depth < 2 or not all(item in effects for item in mutations.items()):
                    continue
                supported_depths.append(int(depth))
                supported_fitness.append(float(fitness) - wt)
                supported_additive.append(sum(effects[item] for item in mutations.items()))

            residuals_by_depth: dict[int, list[float]] = {}
            if supported_depths:
                centred = np.asarray(supported_fitness, dtype=np.float64)
                if transform is not None:
                    centred = np.asarray(transform(centred), dtype=np.float64)
                residual = (
                    centred - np.asarray(supported_additive, dtype=np.float64)
                ) / coefficient_scale
                for depth, value in zip(supported_depths, residual):
                    residuals_by_depth.setdefault(depth, []).append(float(value))
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

        depth_locations: dict[int, float] = {}
        depth_scales: dict[int, float] = {}
        depth_between: dict[int, float] = {}
        depth_shrinkage: dict[int, float] = {}
        for depth, within_scales in by_depth_scale.items():
            locations_at_depth = np.asarray(by_depth_location[depth], dtype=np.float64)
            within = np.asarray(within_scales, dtype=np.float64)
            pooled = float(np.median(locations_at_depth))
            between = _robust_scale(locations_at_depth) if locations_at_depth.size > 1 else 0.0
            # A depth residual is a constant shift per mutation count, so it cannot
            # reorder genotypes within a depth and only ever moves the error. It
            # transfers to an unseen protein in proportion to how far the training
            # landscapes agree on it, and vanishes once they disagree more than the
            # shift is large.
            weight = 1.0
            if shrink_depth:
                weight = pooled**2 / (pooled**2 + between**2 + 1e-12)
                # A depth reached by one or two landscapes has no measurable
                # between-protein spread, so the ratio above would wave the largest
                # and least supported shifts straight through.
                if locations_at_depth.size < _MIN_DEPTH_LANDSCAPES:
                    weight = 0.0
            depth_locations[depth] = pooled * weight
            depth_between[depth] = between
            depth_shrinkage[depth] = weight
            # Shrinking the location says the pooled shift does not transfer. It
            # does not say the target sits at the pooled mean, so the spread that
            # justified the shrinkage stays in the predictive scale. Dropping it
            # leaves sampled pair interactions as the only source of uncertainty,
            # and every interval then falls short of nominal.
            depth_scales[depth] = max(sqrt(float(np.mean(within**2)) + between**2), 1e-6)

        lower_bound = float(np.median(lower_bounds)) if lower_bounds else -5.0
        upper_bound = float(np.median(upper_bounds)) if upper_bounds else 5.0
        if lower_bound >= upper_bound:
            lower_bound, upper_bound = -5.0, 5.0
        return cls(
            order_location=locations,
            order_scale=scales,
            depth_location=depth_locations,
            depth_scale=depth_scales,
            depth_between=depth_between,
            depth_shrinkage=depth_shrinkage,
            texture_scale=float(np.median(texture_scales)) if texture_scales else 0.0,
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
            "depth_between": {str(k): v for k, v in self.depth_between.items()},
            "depth_shrinkage": {str(k): v for k, v in self.depth_shrinkage.items()},
            "texture_scale": self.texture_scale,
            "texture_order": self.texture_order,
            "texture_iid_fraction": self.texture_iid_fraction,
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
            depth_between={
                int(k): float(v) for k, v in dict(values.get("depth_between", {})).items()
            },
            depth_shrinkage={
                int(k): float(v) for k, v in dict(values.get("depth_shrinkage", {})).items()
            },
            lower_bound=float(values["lower_bound"]),
            upper_bound=float(values["upper_bound"]),
            texture_scale=float(values.get("texture_scale", 0.0)),
            texture_order=int(values.get("texture_order", 3)),
            texture_iid_fraction=float(values.get("texture_iid_fraction", 0.3)),
            degrees_of_freedom=float(values.get("degrees_of_freedom", 5.0)),
            n_landscapes=int(values.get("n_landscapes", 0)),
            n_proteins=int(values.get("n_proteins", 0)),
        )


def _fit_readout(
    single_effects: dict[tuple[int, str], float],
    prior: LandscapePrior,
    target_scale: float,
    *,
    margin: float = 0.05,
) -> NonLinearity | None:
    """Fit a saturating readout for one target from its singles.

    Many assays report only deleterious singles, or only beneficial ones, so
    the observed singles alone cannot bracket the readout. The pooled assay
    range is the one quantity that does transfer between landscapes, so the
    prior supplies whichever bound the target never reaches.
    """
    effects = np.asarray(list(single_effects.values()), dtype=np.float64)
    if effects.size < 2:
        return None

    low = min(float(effects.min()), prior.lower_bound * target_scale)
    high = max(float(effects.max()), prior.upper_bound * target_scale)
    span = high - low
    if not np.isfinite(span) or span <= 1e-12:
        return None
    floor = min(low - margin * span, -margin * span)
    ceiling = max(high + margin * span, margin * span)

    # The worst measured single marks where the assay stops resolving. Clipping
    # there reproduces the dead plateau a censored readout produces, which is
    # most of a real combinatorial library.
    worst_single = float(effects.min())
    floor_hard = worst_single if worst_single < 0 else None

    by_position: dict[int, dict[str, float]] = {}
    for (position, amino_acid), effect in single_effects.items():
        by_position.setdefault(int(position), {})[amino_acid] = float(effect)
    try:
        fitted = GlobalEpistasisSigmoid.fit_from_singles(
            by_position,
            margin=margin,
            floor=floor,
            ceiling=ceiling,
        )
        return replace(fitted, floor_hard=floor_hard) if floor_hard is not None else fitted
    except (ValueError, RuntimeError):
        return None


@dataclass(frozen=True)
class PosteriorLandscape:
    """A sparse DMS dataset conditioned on a transferable landscape prior."""

    wild_type_sequence: str
    observations: dict[MutationKey, float]
    coefficients: dict[MutationKey, float]
    single_effects: dict[tuple[int, str], float]
    prior: LandscapePrior
    target_scale: float
    nonlinearity: NonLinearity | None = None
    preimage_effects: dict[tuple[int, str], float] = field(default_factory=dict)
    preimage_scale: float = 1.0

    @classmethod
    def condition(
        cls,
        wild_type_sequence: str,
        observations: pd.DataFrame,
        prior: LandscapePrior,
        *,
        global_epistasis: bool = True,
    ) -> Self:
        """Condition the prior on WT, singles, and any supplied combinations.

        With ``global_epistasis`` the effects are summed through the inverse of
        a fitted saturating readout, so stacked mutations approach the assay
        floor or ceiling instead of extrapolating past it. Summing raw measured
        singles instead carries a large depth-dependent bias on any censored
        assay.
        """
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
        target_scale = _single_effect_scale(
            np.asarray(list(single_effects.values()), dtype=np.float64)
        )
        if target_scale <= 1e-12:
            target_scale = 1.0
        if not any(observed_coefficients[order] for order in observed_coefficients):
            raise ValueError("observations must contain at least one single mutant")

        nonlinearity = None
        preimage_effects: dict[tuple[int, str], float] = {}
        preimage_scale = 1.0
        if global_epistasis:
            nonlinearity = _fit_readout(single_effects, prior, target_scale)
        if nonlinearity is not None:
            preimage_effects = {
                mutation: float(nonlinearity.inverse(effect))
                for mutation, effect in single_effects.items()
            }
            preimage_scale = _single_effect_scale(
                np.asarray(list(preimage_effects.values()), dtype=np.float64)
            )
            if preimage_scale <= 1e-12:
                preimage_scale = 1.0
            # Measured interactions are Mobius coefficients of raw fitness, so
            # they only compose with the singles once mapped through the same
            # inverse readout.
            coefficient_map = {
                genotype: sum(
                    (-1.0) ** (len(genotype) - len(subset))
                    * float(nonlinearity.inverse(observed[subset] - wt))
                    for subset in _subsets(genotype)
                )
                for genotype in coefficient_map
            }

        return cls(
            wild_type_sequence=wild_type_sequence,
            observations=observed,
            coefficients=coefficient_map,
            single_effects=single_effects,
            prior=prior,
            target_scale=target_scale,
            nonlinearity=nonlinearity,
            preimage_effects=preimage_effects,
            preimage_scale=preimage_scale,
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

    @property
    def wild_type(self) -> dict[int, str]:
        """One-based residue map, matching the ``Landscape`` tensor interface."""
        return {
            index + 1: amino_acid
            for index, amino_acid in enumerate(self.posterior.wild_type_sequence)
        }

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
        scale = (
            self.posterior.preimage_scale
            if self.posterior.nonlinearity is not None
            else self.posterior.target_scale
        )
        return scale * prior.order_scale[order] * unit_variance_t

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

        readout = posterior.nonlinearity
        effects = posterior.preimage_effects if readout is not None else posterior.single_effects
        scale = posterior.preimage_scale if readout is not None else posterior.target_scale

        latent = sum(effects[mutation] for mutation in genotype)
        depth = len(genotype)
        depth_location = self._at_depth(posterior.prior.depth_location, depth, 0.0)
        latent += scale * depth_location

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
        latent += observed_interaction + sampled_interaction

        wt = posterior.observations[()]
        value = wt + float(readout.forward(latent)) if readout is not None else wt + latent
        value += self._floor_texture(genotype, value - wt)

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

    def _floor_texture(self, genotype: MutationKey, centred: float) -> float:
        """Break the censored plateau into stable local topology.

        A clipped readout maps a wide band of latent values onto one number, so
        every dead variant ties with its neighbours and the landscape reports a
        single optimum. Real censored assays still resolve fine differences
        down there, and those differences carry the local optima that make a
        landscape hard to search. The perturbation applies only on the floor,
        so it never moves a variant the assay could actually resolve.
        """
        posterior = self.posterior
        prior = posterior.prior
        readout = posterior.nonlinearity
        if readout is None or prior.texture_scale <= 0.0:
            return 0.0
        floor_hard = getattr(readout, "floor_hard", None)
        if floor_hard is None:
            return 0.0
        band = _FLOOR_BAND * (float(readout.ceiling) - float(floor_hard))
        if centred > float(floor_hard) + band:
            return 0.0

        order = int(prior.texture_order)
        iid = _hash_normal(("texture-iid", genotype), self.seed)
        if order <= 0 or len(genotype) < order:
            ordered = 0.0
        else:
            groups = list(combinations(genotype, order))
            ordered = float(
                sum(_hash_normal(("texture-order", group), self.seed) for group in groups)
                / sqrt(len(groups))
            )
        mix = min(max(float(prior.texture_iid_fraction), 0.0), 1.0)
        denominator = sqrt((1.0 - mix) ** 2 + mix**2)
        blended = ((1.0 - mix) * ordered + mix * iid) / denominator if denominator else 0.0
        return prior.texture_scale * posterior.target_scale * blended

    def evaluate_batch(self, variants: list[dict[int, str]]) -> NDArray[np.float64]:
        """Evaluate a sequence of genotypes."""
        return np.asarray([self.evaluate(variant) for variant in variants], dtype=np.float64)
