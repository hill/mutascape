"""Probabilistic completion of whole-protein combinatorial landscapes.

Sparse observations determine the wild-type and single-mutant effects. An
empirical prior learned from other combinatorial landscapes supplies missing
interaction coefficients. Each posterior sample is a stable oracle defined
for every genotype in the target mutation domain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from functools import cached_property
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from math import sqrt
from os import PathLike
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.domain import MutationDomain
from mutascape.mutant import VariantLike
from mutascape.nonlinearity import GlobalEpistasisSigmoid, NonLinearity

if TYPE_CHECKING:
    from mutascape.benchmarks.datasets import LandscapeDataset

MutationKey = tuple[tuple[int, str], ...]
DEFAULT_DMS_PAIRWISE_COUPLING = 0.1

# Below this many contributing landscapes a depth bin cannot support a transferred shift.
_MIN_DEPTH_LANDSCAPES = 3

# Width of the band, as a fraction of the assay range, counted as sitting on the floor.
_FLOOR_BAND = 0.02

# Above this share of single mutants tied at the worst measured value, the assay
# is treated as censored. Measured censored assays sit near 45 per cent and
# uncensored ones near 6 per cent.
_CENSORED_TIE_FRACTION = 0.15

# Quantile grid used to carry the measured coefficient shape between proteins.
_SHAPE_GRID = np.linspace(0.0, 1.0, 257)


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


def dms_pairwise_unit_coefficient(
    first: tuple[int, str],
    second: tuple[int, str],
    wild_type_sequence: str,
    seed: int,
) -> float:
    """Draw one baseline-corrected pair coefficient on a stable hash key."""
    if first[0] == second[0]:
        raise ValueError("a pair interaction needs two distinct positions")
    first, second = sorted((first, second))
    first_wt = wild_type_sequence[first[0] - 1]
    second_wt = wild_type_sequence[second[0] - 1]

    def cell(first_aa: str, second_aa: str) -> float:
        return _hash_normal(
            ("dms-pairwise-cell", first[0], first_aa, second[0], second_aa),
            seed,
        )

    return (
        cell(first[1], second[1])
        - cell(first[1], second_wt)
        - cell(first_wt, second[1])
        + cell(first_wt, second_wt)
    )


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


def _refit_scale(readout, centred_fitness: NDArray[np.float64], *, cap: int = 5000) -> float | None:
    """Refit only the saturation rate, holding the bounds fixed."""
    finite = centred_fitness[np.isfinite(centred_fitness)]
    if finite.size < 32:
        return None
    if finite.size > cap:
        finite = np.random.default_rng(0).choice(finite, size=cap, replace=False)
    values = {0: {str(index): float(value) for index, value in enumerate(finite)}}
    try:
        refitted = GlobalEpistasisSigmoid.fit_from_singles(
            values, margin=0.05, floor=readout.floor, ceiling=readout.ceiling
        )
    except (ValueError, RuntimeError):
        return None
    return float(refitted.scale)


def _magnitude_pairs(
    variants: pd.DataFrame,
    wild_type_sequence: str,
    wild_type_fitness: float,
    transform,
    effects: dict[tuple[int, str], float],
    scale: float,
) -> list[tuple[float, float]]:
    """Pair each measured order-two coefficient with its milder single effect.

    Interaction magnitude is not constant across pairs. It tracks how strongly
    the two mutations act on their own, so a prior with one scale for every pair
    produces residuals that are too uniform to pass as measured ones.
    """
    observed = _observation_map(variants, wild_type_sequence)
    latent = {key: float(transform(value - wild_type_fitness)) for key, value in observed.items()}
    pairs: list[tuple[float, float]] = []
    for genotype in latent:
        if len(genotype) != 2:
            continue
        if any(subset not in latent for subset in _subsets(genotype)):
            continue
        if any(mutation not in effects for mutation in genotype):
            continue
        coefficient = sum(
            (-1.0) ** (2 - len(subset)) * latent[subset] for subset in _subsets(genotype)
        )
        milder = min(abs(effects[mutation]) for mutation in genotype) / scale
        pairs.append((milder, coefficient / scale))
    return pairs


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
    order_shape: dict[int, list[float]] = field(default_factory=dict)
    readout_scale_ratio: float = 1.0
    combinatorial_upside: float = 1.0
    combinatorial_downside: float = 1.0
    magnitude_edges: list[float] = field(default_factory=list)
    magnitude_scale: list[float] = field(default_factory=list)
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
        scale_ratios: list[float] = []
        upsides: list[float] = []
        downsides: list[float] = []
        magnitude_pairs: list[tuple[float, float]] = []

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
                # The saturation rate is what decides how fast stacked mutations
                # approach the floor, and singles carry no information about it.
                # Refitting the same readout against every measured variant gives
                # the rate that landscape actually has, and the ratio between the
                # two transfers even though neither value does.
                full = _refit_scale(readout, finite_fitness - wt)
                if full is not None and full > 1e-9:
                    scale_ratios.append(float(readout.scale) / full)
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
                magnitude_pairs.extend(
                    _magnitude_pairs(
                        dataset.variants,
                        dataset.wild_type_sequence,
                        wt,
                        transform,
                        effects,
                        coefficient_scale,
                    )
                )
                for order in by_order:
                    by_order[order].extend(
                        value / coefficient_scale for value in latent.get(order, ())
                    )

            # A censored assay parks most of its library on the detection floor.
            # The plateau is not flat, and its fine structure is what creates
            # local optima, so the prior carries its spread.
            centred = finite_fitness - wt
            # Stacked mutations reach further than any single one does, and the
            # bounds are the only part of the readout that controls how fast
            # they saturate. How much further transfers between proteins.
            singles_high = float(np.max(list(single_effects.values())))
            singles_low = float(np.min(list(single_effects.values())))
            if singles_high > 1e-9:
                upsides.append(float(np.max(centred)) / singles_high)
            if singles_low < -1e-9:
                downsides.append(float(np.min(centred)) / singles_low)

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
        shapes: dict[int, list[float]] = {}
        previous_scale = 0.25
        for order in range(2, max_order + 1):
            values = np.asarray(by_order[order], dtype=np.float64)
            if values.size:
                locations[order] = float(np.median(values))
                scale = _robust_scale(values)
                scales[order] = max(scale, 1e-6)
                previous_scale = scales[order]
                # A Student t is an assumption. The measured coefficients are
                # available, and their shape is not a t. Carrying the quantile
                # grid reproduces it without naming a family.
                spread = float(np.std(values))
                if values.size >= 64 and spread > 1e-12:
                    standardised = (values - np.mean(values)) / spread
                    shapes[order] = [float(q) for q in np.quantile(standardised, _SHAPE_GRID)]
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

        magnitude_edges: list[float] = []
        magnitude_scale: list[float] = []
        if len(magnitude_pairs) >= 400:
            sizes = np.asarray([item[0] for item in magnitude_pairs], dtype=np.float64)
            values = np.asarray([item[1] for item in magnitude_pairs], dtype=np.float64)
            edges = np.quantile(sizes, np.linspace(0.0, 1.0, 9))
            spreads = []
            for index in range(len(edges) - 1):
                inside = (sizes >= edges[index]) & (sizes <= edges[index + 1])
                spreads.append(float(np.std(values[inside])) if inside.sum() >= 30 else np.nan)
            spread = np.asarray(spreads, dtype=np.float64)
            if np.isfinite(spread).sum() >= 4:
                mean_spread = float(np.nanmean(spread))
                if mean_spread > 1e-9:
                    spread = np.where(np.isfinite(spread), spread, mean_spread) / mean_spread
                    magnitude_edges = [float(x) for x in edges]
                    magnitude_scale = [float(x) for x in spread]

        lower_bound = float(np.median(lower_bounds)) if lower_bounds else -5.0
        upper_bound = float(np.median(upper_bounds)) if upper_bounds else 5.0
        if lower_bound >= upper_bound:
            lower_bound, upper_bound = -5.0, 5.0
        return cls(
            order_location=locations,
            order_scale=scales,
            order_shape=shapes,
            depth_location=depth_locations,
            depth_scale=depth_scales,
            depth_between=depth_between,
            depth_shrinkage=depth_shrinkage,
            texture_scale=float(np.median(texture_scales)) if texture_scales else 0.0,
            readout_scale_ratio=float(np.median(scale_ratios)) if scale_ratios else 1.0,
            combinatorial_upside=float(np.median(upsides)) if upsides else 1.0,
            combinatorial_downside=float(np.median(downsides)) if downsides else 1.0,
            magnitude_edges=magnitude_edges,
            magnitude_scale=magnitude_scale,
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
            "order_shape": {str(k): v for k, v in self.order_shape.items()},
            "readout_scale_ratio": self.readout_scale_ratio,
            "combinatorial_upside": self.combinatorial_upside,
            "combinatorial_downside": self.combinatorial_downside,
            "magnitude_edges": list(self.magnitude_edges),
            "magnitude_scale": list(self.magnitude_scale),
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
            order_shape={
                int(k): [float(x) for x in v]
                for k, v in dict(values.get("order_shape", {})).items()
            },
            readout_scale_ratio=float(values.get("readout_scale_ratio", 1.0)),
            combinatorial_upside=float(values.get("combinatorial_upside", 1.0)),
            combinatorial_downside=float(values.get("combinatorial_downside", 1.0)),
            magnitude_edges=[float(x) for x in values.get("magnitude_edges", [])],
            magnitude_scale=[float(x) for x in values.get("magnitude_scale", [])],
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
    floor_policy: str = "censored",
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

    reach_low = float(effects.min()) * max(prior.combinatorial_downside, 1.0)
    reach_high = float(effects.max()) * max(prior.combinatorial_upside, 1.0)
    low = min(reach_low, prior.lower_bound * target_scale)
    high = max(reach_high, prior.upper_bound * target_scale)
    span = high - low
    if not np.isfinite(span) or span <= 1e-12:
        return None
    floor = min(low - margin * span, -margin * span)
    ceiling = max(high + margin * span, margin * span)

    # Clip only where the singles show the assay stops resolving. Several
    # singles landing on the same worst value is that evidence. Clipping every
    # assay instead put four fifths of a generated library on the floor where
    # the measured one holds a quarter, and it flattened libraries whose
    # combinations never reach the worst single at all.
    worst_single = float(effects.min())
    span_effects = float(effects.max() - worst_single)
    tied = float(np.mean(effects <= worst_single + _FLOOR_BAND * span_effects))
    censored = span_effects > 0 and tied >= _CENSORED_TIE_FRACTION
    if floor_policy == "always":
        censored = True
    elif floor_policy == "never":
        censored = False
    floor_hard = worst_single if censored and worst_single < 0 else None

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
        # The readout scale cannot change anything here. Preimages carry a
        # factor of the scale and the composition divides it straight back out,
        # so only the floor and the ceiling set how fast stacking saturates.
        # ``readout_scale_ratio`` stays on the prior as a measured diagnostic.
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
    donors: tuple = ()
    maximum_depth: int | None = None
    prior_id: str | None = None

    @cached_property
    def domain(self) -> MutationDomain:
        """Declared mutation domain supported by the measured singles."""
        return MutationDomain(
            self.wild_type_sequence,
            tuple(self.single_effects),
            maximum_depth=self.maximum_depth,
        )

    @cached_property
    def source_digest(self) -> str:
        """Stable digest of the sequence and exact supplied observations."""
        payload = {
            "wild_type_sequence": self.wild_type_sequence,
            "observations": [
                {"mutations": list(genotype), "fitness": fitness}
                for genotype, fitness in sorted(self.observations.items())
            ],
        }
        return hashlib.blake2b(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            digest_size=16,
        ).hexdigest()

    @classmethod
    def condition(
        cls,
        wild_type_sequence: str,
        observations: pd.DataFrame,
        prior: LandscapePrior,
        *,
        global_epistasis: bool = True,
        floor_policy: str = "censored",
        donors: tuple = (),
        maximum_depth: int | None = None,
        prior_id: str | None = None,
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
            nonlinearity = _fit_readout(
                single_effects, prior, target_scale, floor_policy=floor_policy
            )
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
            donors=tuple(donors),
            maximum_depth=maximum_depth,
            prior_id=prior_id,
        )

    def draw(self, seed: int) -> DMSPairwiseLandscape:
        """Draw one fixed oracle from the versioned default generator."""
        return self.sample_dms_pairwise(seed, coupling=DEFAULT_DMS_PAIRWISE_COUPLING)

    def sample(
        self,
        seed: int,
        *,
        clip_to_assay_range: bool = False,
        roughness: float = 1.0,
    ) -> CompletedLandscape:
        """Return one fixed oracle realisation.

        Clipping is optional because the latent fitness oracle and a censored
        assay readout represent different objects.

        ``roughness`` scales every structure the model adds beyond the measured
        singles and the readout. At zero the sample is the additive surrogate,
        which traces the accuracy against realism frontier with one knob.
        """
        return CompletedLandscape(
            self,
            seed=int(seed),
            clip_to_assay_range=clip_to_assay_range,
            roughness=float(roughness),
        )

    def sample_dms_pairwise(
        self,
        seed: int,
        *,
        coupling: float = 0.1,
    ) -> DMSPairwiseLandscape:
        """Return the task-validated DMS-pairwise oracle."""
        if coupling < 0:
            raise ValueError("coupling must be non-negative")
        return DMSPairwiseLandscape(self, seed=int(seed), coupling=float(coupling))

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
    roughness: float = 1.0

    @property
    def domain(self) -> MutationDomain:
        return self.posterior.domain

    @cached_property
    def _transplant(self):
        """Mapping onto one donor landscape, fixed by this sample's seed."""
        if not self.posterior.donors:
            return None
        from mutascape.transplant import build_index

        return build_index(list(self.posterior.donors), self.posterior.single_effects, self.seed)

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

    def _magnitude(self, interaction: MutationKey) -> float:
        """Relative interaction spread for one mutation set.

        Two mutations that each do little on their own rarely interact
        strongly. Scaling the draw by the measured profile makes the residuals
        heteroscedastic, which a single pooled scale cannot be.
        """
        prior = self.posterior.prior
        if not prior.magnitude_edges or not prior.magnitude_scale:
            return 1.0
        posterior = self.posterior
        effects = posterior.preimage_effects or posterior.single_effects
        scale = posterior.preimage_scale if posterior.nonlinearity else posterior.target_scale
        sizes = [abs(effects[m]) for m in interaction if m in effects]
        if not sizes:
            return 1.0
        milder = min(sizes) / (scale if scale > 1e-12 else 1.0)
        index = int(np.searchsorted(prior.magnitude_edges, milder, side="right")) - 1
        index = min(max(index, 0), len(prior.magnitude_scale) - 1)
        return float(prior.magnitude_scale[index])

    def _draw_coefficient(self, interaction: MutationKey) -> float:
        observed = self.posterior.coefficients.get(interaction)
        if observed is not None:
            return observed
        order = len(interaction)
        prior = self.posterior.prior
        scale = (
            self.posterior.preimage_scale
            if self.posterior.nonlinearity is not None
            else self.posterior.target_scale
        )

        index = self._transplant
        if index is not None:
            borrowed = index.coefficient(interaction)
            if borrowed is not None:
                return scale * borrowed

        payload = f"{self.seed}|{interaction!r}".encode()
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        draw_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(draw_seed)
        shape = prior.order_shape.get(order)
        if shape:
            unit_variance_t = float(np.interp(rng.random(), _SHAPE_GRID, shape))
        else:
            df = prior.degrees_of_freedom
            unit_variance_t = float(rng.standard_t(df) / sqrt(df / (df - 2.0)))
        # The depth prior carries systematic residual location. Centring each
        # coefficient avoids multiplying an order median by every subset.
        return scale * prior.order_scale[order] * self._magnitude(interaction) * unit_variance_t

    def evaluate(self, mutations: VariantLike) -> float:
        """Evaluate any genotype in the whole-protein mutation domain."""
        posterior = self.posterior
        mutant = self.domain.parse(mutations)
        genotype = _key(mutant, posterior.wild_type_sequence)
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
                    sampled_variance += (
                        posterior.prior.order_scale[order] * self._magnitude(interaction)
                    ) ** 2
                else:
                    observed_interaction += coefficient

        desired_scale = self._at_depth(
            posterior.prior.depth_scale,
            depth,
            sqrt(sampled_variance),
        )
        if sampled_variance > 0:
            sampled_interaction *= self.roughness * desired_scale / sqrt(sampled_variance)
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
        if readout is None or prior.texture_scale <= 0.0 or self.roughness <= 0.0:
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
        return self.roughness * prior.texture_scale * posterior.target_scale * blended

    def evaluate_batch(self, variants: list[VariantLike]) -> NDArray[np.float64]:
        """Evaluate a sequence of genotypes."""
        return np.asarray([self.evaluate(variant) for variant in variants], dtype=np.float64)

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
        return self.score(variant)


@dataclass(frozen=True)
class DMSPairwiseLandscape:
    """One lazy oracle with measured singles, random pairs, and a readout."""

    generator: PosteriorLandscape
    seed: int
    coupling: float = 0.1

    @property
    def domain(self) -> MutationDomain:
        domain = getattr(self.generator, "domain", None)
        if domain is not None:
            return domain
        return MutationDomain(
            self.generator.wild_type_sequence,
            tuple(self.generator.single_effects),
            maximum_depth=getattr(self.generator, "maximum_depth", None),
        )

    @property
    def wild_type(self) -> dict[int, str]:
        return {
            index + 1: amino_acid
            for index, amino_acid in enumerate(self.generator.wild_type_sequence)
        }

    def evaluate(self, mutations: VariantLike) -> float:
        """Evaluate one genotype without enumerating the mutation domain."""
        generator = self.generator
        mutant = self.domain.parse(mutations)
        genotype = _key(mutant, generator.wild_type_sequence)
        if genotype in generator.observations:
            return generator.observations[genotype]
        missing = [mutation for mutation in genotype if mutation not in generator.single_effects]
        if missing:
            labels = ", ".join(f"{position}{amino_acid}" for position, amino_acid in missing)
            raise ValueError(f"missing single-mutant measurements for: {labels}")

        readout = generator.nonlinearity
        effects = generator.preimage_effects if readout is not None else generator.single_effects
        scale = generator.preimage_scale if readout is not None else generator.target_scale
        latent = sum(effects[mutation] for mutation in genotype)
        latent += self.coupling * scale * sum(
            dms_pairwise_unit_coefficient(
                first,
                second,
                generator.wild_type_sequence,
                self.seed,
            )
            for first, second in combinations(genotype, 2)
        )
        centred = float(readout.forward(latent)) if readout is not None else float(latent)
        return float(generator.observations[()] + centred)

    def evaluate_batch(self, variants: list[VariantLike]) -> NDArray[np.float64]:
        """Evaluate genotypes in input order."""
        return np.asarray([self.evaluate(variant) for variant in variants], dtype=np.float64)

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

    def is_observed(self, variant: VariantLike) -> bool:
        """Return whether this exact genotype was supplied as an observation."""
        mutant = self.domain.parse(variant)
        return _key(mutant, self.generator.wild_type_sequence) in self.generator.observations

    def manifest(self) -> dict[str, object]:
        """Return the fixed settings required to identify this benchmark oracle."""
        try:
            package_version = version("mutascape")
        except PackageNotFoundError:
            package_version = "unknown"
        return {
            "mutascape_version": package_version,
            "generator": "dms-pairwise-v1",
            "prior": self.generator.prior_id or "custom",
            "seed": self.seed,
            "coupling": self.coupling,
            "source_digest": self.generator.source_digest,
            "observations": len(self.generator.observations),
            "wild_type_fitness": self.generator.observations[()],
            "positions": list(self.domain.positions),
            "substitutions": len(self.domain.substitutions),
            "maximum_depth": self.domain.maximum_depth,
        }

    def save_manifest(self, path: str | PathLike[str]) -> None:
        """Write the oracle manifest as formatted JSON."""
        from pathlib import Path

        Path(path).write_text(json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n")

    def __getitem__(self, variant: VariantLike) -> float:
        return self.score(variant)

    def __repr__(self) -> str:
        maximum = self.domain.maximum_depth
        return (
            "Oracle("
            'generator="dms-pairwise-v1", '
            f"seed={self.seed}, "
            f"positions={len(self.domain.positions)}, "
            f"substitutions={len(self.domain.substitutions)}, "
            f"maximum_depth={maximum})"
        )
