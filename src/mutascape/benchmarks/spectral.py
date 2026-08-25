"""Calibrate the per-order interaction scale against measured Walsh spectra.

Fitting the order scales from the magnitude of Mobius-inverted coefficients
gets the ratio between orders wrong. Measured cubes carry a median 0.443 of
their variance at order three and 0.266 at order two. Generated cubes invert
that, which is most of the remaining spectral distance and all of the missing
ruggedness, because a pair term reorders far fewer genotypes than a triple.

The correction reads the measured spectrum of the training landscapes only,
never the target, so it stays inside the protein holdout.
"""

from dataclasses import replace

import numpy as np

from mutascape.benchmarks.splits import has_informative_singles, mutation_order_split
from mutascape.benchmarks.topology import (
    measured_tensor,
    posterior_tensors,
    summarise,
    supported_cube,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape

# A correction outside this band means the fixed point is chasing noise in a
# thinly covered cube rather than a real allocation error.
_MIN_CORRECTION = 0.2
_MAX_CORRECTION = 5.0
_REFERENCE_ORDER = 2


def _spectrum(summary: dict[str, float], max_order: int) -> dict[int, float]:
    return {
        order: float(summary.get(f"walsh_order_{order}", 0.0)) for order in range(2, max_order + 1)
    }


def _usable(datasets, *, max_cube: int, max_landscapes: int, min_coverage: float):
    """Training cubes small enough and complete enough to read a spectrum from.

    Measured roughness spans 0.168 to 0.973 across proteins, so a calibration
    set drawn from one protein reports that protein's shape and not the pooled
    one. At most one library per protein enters, and proteins are taken in
    coverage order.
    """
    candidates: list[tuple[float, object]] = []
    for dataset in datasets:
        split = mutation_order_split(dataset.variants, max_train_depth=1)
        if not has_informative_singles(split.train):
            continue
        cube = supported_cube(dataset, split.train)
        if cube.size > max_cube or cube.size < 64:
            continue
        measured, coverage = measured_tensor(dataset, cube)
        if coverage < min_coverage or not np.isfinite(measured).all():
            continue
        candidates.append((coverage, (dataset, cube, measured, split)))

    candidates.sort(key=lambda item: (-item[0], item[1][0].name))
    chosen: list[tuple] = []
    proteins: set[str] = set()
    for _, entry in candidates:
        if len(chosen) >= max_landscapes:
            break
        protein = entry[0].protein
        if protein in proteins:
            continue
        proteins.add(protein)
        chosen.append(entry)
    return chosen


def calibrate_order_scale(
    prior: LandscapePrior,
    datasets,
    *,
    iterations: int = 2,
    draws: int = 1,
    max_cube: int = 20_000,
    max_landscapes: int = 6,
    min_coverage: float = 0.95,
    seed: int = 0,
) -> LandscapePrior:
    """Rescale each order so generated spectra match the training spectra.

    Only the ratio to order two is corrected. Total interaction magnitude is
    already set by the depth scale, so correcting the absolute level as well
    would fight it.
    """
    max_order = max(prior.order_scale) if prior.order_scale else 2
    if max_order <= _REFERENCE_ORDER:
        return prior
    chosen = _usable(
        datasets, max_cube=max_cube, max_landscapes=max_landscapes, min_coverage=min_coverage
    )
    if not chosen:
        return prior

    targets = []
    for _, _, measured, _ in chosen:
        spectrum = _spectrum(summarise(measured), max_order)
        reference = spectrum.get(_REFERENCE_ORDER, 0.0)
        if reference <= 0:
            continue
        targets.append({order: value / reference for order, value in spectrum.items()})
    if not targets:
        return prior

    current = prior
    for step in range(iterations):
        realised: dict[int, list[float]] = {order: [] for order in range(3, max_order + 1)}
        for index, (dataset, cube, _, split) in enumerate(chosen):
            posterior = PosteriorLandscape.condition(
                dataset.wild_type_sequence, split.train, current
            )
            for tensor in posterior_tensors(
                posterior, cube, n_samples=draws, seed=seed + 100 * step + index
            ):
                spectrum = _spectrum(summarise(tensor), max_order)
                reference = spectrum.get(_REFERENCE_ORDER, 0.0)
                if reference <= 0:
                    continue
                for order in realised:
                    realised[order].append(spectrum[order] / reference)

        scales = dict(current.order_scale)
        for order in range(3, max_order + 1):
            observed = [value for value in realised[order] if value > 0]
            wanted = [target[order] for target in targets if target.get(order, 0.0) > 0]
            if not observed or not wanted:
                continue
            ratio = float(np.median(wanted)) / float(np.median(observed))
            correction = float(np.clip(np.sqrt(ratio), _MIN_CORRECTION, _MAX_CORRECTION))
            scales[order] = scales[order] * correction
        current = replace(current, order_scale=scales)
    return current
