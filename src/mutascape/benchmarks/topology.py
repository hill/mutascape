"""Compare the shape of measured, posterior, and additive landscapes.

Point accuracy asks whether a model predicts hidden values. This asks whether
it produces a landscape at all. An additive surrogate fitted on singles has one
peak and puts every unit of variance at order one, whatever its held-out error,
so a benchmark built on one cannot rank search methods the way the measured
landscape does.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.benchmarks.baselines import RidgeLandscapeBaseline
from mutascape.benchmarks.datasets import LandscapeDataset
from mutascape.completion import PosteriorLandscape
from mutascape.validation.fourier import fitness_tensor_from_real, walsh_decompose
from mutascape.validation.metrics import local_optima_from_tensor


@dataclass(frozen=True)
class CubeSpec:
    """The complete factorial a posterior landscape can actually answer for."""

    positions: tuple[int, ...]
    alphabets: dict[int, list[str]]
    wild_type: dict[int, str]

    @property
    def size(self) -> int:
        return int(np.prod([len(self.alphabets[p]) for p in self.positions]))

    def variants(self) -> tuple[list[dict[int, str]], list[tuple[int, ...]]]:
        """Every genotype in the cube, with its tensor index."""
        lookup = [{aa: i for i, aa in enumerate(self.alphabets[p])} for p in self.positions]
        variants: list[dict[int, str]] = []
        indices: list[tuple[int, ...]] = []
        for combination in product(*(self.alphabets[p] for p in self.positions)):
            variants.append(
                {
                    position: amino_acid
                    for position, amino_acid in zip(self.positions, combination)
                    if amino_acid != self.wild_type[position]
                }
            )
            indices.append(tuple(lookup[i][aa] for i, aa in enumerate(combination)))
        return variants, indices


def supported_cube(dataset: LandscapeDataset, train: pd.DataFrame) -> CubeSpec:
    """Restrict the cube to substitutions backed by a measured single.

    The posterior refuses a genotype whose singles it never saw, so the
    measured and additive arms must be held to the same domain or the three
    tensors would not describe the same object.
    """
    wild_type = {p: dataset.wild_type_sequence[p - 1] for p in dataset.mutable_positions}
    singles = {
        item
        for mutations in train.loc[train["depth"] == 1, "mutations"]
        for item in mutations.items()
    }
    alphabets = {
        position: sorted({wild_type[position]} | {aa for p, aa in singles if p == position})
        for position in dataset.mutable_positions
    }
    return CubeSpec(tuple(dataset.mutable_positions), alphabets, wild_type)


def _fill(tensor: NDArray[np.float64], values, indices) -> NDArray[np.float64]:
    for index, value in zip(indices, values):
        tensor[index] = value
    return tensor


def measured_tensor(dataset: LandscapeDataset, cube: CubeSpec) -> tuple[NDArray[np.float64], float]:
    """Assemble the measured cube, with wild type recentred to zero."""
    variants = dataset.variants
    wt_rows = variants.loc[variants["depth"] == 0, "fitness"]
    wt = float(wt_rows.iloc[0])
    centred = variants.copy()
    centred["fitness"] = centred["fitness"].astype(float) - wt
    return fitness_tensor_from_real(centred, cube.positions, cube.wild_type, cube.alphabets)


def additive_tensor(train: pd.DataFrame, cube: CubeSpec) -> NDArray[np.float64]:
    """Cube produced by ridge on singles, the standard smooth surrogate."""
    baseline = RidgeLandscapeBaseline(max_order=1).fit(train)
    variants, indices = cube.variants()
    wt_rows = train.loc[train["depth"] == 0, "fitness"]
    wt = float(wt_rows.iloc[0])
    values = baseline.predict(variants) - wt
    shape = tuple(len(cube.alphabets[p]) for p in cube.positions)
    return _fill(np.empty(shape, dtype=np.float64), values, indices)


def posterior_tensors(
    posterior: PosteriorLandscape,
    cube: CubeSpec,
    *,
    n_samples: int = 30,
    seed: int = 0,
):
    """Yield one complete cube per posterior draw."""
    variants, indices = cube.variants()
    shape = tuple(len(cube.alphabets[p]) for p in cube.positions)
    wt = posterior.observations[()]
    for index in range(n_samples):
        values = posterior.sample(seed + index).evaluate_batch(variants) - wt
        yield _fill(np.empty(shape, dtype=np.float64), values, indices)


def summarise(tensor: NDArray[np.float64]) -> dict[str, float]:
    """Topology descriptors for one complete cube."""
    optima = local_optima_from_tensor(tensor)
    spectrum = walsh_decompose(tensor)
    summary: dict[str, float] = {
        "n_optima": float(optima["n_optima"]),
        "peak_fraction": float(optima["peak_fraction"]),
        "largest_basin_fraction": float(optima["largest_basin_fraction"]),
        "basin_entropy": float(optima["basin_entropy"]),
    }
    for order, fraction in spectrum.items():
        summary[f"walsh_order_{order}"] = float(fraction)
    summary["walsh_nonadditive"] = float(
        sum(value for order, value in spectrum.items() if order >= 2)
    )
    finite = tensor[np.isfinite(tensor)]
    summary["fitness_std"] = float(np.std(finite)) if finite.size else float("nan")
    return summary


def walsh_l1(reference: dict[str, float], candidate: dict[str, float]) -> float:
    """L1 distance between two order-wise Walsh spectra."""
    orders = sorted(
        {key for key in reference if key.startswith("walsh_order_")}
        & {key for key in candidate if key.startswith("walsh_order_")}
    )
    return float(sum(abs(reference[key] - candidate[key]) for key in orders))
