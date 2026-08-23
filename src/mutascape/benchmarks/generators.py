"""Landscape generators the field already benchmarks on, on one common cube.

NK, Rough Mount Fuji and a Potts model are the standard synthetic oracles. None
of them sees a measured combinatorial landscape, so none has ever been scored
against one. Building them over the same cube as the measured tensor puts every
generator on the same domain and the same metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.benchmarks.baselines import RidgeLandscapeBaseline
from mutascape.benchmarks.topology import CubeSpec


def _shape(cube: CubeSpec) -> tuple[int, ...]:
    return tuple(len(cube.alphabets[position]) for position in cube.positions)


def _site_fields(cube: CubeSpec, train: pd.DataFrame) -> list[NDArray[np.float64]]:
    """Per-site effect of each residue, taken from the measured singles.

    A generator that ignores the target's own singles is not comparable to one
    that uses them, so every generator here is anchored the same way.
    """
    wild_type = cube.wild_type
    singles: dict[tuple[int, str], float] = {}
    for mutations, fitness in train.loc[train["depth"] == 1, ["mutations", "fitness"]].itertuples(
        index=False, name=None
    ):
        position, amino_acid = next(iter(mutations.items()))
        singles[(position, amino_acid)] = float(fitness)
    wt = float(train.loc[train["depth"] == 0, "fitness"].iloc[0])

    fields = []
    for position in cube.positions:
        residues = cube.alphabets[position]
        values = np.array(
            [
                0.0
                if residue == wild_type[position]
                else singles.get((position, residue), np.nan) - wt
                for residue in residues
            ],
            dtype=np.float64,
        )
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.nanmean(values) if np.isfinite(values).any() else 0.0
        fields.append(values)
    return fields


def _additive(cube: CubeSpec, fields: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    shape = _shape(cube)
    tensor = np.zeros(shape, dtype=np.float64)
    for site, values in enumerate(fields):
        view = values.reshape([-1 if index == site else 1 for index in range(len(shape))])
        tensor = tensor + view
    return tensor


def nk_tensor(cube: CubeSpec, *, k: int, seed: int, scale: float = 1.0) -> NDArray[np.float64]:
    """Kauffman NK landscape over the cube's sites.

    Each site contributes a value drawn independently for every configuration
    of itself and ``k`` other sites, which is the classical ruggedness dial.
    """
    shape = _shape(cube)
    n_sites = len(shape)
    rng = np.random.default_rng(seed)
    tensor = np.zeros(shape, dtype=np.float64)
    for site in range(n_sites):
        others = [index for index in range(n_sites) if index != site]
        neighbours = sorted(rng.choice(others, size=min(k, len(others)), replace=False).tolist())
        group = sorted([site, *neighbours])
        table = rng.normal(size=tuple(shape[index] for index in group))
        broadcast = table.reshape(
            [shape[index] if index in group else 1 for index in range(n_sites)]
        )
        tensor = tensor + broadcast
    return scale * tensor / np.sqrt(n_sites)


def rough_mount_fuji_tensor(
    cube: CubeSpec,
    train: pd.DataFrame,
    *,
    roughness: float,
    seed: int,
) -> NDArray[np.float64]:
    """Additive field from the measured singles plus independent site noise."""
    fields = _site_fields(cube, train)
    additive = _additive(cube, fields)
    rng = np.random.default_rng(seed)
    spread = float(np.std(additive)) or 1.0
    return additive + roughness * spread * rng.normal(size=additive.shape)


def potts_tensor(
    cube: CubeSpec,
    train: pd.DataFrame,
    *,
    coupling: float,
    seed: int,
) -> NDArray[np.float64]:
    """Fields from the measured singles with random pairwise couplings.

    SLIP tunes the coupling strength against a target epistatic horizon. The
    dial is exposed here so the audit can ask whether any setting of it
    reproduces a measured landscape.
    """
    shape = _shape(cube)
    n_sites = len(shape)
    fields = _site_fields(cube, train)
    tensor = _additive(cube, fields)
    spread = float(np.std(tensor)) or 1.0
    rng = np.random.default_rng(seed)
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            table = rng.normal(size=(shape[first], shape[second]))
            broadcast = table.reshape(
                [shape[index] if index in (first, second) else 1 for index in range(n_sites)]
            )
            tensor = tensor + coupling * spread * broadcast
    return tensor


def ridge_pairwise_tensor(cube: CubeSpec, train: pd.DataFrame) -> NDArray[np.float64]:
    """Pairwise ridge fitted on the same singles, as a learned smooth surrogate."""
    baseline = RidgeLandscapeBaseline(max_order=2).fit(train)
    variants, indices = cube.variants()
    wt = float(train.loc[train["depth"] == 0, "fitness"].iloc[0])
    tensor = np.empty(_shape(cube), dtype=np.float64)
    for index, value in zip(indices, baseline.predict(variants) - wt):
        tensor[index] = value
    return tensor


def build_generators(
    cube: CubeSpec,
    train: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, NDArray[np.float64]]:
    """Every comparator generator, built on one cube from wild type and singles."""
    return {
        "nk_k1": nk_tensor(cube, k=1, seed=seed),
        "nk_k2": nk_tensor(cube, k=2, seed=seed + 1),
        "rmf_low": rough_mount_fuji_tensor(cube, train, roughness=0.25, seed=seed + 2),
        "rmf_high": rough_mount_fuji_tensor(cube, train, roughness=1.0, seed=seed + 3),
        "potts_low": potts_tensor(cube, train, coupling=0.25, seed=seed + 4),
        "potts_high": potts_tensor(cube, train, coupling=1.0, seed=seed + 5),
        "ridge_pairwise": ridge_pairwise_tensor(cube, train),
    }
