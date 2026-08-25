"""Structural and biophysical covariates for pairwise interaction coefficients.

The pooled prior gives every pair of a given order the same distribution, so it
carries no information about which pair interacts. These features let a model
predict a coefficient from the residues involved.

Structure enters through the deposited backbone, not through a fitted parameter,
so it is available before any combination is measured.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.datasets import LandscapeDataset
from mutascape.completion import _observation_map, _single_effect_scale, _subsets
from mutascape.contacts import ContactMap

# Kyte-Doolittle hydropathy, used for a crude chemical-similarity feature.
HYDROPATHY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

VOLUME = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}


@dataclass(frozen=True)
class StructureIndex:
    """Cbeta distances between the mutable residues of one landscape."""

    source_positions: dict[int, int]
    distances: dict[tuple[int, int], float]
    neighbour_counts: dict[int, int]

    def distance(self, left: int, right: int) -> float:
        return self.distances.get((min(left, right), max(left, right)), float("nan"))


def structure_index(
    dataset: LandscapeDataset,
    pdb_path: Path | str,
    *,
    contact_threshold: float = 8.0,
) -> StructureIndex | None:
    """Map a landscape's mutable sites onto a structure.

    SSMuLA numbers its mutable sites 1..k in the mutable-site sequence and keeps
    the residue numbers in the spec, so the spec supplies the mapping. Reading
    the sequence positions as residue numbers would silently select the wrong
    residues.
    """
    source = dataset.spec.positions
    if source is None or len(source) != len(dataset.mutable_positions):
        return None
    try:
        contacts = ContactMap.from_pdb(pdb_path, threshold=contact_threshold)
    except OSError, ValueError:
        return None
    if contacts.distances is None:
        return None

    lookup = {position: index for index, position in enumerate(contacts.positions)}
    if not all(position in lookup for position in source):
        return None

    mapping = dict(zip(dataset.mutable_positions, source))
    distances: dict[tuple[int, int], float] = {}
    for left, right in combinations(dataset.mutable_positions, 2):
        row, column = lookup[mapping[left]], lookup[mapping[right]]
        distances[(min(left, right), max(left, right))] = float(contacts.distances[row, column])
    neighbours = {
        site: int(np.sum(contacts.matrix[lookup[mapping[site]]]))
        for site in dataset.mutable_positions
    }
    return StructureIndex(mapping, distances, neighbours)


def pair_features(
    dataset: LandscapeDataset,
    structure: StructureIndex | None,
    *,
    max_order: int = 2,
) -> pd.DataFrame:
    """One row per measured pairwise interaction, with its covariates.

    A row appears only when the pair and both its singles were measured, which
    is the same rule the empirical prior uses.
    """
    if max_order != 2:
        raise ValueError("pair features describe order two only")
    observed = _observation_map(dataset.variants, dataset.wild_type_sequence)
    wt = observed[()]
    singles = {
        genotype[0]: value - wt for genotype, value in observed.items() if len(genotype) == 1
    }
    if len(singles) < 2:
        return pd.DataFrame()
    scale = _single_effect_scale(np.asarray(list(singles.values()), dtype=np.float64))
    if scale <= 1e-12:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for genotype, value in observed.items():
        if len(genotype) != 2:
            continue
        if any(subset not in observed for subset in _subsets(genotype)):
            continue
        (left_position, left_aa), (right_position, right_aa) = genotype
        coefficient = sum(
            (-1.0) ** (2 - len(subset)) * observed[subset] for subset in _subsets(genotype)
        )
        left_effect = singles[(left_position, left_aa)]
        right_effect = singles[(right_position, right_aa)]
        wt_left = dataset.wild_type_sequence[left_position - 1]
        wt_right = dataset.wild_type_sequence[right_position - 1]
        source = structure.source_positions if structure else {}
        rows.append(
            {
                "dataset": dataset.name,
                "protein": dataset.protein,
                "beta": coefficient / scale,
                "abs_beta": abs(coefficient) / scale,
                "effect_product": left_effect * right_effect / scale**2,
                "effect_min": min(abs(left_effect), abs(right_effect)) / scale,
                "effect_max": max(abs(left_effect), abs(right_effect)) / scale,
                "same_sign": float(np.sign(left_effect) == np.sign(right_effect)),
                "cb_distance": structure.distance(left_position, right_position)
                if structure
                else np.nan,
                "sequence_separation": abs(
                    source.get(left_position, left_position)
                    - source.get(right_position, right_position)
                ),
                "neighbours_min": min(
                    structure.neighbour_counts.get(left_position, 0),
                    structure.neighbour_counts.get(right_position, 0),
                )
                if structure
                else np.nan,
                "hydropathy_shift": abs(HYDROPATHY.get(left_aa, 0.0) - HYDROPATHY.get(wt_left, 0.0))
                + abs(HYDROPATHY.get(right_aa, 0.0) - HYDROPATHY.get(wt_right, 0.0)),
                "volume_shift": abs(VOLUME.get(left_aa, 0.0) - VOLUME.get(wt_left, 0.0))
                + abs(VOLUME.get(right_aa, 0.0) - VOLUME.get(wt_right, 0.0)),
            }
        )
    return pd.DataFrame(rows)


FEATURE_COLUMNS = (
    "effect_product",
    "effect_min",
    "effect_max",
    "same_sign",
    "cb_distance",
    "sequence_separation",
    "neighbours_min",
    "hydropathy_shift",
    "volume_shift",
)


def build_feature_table(
    datasets: list[LandscapeDataset],
    structures: dict[str, StructureIndex | None],
) -> pd.DataFrame:
    """Pooled pair features across landscapes."""
    frames = [pair_features(dataset, structures.get(dataset.name)) for dataset in datasets]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
