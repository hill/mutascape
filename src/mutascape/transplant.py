"""Borrow the interaction structure of a landscape that was measured.

Every parametric prior so far samples each interaction on its own from a fitted
family. Real interactions are neither independent nor drawn from any family we
have named, and the classifier keeps finding the difference.

This takes a different route. A donor landscape, always from another protein,
supplies its own measured coefficients. The target's sites and substitutions map
onto the donor's, and the donor's numbers come across rescaled. Nothing is
sampled, so correlation, sparsity and heteroscedasticity arrive together and
already have the shape measured data has.

It is a nonparametric prior over landscapes rather than a copy of one. The donor
is chosen by seed, and it is never the protein being evaluated.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mutascape.completion import (
    MutationKey,
    _observation_map,
    _single_effect_scale,
    _subsets,
    _training_readout,
)


@dataclass(frozen=True)
class DonorLandscape:
    """One measured landscape's normalised order-two coefficients."""

    name: str
    protein: str
    site_count: int
    alphabet: tuple[int, ...]
    coefficients: dict[tuple[int, int, int, int], float]

    @property
    def size(self) -> int:
        return len(self.coefficients)


def build_donor(dataset) -> DonorLandscape | None:
    """Extract one landscape's measured interactions on a normalised scale."""
    observed = _observation_map(dataset.variants, dataset.wild_type_sequence)
    wt = observed[()]
    singles = {key[0]: value - wt for key, value in observed.items() if len(key) == 1}
    if len(singles) < 4:
        return None
    finite = np.asarray([value - wt for value in observed.values()], dtype=np.float64)
    readout = _training_readout(singles, finite)
    if readout is None:
        return None
    latent = {key: float(readout.inverse(value - wt)) for key, value in observed.items()}
    preimages = np.asarray(
        [float(readout.inverse(value)) for value in singles.values()], dtype=np.float64
    )
    scale = _single_effect_scale(preimages)
    if scale <= 1e-12:
        return None

    sites = sorted({position for position, _ in singles})
    site_index = {position: index for index, position in enumerate(sites)}
    residues: dict[int, list[str]] = {}
    for position, amino_acid in sorted(singles):
        residues.setdefault(position, []).append(amino_acid)
    residue_index = {
        (position, amino_acid): order
        for position, options in residues.items()
        for order, amino_acid in enumerate(options)
    }

    coefficients: dict[tuple[int, int, int, int], float] = {}
    for genotype in latent:
        if len(genotype) != 2:
            continue
        if any(subset not in latent for subset in _subsets(genotype)):
            continue
        if any(mutation not in residue_index for mutation in genotype):
            continue
        value = sum((-1.0) ** (2 - len(subset)) * latent[subset] for subset in _subsets(genotype))
        left, right = genotype
        key = (
            site_index[left[0]],
            residue_index[left],
            site_index[right[0]],
            residue_index[right],
        )
        coefficients[key] = value / scale

    if len(coefficients) < 20:
        return None
    return DonorLandscape(
        name=dataset.name,
        protein=dataset.protein,
        site_count=len(sites),
        alphabet=tuple(len(residues[position]) for position in sites),
        coefficients=coefficients,
    )


def build_donors(datasets) -> list[DonorLandscape]:
    """Extract every usable donor from the training landscapes."""
    donors = [build_donor(dataset) for dataset in datasets]
    return [donor for donor in donors if donor is not None]


@dataclass(frozen=True)
class TransplantIndex:
    """A fixed mapping from one target's substitutions onto a donor's."""

    donor: DonorLandscape
    site_of: dict[int, int]
    residue_of: dict[tuple[int, str], int]

    def coefficient(self, interaction: MutationKey) -> float | None:
        """Return the donor's measured value for this pair, if it holds one."""
        if len(interaction) != 2:
            return None
        left, right = interaction
        if left not in self.residue_of or right not in self.residue_of:
            return None
        key = (
            self.site_of[left[0]],
            self.residue_of[left],
            self.site_of[right[0]],
            self.residue_of[right],
        )
        value = self.donor.coefficients.get(key)
        if value is None:
            # The donor never measured that combination, so try the mirrored key
            # before giving up and letting the caller fall back.
            key = (key[2], key[3], key[0], key[1])
            value = self.donor.coefficients.get(key)
        return value


def build_index(
    donors: list[DonorLandscape],
    single_effects: dict[tuple[int, str], float],
    seed: int,
    *,
    exclude_protein: str | None = None,
) -> TransplantIndex | None:
    """Pick a donor by seed and map the target's substitutions onto it.

    Sites map in order and residues wrap around the donor's alphabet, so every
    target substitution receives a donor counterpart even when the donor library
    is smaller.
    """
    usable = [d for d in donors if d.protein != exclude_protein]
    if not usable:
        return None
    rng = np.random.default_rng(seed)
    positions = sorted({position for position, _ in single_effects})
    # Prefer a donor with at least as many sites, so distinct target sites do
    # not collapse onto one donor site.
    roomy = [d for d in usable if d.site_count >= len(positions)]
    pool = roomy or usable
    donor = pool[int(rng.integers(len(pool)))]

    site_of = {position: index % donor.site_count for index, position in enumerate(positions)}
    residue_of: dict[tuple[int, str], int] = {}
    for position in positions:
        options = sorted(amino_acid for site, amino_acid in single_effects if site == position)
        width = donor.alphabet[site_of[position]]
        for order, amino_acid in enumerate(options):
            residue_of[(position, amino_acid)] = order % width
    return TransplantIndex(donor=donor, site_of=site_of, residue_of=residue_of)


def summarise(donors: list[DonorLandscape]) -> pd.DataFrame:
    """One row per donor, for reporting what the pool contains."""
    return pd.DataFrame(
        [
            {
                "donor": donor.name,
                "protein": donor.protein,
                "sites": donor.site_count,
                "coefficients": donor.size,
            }
            for donor in donors
        ]
    )
