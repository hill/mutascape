"""
Epistatic coupling models for synthetic fitness landscapes.

Each model generates a pairwise coupling tensor J_ij(a, b) that defines how
mutations at contacting positions interact. The coupling topology (which pairs
get non-zero J) is determined by a structural contact map. The coupling
*strengths* are determined by the model.

Models implemented
------------------
- RandomEpistasis: couplings drawn from N(0, sigma) for contacting pairs
- SignConsistentEpistasis: coupling signs respect single-site effect signs
  (beneficial mutations at contacting sites cooperate)

Planned
-------
- PottsEpistasis: import couplings from inverse Potts model fits (cf. SLIP,
  Thomas et al. 2022)

References
----------
- Thomas et al. (2022). Tuned Fitness Landscapes for Benchmarking Model-Guided
  Protein Design. (SLIP - Potts model approach)
- Sandhu et al. (2025). Investigating the determinants of performance in machine
  learning for protein fitness. (NK landscape approach)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mutascape.contacts import ContactMap


@dataclass(frozen=True)
class CouplingTensor:
    """Sparse pairwise coupling tensor J_ij(a, b).

    Only stores couplings for contacting position pairs (from the contact map).
    For a pair of positions (i, j), the coupling matrix is a 20x20 matrix
    (one entry per amino acid pair).

    Attributes
    ----------
    couplings : dict[tuple[int, int], NDArray[np.float64]]
        Maps (position_i, position_j) -> coupling matrix of shape (20, 20).
        Keys are ordered such that position_i < position_j.
        Positions use PDB residue numbering.
    positions : tuple[int, ...]
        All positions involved in the coupling tensor.
    aa_index : dict[str, int]
        Maps single-letter amino acid codes to matrix indices (0-19).
    """

    couplings: dict[tuple[int, int], NDArray[np.float64]]
    positions: tuple[int, ...]
    aa_index: dict[str, int]

    @property
    def n_pairs(self) -> int:
        """Number of position pairs with non-zero couplings."""
        return len(self.couplings)

    def get_coupling(self, pos_i: int, pos_j: int, aa_i: str, aa_j: str) -> float:
        """Look up the coupling between two specific mutations.

        Parameters
        ----------
        pos_i, pos_j : int
            Residue positions (PDB numbering).
        aa_i, aa_j : str
            Single-letter amino acid codes.

        Returns
        -------
        float
            Coupling value J_ij(aa_i, aa_j). Returns 0.0 if the pair
            is not in contact.
        """
        key = (min(pos_i, pos_j), max(pos_i, pos_j))
        if key not in self.couplings:
            return 0.0

        matrix = self.couplings[key]
        idx_i = self.aa_index[aa_i]
        idx_j = self.aa_index[aa_j]

        # If we swapped the order, transpose the lookup
        if pos_i <= pos_j:
            return float(matrix[idx_i, idx_j])
        else:
            return float(matrix[idx_j, idx_i])

    def mean_coupling_magnitude(self) -> float:
        """Average absolute coupling strength across all pairs and AA combos."""
        if not self.couplings:
            return 0.0
        magnitudes = [np.mean(np.abs(m)) for m in self.couplings.values()]
        return float(np.mean(magnitudes))


# Standard amino acid alphabet
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class EpistasisModel(ABC):
    """Base class for epistatic coupling models.

    Subclasses implement `generate()` which produces a CouplingTensor
    given a contact map and (optionally) single-site fitness effects.
    """

    @abstractmethod
    def generate(
        self,
        contact_map: ContactMap,
        single_site_effects: dict[int, dict[str, float]] | None = None,
    ) -> CouplingTensor:
        """Generate pairwise couplings for all contacting pairs.

        Parameters
        ----------
        contact_map : ContactMap
            Defines which position pairs get non-zero couplings.
        single_site_effects : dict or None
            Maps position -> {amino_acid: fitness_effect}. Required by
            some models (e.g. SignConsistentEpistasis) to determine
            coupling signs.

        Returns
        -------
        CouplingTensor
            Sparse coupling tensor for the landscape.
        """
        ...


@dataclass
class RandomEpistasis(EpistasisModel):
    """Random Gaussian couplings for contacting residue pairs.

    The simplest coupling model. For each contacting pair (i, j), draws
    a 20x20 coupling matrix from N(0, sigma). The matrix is made symmetric
    (J_ij(a,b) = J_ij(b,a)) to ensure the coupling is order-independent.

    This is analogous to the random interactions in NK landscapes (Sandhu
    et al. 2025) but restricted to structurally plausible pairs rather
    than K random neighbours.

    Parameters
    ----------
    sigma : float
        Standard deviation of coupling distribution. Controls epistatic
        strength. Typical values: 0.1-1.0 (relative to DMS fitness scale).
    seed : int or None
        Random seed for reproducibility.
    """

    sigma: float = 0.3
    seed: int | None = None

    def generate(
        self,
        contact_map: ContactMap,
        single_site_effects: dict[int, dict[str, float]] | None = None,
    ) -> CouplingTensor:
        rng = np.random.default_rng(self.seed)
        pairs = contact_map.contact_pairs()
        n_aa = len(AMINO_ACIDS)

        couplings: dict[tuple[int, int], NDArray[np.float64]] = {}
        for pos_i, pos_j in pairs:
            # Draw random coupling matrix
            raw = rng.normal(0, self.sigma, size=(n_aa, n_aa))
            # Make symmetric: J_ij(a, b) = J_ij(b, a)
            matrix = (raw + raw.T) / 2.0
            couplings[(pos_i, pos_j)] = matrix

        return CouplingTensor(
            couplings=couplings,
            positions=contact_map.positions,
            aa_index=AA_INDEX,
        )


@dataclass
class SignConsistentEpistasis(EpistasisModel):
    """Couplings whose signs respect single-site fitness effects.

    The biophysical intuition: if two mutations are individually beneficial
    and are at contacting positions, they likely cooperate (positive coupling).
    If both are deleterious, they compound the damage (positive coupling,
    amplifying the deleterious effect). If one is beneficial and one
    deleterious, they may partially cancel (negative coupling).

    More precisely, for positions i, j with single-site effects h_i(a) and
    h_j(b):

        sign(J_ij(a, b)) = sign(h_i(a)) * sign(h_j(b))
        |J_ij(a, b)| ~ |N(0, sigma)|

    This produces landscapes with smoother fitness ridges connecting
    beneficial mutations, which is biophysically plausible for many
    protein fitness landscapes.

    Parameters
    ----------
    sigma : float
        Scale of coupling magnitudes.
    seed : int or None
        Random seed for reproducibility.
    """

    sigma: float = 0.3
    seed: int | None = None

    def generate(
        self,
        contact_map: ContactMap,
        single_site_effects: dict[int, dict[str, float]] | None = None,
    ) -> CouplingTensor:
        if single_site_effects is None:
            raise ValueError(
                "SignConsistentEpistasis requires single_site_effects to determine "
                "coupling signs. Pass the DMS fitness effects dict."
            )

        rng = np.random.default_rng(self.seed)
        pairs = contact_map.contact_pairs()
        n_aa = len(AMINO_ACIDS)

        couplings: dict[tuple[int, int], NDArray[np.float64]] = {}
        for pos_i, pos_j in pairs:
            matrix = np.zeros((n_aa, n_aa), dtype=np.float64)
            effects_i = single_site_effects.get(pos_i, {})
            effects_j = single_site_effects.get(pos_j, {})

            for a_idx, aa_a in enumerate(AMINO_ACIDS):
                h_a = effects_i.get(aa_a, 0.0)
                for b_idx, aa_b in enumerate(AMINO_ACIDS):
                    h_b = effects_j.get(aa_b, 0.0)

                    # Magnitude from half-normal
                    magnitude = abs(rng.normal(0, self.sigma))

                    # Sign from product of single-site effect signs
                    # Both beneficial or both deleterious -> positive (cooperative/compounding)
                    # Mixed -> negative (antagonistic)
                    if h_a == 0.0 or h_b == 0.0:
                        sign = 0.0  # no coupling if either is wild-type-like
                    else:
                        sign = np.sign(h_a) * np.sign(h_b)

                    matrix[a_idx, b_idx] = sign * magnitude

            # Symmetrise
            matrix = (matrix + matrix.T) / 2.0
            couplings[(pos_i, pos_j)] = matrix

        return CouplingTensor(
            couplings=couplings,
            positions=contact_map.positions,
            aa_index=AA_INDEX,
        )
