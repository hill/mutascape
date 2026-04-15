"""
Contact map extraction from protein structures.

Parses PDB/mmCIF files, extracts C-beta coordinates (C-alpha for glycine),
computes pairwise distance matrices, and applies a threshold to produce
binary contact maps. These contact maps define the epistatic topology in
mutascape fitness landscapes.

The biophysical rationale: residues within ~8 Angstroms (C-beta to C-beta)
can interact through van der Waals forces, hydrogen bonds, salt bridges,
and hydrophobic packing. Mutations at contacting positions are more likely
to show non-additive fitness effects (epistasis) because they physically
influence each other's local environment.

References
----------
- Jumper et al. (2021). Highly accurate protein structure prediction with AlphaFold.
- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure
  with a language model. (ESMFold)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ContactMap:
    """Binary contact map between protein residue positions.

    Stores a symmetric boolean matrix where entry (i, j) is True if
    positions i and j are within the distance threshold in 3D space.

    Attributes
    ----------
    matrix : NDArray[np.bool_]
        Symmetric boolean contact matrix of shape (n_positions, n_positions).
    positions : tuple[int, ...]
        Residue numbers (1-indexed, PDB numbering) corresponding to each
        row/column of the matrix.
    threshold : float
        Distance threshold in Angstroms used to define contacts.
    distances : NDArray[np.float64] | None
        Full pairwise distance matrix (Angstroms), if retained.
    """

    matrix: NDArray[np.bool_]
    positions: tuple[int, ...]
    threshold: float
    distances: NDArray[np.float64] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        n = len(self.positions)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"Contact matrix shape {self.matrix.shape} does not match "
                f"number of positions ({n})"
            )

    @property
    def n_positions(self) -> int:
        """Number of residue positions in the contact map."""
        return len(self.positions)

    @property
    def n_contacts(self) -> int:
        """Number of unique contacting pairs (excluding self-contacts)."""
        return int(np.sum(np.triu(self.matrix, k=1)))

    @property
    def density(self) -> float:
        """Fraction of possible pairs that are in contact."""
        n = self.n_positions
        max_pairs = n * (n - 1) / 2
        return self.n_contacts / max_pairs if max_pairs > 0 else 0.0

    def contact_pairs(self) -> list[tuple[int, int]]:
        """Return list of contacting (position_i, position_j) pairs.

        Returns residue numbers (PDB numbering), not matrix indices.
        """
        rows, cols = np.where(np.triu(self.matrix, k=1))
        return [(self.positions[r], self.positions[c]) for r, c in zip(rows, cols)]

    def submap(self, positions: Sequence[int]) -> ContactMap:
        """Extract a sub-contact-map for a subset of positions.

        Parameters
        ----------
        positions : sequence of int
            Residue numbers to retain (must be a subset of self.positions).

        Returns
        -------
        ContactMap
            New contact map restricted to the specified positions.
        """
        pos_set = set(positions)
        indices = [i for i, p in enumerate(self.positions) if p in pos_set]
        if len(indices) != len(pos_set):
            missing = pos_set - set(self.positions)
            raise ValueError(f"Positions not found in contact map: {missing}")

        ix = np.array(indices)
        sub_matrix = self.matrix[np.ix_(ix, ix)]
        sub_positions = tuple(self.positions[i] for i in indices)

        sub_distances = None
        if self.distances is not None:
            sub_distances = self.distances[np.ix_(ix, ix)]

        return ContactMap(
            matrix=sub_matrix,
            positions=sub_positions,
            threshold=self.threshold,
            distances=sub_distances,
        )

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str | Path,
        threshold: float = 8.0,
        chain_id: str | None = None,
        positions: Sequence[int] | None = None,
    ) -> ContactMap:
        """Build a contact map from a PDB or mmCIF file.

        Extracts C-beta atoms (C-alpha for glycine), computes pairwise
        Euclidean distances, and thresholds to produce a binary contact map.

        Parameters
        ----------
        pdb_path : str or Path
            Path to PDB or mmCIF structure file.
        threshold : float
            Distance cutoff in Angstroms (default 8.0). Residue pairs with
            C-beta distance <= threshold are considered in contact.
        chain_id : str or None
            Which chain to extract. If None, uses the first chain.
        positions : sequence of int or None
            If provided, restrict the contact map to these residue numbers.
            If None, includes all resolved residues.

        Returns
        -------
        ContactMap
            Binary contact map with distance information retained.
        """
        coords, residue_ids = _extract_cbeta_coords(pdb_path, chain_id)

        # Filter to requested positions if specified
        if positions is not None:
            pos_set = set(positions)
            mask = [i for i, rid in enumerate(residue_ids) if rid in pos_set]
            coords = coords[mask]
            residue_ids = [residue_ids[i] for i in mask]

        # Compute pairwise C-beta distance matrix
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        # Threshold to binary contacts (exclude self-contacts on diagonal)
        contact_matrix = dist_matrix <= threshold
        np.fill_diagonal(contact_matrix, False)

        return cls(
            matrix=contact_matrix,
            positions=tuple(residue_ids),
            threshold=threshold,
            distances=dist_matrix,
        )

    @classmethod
    def from_distance_matrix(
        cls,
        distances: NDArray[np.float64],
        positions: Sequence[int],
        threshold: float = 8.0,
    ) -> ContactMap:
        """Build a contact map from a precomputed distance matrix.

        Parameters
        ----------
        distances : NDArray
            Symmetric pairwise distance matrix of shape (n, n).
        positions : sequence of int
            Residue numbers corresponding to rows/columns.
        threshold : float
            Distance cutoff in Angstroms.

        Returns
        -------
        ContactMap
        """
        contact_matrix = distances <= threshold
        np.fill_diagonal(contact_matrix, False)
        return cls(
            matrix=contact_matrix,
            positions=tuple(positions),
            threshold=threshold,
            distances=distances,
        )


def _extract_cbeta_coords(
    pdb_path: str | Path,
    chain_id: str | None = None,
) -> tuple[NDArray[np.float64], list[int]]:
    """Extract C-beta coordinates from a PDB/mmCIF file.

    Uses C-alpha for glycine residues (which lack a C-beta atom).

    Parameters
    ----------
    pdb_path : str or Path
        Path to structure file (.pdb or .cif/.mmcif).
    chain_id : str or None
        Chain to extract. If None, uses the first chain found.

    Returns
    -------
    coords : NDArray of shape (n_residues, 3)
        C-beta (or C-alpha for Gly) coordinates.
    residue_ids : list of int
        PDB residue sequence numbers, 1-indexed.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    pdb_path = Path(pdb_path)
    suffix = pdb_path.suffix.lower()

    if suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("protein", str(pdb_path))
    model = structure[0]  # first model

    # Select chain
    if chain_id is not None:
        chain = model[chain_id]
    else:
        chain = next(iter(model))

    coords = []
    residue_ids = []

    for residue in chain.get_residues():
        # Skip hetero atoms (water, ligands, etc.)
        if residue.id[0] != " ":
            continue

        resname = residue.get_resname().strip()
        res_id = residue.id[1]

        # Use CB for most residues, CA for glycine
        if resname == "GLY":
            atom_name = "CA"
        else:
            atom_name = "CB"

        if atom_name in residue:
            coords.append(residue[atom_name].get_vector().get_array())
            residue_ids.append(res_id)
        elif "CA" in residue:
            # Fallback to CA if CB is missing (e.g. incomplete structure)
            coords.append(residue["CA"].get_vector().get_array())
            residue_ids.append(res_id)

    if len(coords) == 0:
        raise ValueError(f"No residues extracted from {pdb_path} (chain={chain_id})")

    return np.array(coords, dtype=np.float64), residue_ids
