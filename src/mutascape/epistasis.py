"""Epistatic coupling models for synthetic fitness landscapes.

Each model generates a pairwise coupling tensor J_ij(a, b) that defines how
mutations at contacting positions interact. The coupling topology (which pairs
get non-zero J) is determined by a structural contact map. The coupling
*strengths* are determined by the model.

Models implemented:
- RandomEpistasis: couplings drawn from N(0, sigma) for contacting pairs
- SignConsistentEpistasis: coupling signs follow the linear-sum direction of
  the two single-site effects (amplifies whichever way the singles point)
- MSACouplingsEpistasis: imports pairwise couplings from an MSA-derived tensor

References:
- Thomas et al. (2022). Tuned Fitness Landscapes for Benchmarking Model-Guided
  Protein Design. (SLIP - Potts model approach)
- Sandhu et al. (2025). Investigating the determinants of performance in machine
  learning for protein fitness. (NK landscape approach)
"""

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
        pos_i, pos_j : int
            Residue positions (PDB numbering).
        aa_i, aa_j : str
            Single-letter amino acid codes.

        Returns
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
        contact_map : ContactMap
            Defines which position pairs get non-zero couplings.
        single_site_effects : dict or None
            Maps position -> {amino_acid: fitness_effect}. Required by
            some models (e.g. SignConsistentEpistasis) to determine
            coupling signs.

        Returns
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
    """Couplings that amplify the linear-sum direction of single-site effects.

    For positions i, j with single-site effects h_i(a) and h_j(b):

        sign(J_ij(a, b)) = sign(h_i(a) + h_j(b))
        |J_ij(a, b)| ~ |N(0, sigma)|

    Behaviour on double mutants relative to the additive prediction:

    - both beneficial: J > 0, benefit is reinforced (cooperation)
    - both deleterious: J < 0, damage is reinforced (synthetic-lethal-like)
    - mixed signs: J takes the sign of whichever single dominates the sum,
      accentuating that direction

    Couplings involving a wild-type AA (effect = 0) are set to zero, so the
    J(a,b) - J(wt,b) - J(a,wt) + J(wt,wt) baseline correction inside
    Landscape.evaluate drops out cleanly to J(a, b).

    Note: the simpler product rule sign(J) = sign(h_a) * sign(h_b) would
    instead *compensate* both-deleterious pairs (positive J added to a
    negative sum softens it). That compensation mechanism overlaps with the
    saturating global-epistasis function in `mutascape.nonlinearity`; this amplification
    rule is the complementary "direct pairwise reinforcement" mechanism.

    Parameters
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

                    magnitude = abs(rng.normal(0, self.sigma))

                    # Sign of the pairwise term reinforces whichever direction
                    # the linear sum of singles already points. Wild-type-like
                    # effects (h = 0) zero the coupling so the J(wt, *) baseline
                    # drops out in Landscape.evaluate.
                    if h_a == 0.0 or h_b == 0.0:
                        sign = 0.0
                    else:
                        sign = np.sign(h_a + h_b)

                    matrix[a_idx, b_idx] = sign * magnitude

            # Symmetrise
            matrix = (matrix + matrix.T) / 2.0
            couplings[(pos_i, pos_j)] = matrix

        return CouplingTensor(
            couplings=couplings,
            positions=contact_map.positions,
            aa_index=AA_INDEX,
        )


@dataclass
class MSACouplingsEpistasis(EpistasisModel):
    """Pairwise couplings derived from MSA coevolution analysis.

    Drop-in replacement for RandomEpistasis. Couplings come from a precomputed
    tensor `J[i, j, a, b]` (typically plmDCA, EVcouplings, or log-odds from a
    multiple sequence alignment) rather than Gaussian-random values. The
    structural prior still applies: only contacting pairs (per contact_map)
    keep their J; non-contact pairs are zeroed so the "structure-informed"
    story is preserved.

    Parameters
    coupling_matrix : NDArray[float64]
        Shape (L, L, 20, 20). J[i, j, a, b] is the coupling between amino acid
        a at the i-th focal position and amino acid b at the j-th. Should be
        symmetric on the position axes so that J[i, j] = J[j, i].T.
    position_order : tuple[int, ...]
        Ordered focal positions matching the first two axes of coupling_matrix.
        Must be a superset of the contact_map positions passed to `generate`.
    scale : float
        Multiplier applied to coupling_matrix (default 1.0). Useful for
        regularising the MSA signal so it lives on the same scale as the
        additive h_i from DMS.
    alphabet : str
        20-letter AA alphabet; order must match the last two axes of
        coupling_matrix. Default "ACDEFGHIKLMNPQRSTVWY".
    restrict_to_contacts : bool
        If True (default), only contacting pairs from the contact map get
        non-zero couplings. If False, all precomputed MSA couplings are used
        regardless of structural contact.
    """

    coupling_matrix: NDArray[np.float64]
    position_order: tuple[int, ...]
    scale: float = 1.0
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY"
    restrict_to_contacts: bool = True

    def __post_init__(self):
        L = len(self.position_order)
        K = len(self.alphabet)
        expected = (L, L, K, K)
        if self.coupling_matrix.shape != expected:
            raise ValueError(
                f"coupling_matrix shape {self.coupling_matrix.shape} does not "
                f"match expected {expected} for {L} positions and {K}-letter alphabet"
            )

    def generate(
        self,
        contact_map: ContactMap,
        single_site_effects: dict[int, dict[str, float]] | None = None,
    ) -> CouplingTensor:
        pos_to_idx = {p: i for i, p in enumerate(self.position_order)}
        aa_remap = {aa: i for i, aa in enumerate(self.alphabet)}
        mutascape_aa_order = AMINO_ACIDS

        if self.restrict_to_contacts:
            pairs = contact_map.contact_pairs()
        else:
            pairs = [(a, b) for a in contact_map.positions for b in contact_map.positions if a < b]

        missing = [p for p in contact_map.positions if p not in pos_to_idx]
        if missing:
            raise ValueError(
                f"coupling_matrix position_order is missing contact-map positions: {missing}"
            )

        reorder = np.array([aa_remap[a] for a in mutascape_aa_order])

        couplings: dict[tuple[int, int], NDArray[np.float64]] = {}
        for pos_i, pos_j in pairs:
            i, j = pos_to_idx[pos_i], pos_to_idx[pos_j]
            J = self.coupling_matrix[i, j] * self.scale
            couplings[(pos_i, pos_j)] = J[np.ix_(reorder, reorder)]

        return CouplingTensor(
            couplings=couplings,
            positions=contact_map.positions,
            aa_index=AA_INDEX,
        )

    @classmethod
    def from_npz(
        cls,
        path,
        scale: float = 1.0,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
        restrict_to_contacts: bool = True,
    ) -> MSACouplingsEpistasis:
        """Load from an .npz with keys `coupling_matrix` (L,L,20,20 float) and
        `positions` (L int)."""
        data = np.load(path)
        return cls(
            coupling_matrix=data["coupling_matrix"].astype(np.float64),
            position_order=tuple(int(p) for p in data["positions"]),
            scale=scale,
            alphabet=alphabet,
            restrict_to_contacts=restrict_to_contacts,
        )

    @classmethod
    def from_msa_fasta(
        cls,
        fasta_path,
        positions: tuple[int, ...],
        reference_id: str | None = None,
        pseudocount: float = 0.5,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
        scale: float = 1.0,
        restrict_to_contacts: bool = True,
    ) -> MSACouplingsEpistasis:
        """Compute log-odds couplings J[i,j,a,b] = log p(a,b|i,j) - log p(a|i) - log p(b|j)
        from an aligned FASTA MSA.

        `positions` are indices into the alignment (0-based) that correspond to
        the focal residues. If `reference_id` is given, positions are interpreted
        as indices in that reference sequence (ungapped), which are then mapped
        to alignment columns by skipping gaps.

        Uses a Laplace-smoothed maximum likelihood estimator, not plmDCA. This
        is a fast heuristic that captures coevolution signal without external
        tools; use `from_npz` with a plmDCA-derived matrix for the full-fidelity
        path.
        """
        from Bio import SeqIO

        seqs = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not seqs:
            raise ValueError(f"No sequences in {fasta_path}")

        if reference_id is not None:
            ref = next((s for s in seqs if s.id == reference_id), None)
            if ref is None:
                raise ValueError(f"reference_id {reference_id} not found in {fasta_path}")
            ungapped_to_aligned = []
            for col, c in enumerate(str(ref.seq)):
                if c != "-":
                    ungapped_to_aligned.append(col)
            aligned_cols = tuple(ungapped_to_aligned[p] for p in positions)
        else:
            aligned_cols = tuple(positions)

        aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
        K = len(alphabet)
        L = len(aligned_cols)

        matrix = np.array([list(str(s.seq)) for s in seqs])
        col_data = [matrix[:, c] for c in aligned_cols]

        # Drop sequences that have a gap or non-standard AA at any focal column.
        keep = np.ones(len(seqs), dtype=bool)
        for col in col_data:
            keep &= np.array([c in aa_to_idx for c in col])
        n_kept = int(keep.sum())
        if n_kept < 50:
            raise ValueError(f"Too few MSA sequences ({n_kept}) for coupling estimation")

        idx_cols = np.stack([np.array([aa_to_idx[c] for c in col[keep]]) for col in col_data])

        J = np.zeros((L, L, K, K), dtype=np.float64)
        marginals = np.zeros((L, K), dtype=np.float64)
        for i in range(L):
            counts = np.bincount(idx_cols[i], minlength=K).astype(np.float64) + pseudocount
            marginals[i] = counts / counts.sum()

        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                joint = np.zeros((K, K), dtype=np.float64) + pseudocount / K
                np.add.at(joint, (idx_cols[i], idx_cols[j]), 1.0)
                joint /= joint.sum()
                lo = np.log(joint) - np.log(marginals[i])[:, None] - np.log(marginals[j])[None, :]
                J[i, j] = lo

        return cls(
            coupling_matrix=J,
            position_order=tuple(positions),
            scale=scale,
            alphabet=alphabet,
            restrict_to_contacts=restrict_to_contacts,
        )


@dataclass(frozen=True)
class ContactNKTensor:
    """Per-position NK-style functions over (position + structural neighbours).

    Each entry of `per_position_functions` is a random Gaussian tensor of shape
    (20,) * (1 + K_i), where K_i is the number of structural neighbours of
    position i under `neighbours_by_position`. The tensor is pinned so the
    all-wildtype combination evaluates to 0, which keeps single-mutant
    predictions consistent with DMS data when this component is added on top
    of the additive `h_i` terms in `Landscape.evaluate`.

    Attributes
    per_position_functions : dict[int, NDArray[np.float64]]
        Maps position -> random function tensor. The tensor's first axis is
        the AA at `position` and remaining axes are AAs at the ordered
        neighbours from `neighbours_by_position[position]`.
    neighbours_by_position : dict[int, tuple[int, ...]]
        Maps position -> ordered tuple of neighbour positions.
    positions : tuple[int, ...]
        All positions covered by the tensor.
    aa_index : dict[str, int]
        Maps single-letter amino acid codes to matrix indices.
    """

    per_position_functions: dict[int, NDArray[np.float64]]
    neighbours_by_position: dict[int, tuple[int, ...]]
    positions: tuple[int, ...]
    aa_index: dict[str, int]


@dataclass
class ContactNKEpistasis(EpistasisModel):
    """Position-centric NK-style couplings informed by the contact map.

    Classic NK (Kauffman) defines each position's fitness contribution as a
    random function over itself plus K random partner positions. Here the
    partners are taken from the structural contact map, so the higher-order
    epistasis is grounded in the same "structure determines topology" prior
    as the existing pairwise model.

    Each position i contributes a random Gaussian function f_i of the joint
    amino-acid state at {i} ∪ N(i), where N(i) is the set of structural
    neighbours (capped at `max_neighbours` closest contacts if specified).
    The function is pinned so f_i(all wildtype) = 0, preserving the DMS
    single-site identity when this model is combined with `h_i`.

    Because f_i is drawn over the (1 + K_i)-tuple of AAs, it contributes
    variance at every Walsh order from 1 through 1 + K_i. In practice the
    low-order components are small compared to the explicit `h_i` and
    `J_ij` already in the landscape, and the higher-order components are
    what a pairwise-only model cannot reach.

    Parameters
    sigma : float
        Standard deviation of the random NK functions. Controls higher-order
        ruggedness strength. Typical values: 0.1-1.0.
    max_neighbours : int or None
        Cap the number of structural neighbours per position. None means
        "use all contacts." Lower values keep compute bounded for large
        proteins (per-position tensor size is 20^(1+K)).
    seed : int or None
        Random seed for reproducibility.
    """

    sigma: float = 0.3
    max_neighbours: int | None = None
    seed: int | None = None

    def generate(
        self,
        contact_map: ContactMap,
        single_site_effects: dict[int, dict[str, float]] | None = None,
    ) -> ContactNKTensor:
        rng = np.random.default_rng(self.seed)
        n_aa = len(AMINO_ACIDS)

        per_position: dict[int, NDArray[np.float64]] = {}
        neighbours_by_pos: dict[int, tuple[int, ...]] = {}

        for pos in contact_map.positions:
            neighbours = contact_map.neighbours(pos)
            if self.max_neighbours is not None and len(neighbours) > self.max_neighbours:
                neighbours = neighbours[: self.max_neighbours]
            neighbours_by_pos[pos] = neighbours

            shape = (n_aa,) * (1 + len(neighbours))
            f = rng.normal(0.0, self.sigma, size=shape)
            per_position[pos] = f

        return ContactNKTensor(
            per_position_functions=per_position,
            neighbours_by_position=neighbours_by_pos,
            positions=contact_map.positions,
            aa_index=AA_INDEX,
        )
