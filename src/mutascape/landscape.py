"""
Core fitness landscape class.

Combines real single-site DMS data with structurally informed epistatic
couplings to create synthetic combinatorial fitness landscapes. The landscape
can evaluate arbitrary multi-mutant variants and sample random variants at
specified mutation depths.

The fitness function:

    f(x) = sum_i h_i(x_i) + alpha * sum_{i<j} C_ij * J_ij(x_i, x_j) + noise

Where h_i are real DMS-measured single-site effects, J_ij are pairwise
couplings from the chosen epistasis model, C_ij is the contact map, and
alpha scales epistatic strength (alpha=0 recovers the additive landscape).

This approach differs from SLIP (Thomas et al. 2022), which derives both h
and J from Potts model fits to MSAs, and from NK landscapes (Sandhu et al.
2025), which use abstract random interactions. mutascape grounds single-site
effects in experimental data and epistatic topology in 3D structure.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    AMINO_ACIDS,
    CouplingTensor,
    EpistasisModel,
)
from mutascape.mutant import Mutant
from mutascape.scoring import PLMScorer


class Variant(NamedTuple):
    """A variant in a combinatorial fitness landscape."""

    mutations: Mutant
    sequence: str
    fitness: float


@dataclass
class Landscape:
    """Synthetic combinatorial protein fitness landscape.

    Attributes
    ----------
    wild_type : dict[int, str]
        Maps position -> wild-type amino acid (single letter).
    single_site_effects : dict[int, dict[str, float]]
        Maps position -> {amino_acid: fitness_effect}. Effects are relative
        to wild type (wild-type AA has effect 0.0 by convention).
    contact_map : ContactMap
        Structural contact map defining epistatic topology.
    couplings : CouplingTensor
        Pairwise epistatic couplings from the chosen model.
    alpha : float
        Global epistatic strength multiplier. alpha=0 gives an additive
        landscape; larger values increase ruggedness.
    noise_sigma : float
        Standard deviation of Gaussian measurement noise (0.0 = no noise).
    seed : int or None
        Random seed for noise and sampling.
    """

    wild_type: dict[int, str]
    single_site_effects: dict[int, dict[str, float]]
    contact_map: ContactMap
    couplings: CouplingTensor
    alpha: float = 1.0
    noise_sigma: float = 0.0
    seed: int | None = None
    wt_sequence: str | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.wt_sequence is not None:
            self._validate_wt_sequence()

    def __repr__(self) -> str:
        positions = ", ".join(
            f"{pos}({self.wild_type[pos]})" for pos in self.positions
        )
        return (
            f"Landscape(positions=[{positions}], "
            f"alpha={self.alpha}, "
            f"contacts={self.contact_map.n_contacts}, "
            f"seq_len={len(self.wt_sequence) if self.wt_sequence else '?'})"
        )

    @property
    def positions(self) -> list[int]:
        """Mutable positions in the landscape."""
        return sorted(self.wild_type.keys())

    @property
    def n_positions(self) -> int:
        """Number of mutable positions."""
        return len(self.wild_type)

    def _validate_wt_sequence(self) -> None:
        """Check that wt_sequence is consistent with wild_type position dict."""
        seq = self.wt_sequence
        assert seq is not None
        for pos, expected_aa in self.wild_type.items():
            idx = pos - 1
            if idx < 0 or idx >= len(seq):
                raise ValueError(
                    f"Position {pos} is out of range for WT sequence of length {len(seq)}"
                )
            actual_aa = seq[idx]
            if actual_aa != expected_aa:
                raise ValueError(
                    f"WT sequence mismatch at position {pos}: "
                    f"sequence has '{actual_aa}' but DMS data says '{expected_aa}'"
                )

    def mutant_sequence(self, mutations: dict[int, str]) -> str:
        """Apply mutations to the wild-type sequence and return the full mutant string.

        Parameters
        ----------
        mutations : dict[int, str]
            Maps position -> mutant amino acid.

        Returns
        -------
        str
            Full mutant amino acid sequence.
        """
        if self.wt_sequence is None:
            raise ValueError(
                "wt_sequence is not set. Pass wt_fasta to Landscape.from_dms() "
                "or set wt_sequence directly."
            )
        seq = list(self.wt_sequence)
        for pos, aa in mutations.items():
            seq[pos - 1] = aa
        return "".join(seq)

    def __getitem__(self, mutations: dict[int, str] | str) -> float:
        """Look up fitness: landscape[Mutant("A5G")] or landscape["A5G+L10V"]."""
        if isinstance(mutations, str):
            mutations = Mutant(mutations)
        return self.evaluate(mutations)

    def evaluate(self, mutations: dict[int, str]) -> float:
        """Compute fitness of a variant specified by its mutations.

        Parameters
        ----------
        mutations : dict[int, str]
            Maps position -> mutant amino acid. Positions not included
            are assumed to be wild type.

        Returns
        -------
        float
            Predicted fitness value. A purely additive prediction (alpha=0)
            would just sum the single-site effects.

        Examples
        --------
        >>> landscape.evaluate({57: "G", 72: "V"})
        0.42
        """
        # Additive component: sum of single-site effects for mutated positions
        additive = 0.0
        for pos, aa in mutations.items():
            effects = self.single_site_effects.get(pos, {})
            additive += effects.get(aa, 0.0)

        # Epistatic component: pairwise couplings between all position pairs
        # where at least one position is mutated
        epistatic = 0.0
        all_positions = set(self.wild_type.keys())
        mutated_positions = set(mutations.keys())

        for pos_i, pos_j in combinations(sorted(all_positions), 2):
            # Skip if neither position is mutated (no epistatic contribution)
            if pos_i not in mutated_positions and pos_j not in mutated_positions:
                continue

            # Get the amino acid at each position (mutant or wild type)
            aa_i = mutations.get(pos_i, self.wild_type[pos_i])
            aa_j = mutations.get(pos_j, self.wild_type[pos_j])

            # Skip if either is wild type at that position (coupling is relative
            # to wt-wt baseline which is 0 by definition)
            wt_i = self.wild_type[pos_i]
            wt_j = self.wild_type[pos_j]
            if aa_i == wt_i and aa_j == wt_j:
                continue

            coupling = self.couplings.get_coupling(pos_i, pos_j, aa_i, aa_j)

            # Subtract the wild-type baseline couplings to keep effects relative
            # J(mut_i, mut_j) - J(wt_i, mut_j) - J(mut_i, wt_j) + J(wt_i, wt_j)
            coupling -= self.couplings.get_coupling(pos_i, pos_j, wt_i, aa_j)
            coupling -= self.couplings.get_coupling(pos_i, pos_j, aa_i, wt_j)
            coupling += self.couplings.get_coupling(pos_i, pos_j, wt_i, wt_j)

            epistatic += coupling

        fitness = additive + self.alpha * epistatic

        # Add noise if configured
        if self.noise_sigma > 0:
            fitness += self._rng.normal(0, self.noise_sigma)

        return float(fitness)

    def evaluate_batch(self, variants: list[dict[int, str]]) -> NDArray[np.float64]:
        """Evaluate fitness for a batch of variants.

        Parameters
        ----------
        variants : list of dict
            Each dict maps position -> mutant amino acid.

        Returns
        -------
        NDArray of shape (n_variants,)
            Fitness values.
        """
        return np.array([self.evaluate(v) for v in variants])

    def additive_fitness(self, mutations: dict[int, str]) -> float:
        """Compute the purely additive (no epistasis) fitness prediction.

        Useful for comparing against the full landscape to quantify
        epistatic contributions.
        """
        return sum(
            self.single_site_effects.get(pos, {}).get(aa, 0.0)
            for pos, aa in mutations.items()
        )

    def _sample(
        self,
        n: int,
        n_mutations: int,
        exclude_wild_type_aa: bool = True,
    ) -> tuple[list[dict[int, str]], NDArray[np.float64]]:
        """Sample random multi-mutant variants and compute their fitness.

        Parameters
        ----------
        n : int
            Number of variants to sample.
        n_mutations : int
            Number of mutations per variant.
        exclude_wild_type_aa : bool
            If True, sampled mutations cannot be the wild-type amino acid
            at that position (i.e. all positions are truly mutated).

        Returns
        -------
        variants : list of dict
            Each dict maps position -> mutant amino acid.
        fitness : NDArray of shape (n,)
            Fitness values for each variant.
        """
        positions = self.positions
        if n_mutations > len(positions):
            raise ValueError(
                f"Cannot sample {n_mutations} mutations from {len(positions)} positions"
            )

        variants = []
        for _ in range(n):
            # Pick random positions to mutate
            mut_positions = self._rng.choice(positions, size=n_mutations, replace=False)
            variant: dict[int, str] = {}

            for pos in mut_positions:
                wt_aa = self.wild_type[pos]
                # Choose from amino acids with measured effects, excluding wt if requested
                available = list(self.single_site_effects.get(pos, {}).keys())
                if exclude_wild_type_aa:
                    available = [aa for aa in available if aa != wt_aa]
                if not available:
                    # Fallback: pick any non-wt amino acid
                    available = [aa for aa in AMINO_ACIDS if aa != wt_aa]

                variant[int(pos)] = self._rng.choice(available)

            variants.append(variant)

        fitness = self.evaluate_batch(variants)
        return variants, fitness

    def stats(
        self, n_samples: int = 5000, n_mutations: int = 2
    ) -> dict[str, float]:
        """Compute summary statistics characterising the landscape.

        Samples random double (or n-fold) mutants and compares their
        actual fitness to the additive prediction to quantify epistasis.

        Parameters
        ----------
        n_samples : int
            Number of random variants to sample.
        n_mutations : int
            Number of mutations per variant.

        Returns
        -------
        dict with keys:
            - 'mean_fitness': average fitness across samples
            - 'std_fitness': standard deviation of fitness
            - 'additive_correlation': Pearson r between actual and additive fitness
            - 'mean_epistatic_deviation': mean |actual - additive|
            - 'fraction_sign_epistasis': fraction of variants where at least one
              mutation has its sign flipped relative to single-site prediction
        """
        variants, fitness = self._sample(n_samples, n_mutations)
        additive = np.array([self.additive_fitness(v) for v in variants])

        # Pearson correlation between actual and additive
        if np.std(fitness) > 0 and np.std(additive) > 0:
            corr = float(np.corrcoef(fitness, additive)[0, 1])
        else:
            corr = 1.0  # degenerate case

        # Sign epistasis: cases where a mutation beneficial alone becomes
        # deleterious in the multi-mutant context (or vice versa)
        n_sign_epistasis = 0
        for variant, fit_val in zip(variants, fitness):
            for pos, aa in variant.items():
                single_effect = self.single_site_effects.get(pos, {}).get(aa, 0.0)
                # This mutation's "contextual" effect in this background
                reduced_variant = {p: a for p, a in variant.items() if p != pos}
                bg_fitness = self.evaluate(reduced_variant) if reduced_variant else 0.0
                contextual_effect = fit_val - bg_fitness

                if single_effect != 0 and np.sign(contextual_effect) != np.sign(single_effect):
                    n_sign_epistasis += 1
                    break  # count each variant at most once

        return {
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "additive_correlation": corr,
            "mean_epistatic_deviation": float(np.mean(np.abs(fitness - additive))),
            "fraction_sign_epistasis": n_sign_epistasis / n_samples,
        }

    def generate(
        self,
        n_mutations: int | tuple[int, int],
        n: int | None = None,
        seed: int | None = None,
    ) -> Iterator[Variant]:
        """Yield random variants from the combinatorial landscape.

        Parameters
        ----------
        n_mutations : int or (min, max)
            Fixed mutation count, or inclusive range to sample uniformly from.
        n : int or None
            Number of variants to yield. None for an infinite generator.
        seed : int or None
            Random seed for this generator. Uses the landscape's RNG if None.

        Yields
        ------
        Variant
            Named tuple with mutations, sequence, fitness, and label.
        """
        if self.wt_sequence is None:
            raise ValueError(
                "wt_sequence is not set. Pass wt_fasta to Landscape.from_dms() "
                "or set wt_sequence directly."
            )

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        positions = self.positions

        if isinstance(n_mutations, int):
            min_muts, max_muts = n_mutations, n_mutations
        else:
            min_muts, max_muts = n_mutations

        count = 0
        while n is None or count < n:
            depth = rng.integers(min_muts, max_muts + 1)
            mut_positions = rng.choice(positions, size=depth, replace=False)
            raw_mutations: dict[int, str] = {}

            for pos in mut_positions:
                wt_aa = self.wild_type[pos]
                available = [
                    aa
                    for aa in self.single_site_effects.get(pos, {})
                    if aa != wt_aa
                ]
                if not available:
                    available = [aa for aa in AMINO_ACIDS if aa != wt_aa]
                raw_mutations[int(pos)] = rng.choice(available)

            mutations = Mutant.from_dict(raw_mutations, self.wild_type)
            fitness = self.evaluate(mutations)
            sequence = self.mutant_sequence(mutations)

            yield Variant(
                mutations=mutations,
                sequence=sequence,
                fitness=fitness,
            )
            count += 1

    def to_dataframe(
        self,
        n_variants: int,
        n_mutations: int | tuple[int, int],
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Generate a DataFrame of random combinatorial variants.

        Parameters
        ----------
        n_variants : int
            Target number of unique variants (may oversample to hit target
            after deduplication).
        n_mutations : int or (min, max)
            Fixed mutation count, or inclusive range.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: sequence, variant, fitness, n_mutations.
        """
        seen: set[frozenset[tuple[int, str]]] = set()
        rows: list[dict] = []
        stale = 0

        for variant in self.generate(n_mutations, n=None, seed=seed):
            key = frozenset(variant.mutations.items())
            if key in seen:
                stale += 1
                # Stop if we've seen too many duplicates (likely exhausted the space)
                if stale > n_variants:
                    break
                continue
            stale = 0
            seen.add(key)
            rows.append({
                "sequence": variant.sequence,
                "variant": variant.mutations.label,
                "fitness": variant.fitness,
                "n_mutations": variant.mutations.depth,
            })
            if len(rows) >= n_variants:
                break

        return pd.DataFrame(rows)

    def plot_epistasis(
        self,
        depths: list[int] | None = None,
        n_samples: int = 500,
        seed: int | None = None,
    ):
        """Scatter plot of additive vs actual fitness at each mutation depth."""
        from mutascape.plotting import plot_epistasis_scatter

        return plot_epistasis_scatter(
            self, depths=depths or [1, 2, 4], n_samples=n_samples, seed=seed
        )

    def plot_fitness(
        self,
        depths: list[int] | None = None,
        n_samples: int = 500,
    ):
        """Fitness distributions at different mutation depths."""
        from mutascape.plotting import plot_fitness_distribution

        return plot_fitness_distribution(self, depths=depths or [1, 2, 3, 4], n_samples=n_samples)

    def plot_alpha_sweep(
        self,
        alphas: list[float] | None = None,
        n_samples: int = 2000,
        n_mutations: int = 3,
    ):
        """Show how landscape properties change with epistatic strength.

        Returns (figure, sweep_data) where sweep_data is {alpha: stats_dict}.
        """
        from mutascape.plotting import plot_alpha_sweep

        return plot_alpha_sweep(self, alphas=alphas, n_samples=n_samples, n_mutations=n_mutations)

    def plot_contacts(self):
        """Heatmap of the structural contact map."""
        from mutascape.plotting import plot_contact_map

        return plot_contact_map(self)

    @classmethod
    def from_plm(
        cls,
        sequence: str,
        contact_map: ContactMap,
        epistasis: EpistasisModel,
        scorer: PLMScorer,
        positions: list[int] | None = None,
        alpha: float = 1.0,
        noise_sigma: float = 0.0,
        seed: int | None = None,
    ) -> Landscape:
        """Build a landscape from pLM-predicted single-site effects.

        Uses masked marginal log-likelihood ratios as fitness proxies
        for every position (or a subset) in the protein.

        Parameters
        ----------
        sequence : str
            Wild-type protein sequence.
        contact_map : ContactMap
            Structural contact map.
        epistasis : EpistasisModel
            Coupling model.
        scorer : PLMScorer
            Protein language model scorer.
        positions : list[int] or None
            1-indexed positions to include. None uses all positions.
        alpha : float
            Epistatic strength multiplier.
        noise_sigma : float
            Measurement noise std dev.
        seed : int or None
            Random seed.
        """
        if positions is None:
            positions = list(range(1, len(sequence) + 1))

        single_site_effects = scorer.score(sequence, positions)
        wild_type = {pos: sequence[pos - 1] for pos in positions}

        # Restrict contact map to scored positions
        available = [p for p in positions if p in set(contact_map.positions)]
        if not available:
            raise ValueError("No overlap between scored positions and contact map")
        restricted_contacts = contact_map.submap(available)

        couplings = epistasis.generate(restricted_contacts, single_site_effects)

        return cls(
            wild_type=wild_type,
            single_site_effects=single_site_effects,
            contact_map=restricted_contacts,
            couplings=couplings,
            alpha=alpha,
            noise_sigma=noise_sigma,
            seed=seed,
            wt_sequence=sequence,
        )

    @classmethod
    def from_dms(
        cls,
        dms_path: str | Path,
        contact_map: ContactMap,
        epistasis: EpistasisModel,
        alpha: float = 1.0,
        noise_sigma: float = 0.0,
        seed: int | None = None,
        wt_fasta: str | Path | None = None,
        wt_sequence: str | None = None,
        scorer: PLMScorer | None = None,
        position_col: str = "position",
        wildtype_col: str = "wildtype",
        mutant_col: str = "mutant",
        fitness_col: str = "fitness",
    ) -> Landscape:
        """Construct a landscape from a DMS CSV file and structural data.

        Expected CSV format: one row per single-site variant, with columns
        for position, wild-type AA, mutant AA, and fitness effect.

        Parameters
        ----------
        dms_path : str or Path
            Path to DMS CSV file.
        contact_map : ContactMap
            Structural contact map.
        epistasis : EpistasisModel
            Coupling model to use (RandomEpistasis, SignConsistentEpistasis, etc).
        alpha : float
            Epistatic strength multiplier.
        noise_sigma : float
            Measurement noise std dev.
        seed : int or None
            Random seed.
        wt_fasta : str, Path, or None
            Path to a FASTA file containing the wild-type sequence. When
            provided, enables sequence-level operations (mutant_sequence,
            generate, to_dataframe). Mutually exclusive with wt_sequence.
        wt_sequence : str or None
            Wild-type sequence string. Alternative to wt_fasta when you
            already have the sequence in memory.
        scorer : PLMScorer or None
            When provided, predicts single-site effects for every position
            in the WT sequence not covered by DMS data. Requires wt_fasta
            or wt_sequence to be set. DMS data takes priority where both
            exist.
        position_col, wildtype_col, mutant_col, fitness_col : str
            Column names in the CSV.

        Returns
        -------
        Landscape
        """
        df = pd.read_csv(dms_path)

        # Build wild-type dict and single-site effects from DMS data
        wild_type: dict[int, str] = {}
        single_site_effects: dict[int, dict[str, float]] = {}

        for _, row in df.iterrows():
            pos = int(row[position_col])
            wt_aa = str(row[wildtype_col])
            mut_aa = str(row[mutant_col])
            effect = float(row[fitness_col])

            wild_type[pos] = wt_aa

            if pos not in single_site_effects:
                single_site_effects[pos] = {wt_aa: 0.0}
            single_site_effects[pos][mut_aa] = effect

        # Resolve WT sequence
        if wt_fasta is not None and wt_sequence is not None:
            raise ValueError("Pass wt_fasta or wt_sequence, not both")
        if wt_fasta is not None:
            from Bio import SeqIO

            record = next(SeqIO.parse(str(wt_fasta), "fasta"))
            wt_sequence = str(record.seq)

        # Fill in unscreened positions with pLM-predicted effects
        if scorer is not None:
            if wt_sequence is None:
                raise ValueError(
                    "scorer requires wt_sequence or wt_fasta to identify unscreened positions"
                )
            dms_positions = set(wild_type.keys())
            all_positions = list(range(1, len(wt_sequence) + 1))
            unscreened = [p for p in all_positions if p not in dms_positions]

            if unscreened:
                plm_effects = scorer.score(wt_sequence, positions=unscreened)
                for pos, effects in plm_effects.items():
                    wild_type[pos] = wt_sequence[pos - 1]
                    single_site_effects[pos] = effects

        # Restrict contact map to positions with effects
        all_effect_positions = sorted(wild_type.keys())
        available_positions = [p for p in all_effect_positions if p in set(contact_map.positions)]
        if len(available_positions) < len(all_effect_positions):
            n_missing = len(all_effect_positions) - len(available_positions)
            import warnings

            warnings.warn(
                f"{n_missing} positions not found in contact map and will have "
                f"no epistatic interactions."
            )

        if available_positions:
            restricted_contacts = contact_map.submap(available_positions)
        else:
            raise ValueError("No overlap between positions and contact map")

        # Generate couplings
        couplings = epistasis.generate(restricted_contacts, single_site_effects)

        return cls(
            wild_type=wild_type,
            single_site_effects=single_site_effects,
            contact_map=restricted_contacts,
            couplings=couplings,
            alpha=alpha,
            noise_sigma=noise_sigma,
            seed=seed,
            wt_sequence=wt_sequence,
        )
