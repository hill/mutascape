"""Core fitness landscape class.

Combines single-site effects, structural contact maps, optional epistatic
couplings, a global-epistasis function, and optional noise to create synthetic
combinatorial fitness landscapes. The landscape can evaluate arbitrary
multi-mutant variants and sample random variants at specified mutation depths.

The general fitness function is:

    f(x) = g(sum_i g^-1(h_i(x_i))
             + alpha * sum_{i<j} C_ij * J_ij(x_i, x_j)
             + alpha_ho * H(x)
             + noise)

Where h_i are single-site effects from DMS or a pLM scorer, J_ij are optional
pairwise couplings from the chosen epistasis model, C_ij is the contact map,
H is an optional structural NK-style higher-order term, and g is usually either
the identity or a fitted global-epistasis function.

This approach differs from SLIP (Thomas et al. 2022), which derives both h
and J from Potts model fits to MSAs, and from NK landscapes (Sandhu et al.
2025), which use abstract random interactions. mutascape grounds single-site
effects in experimental data where available and keeps structural priors
explicit rather than implicit.
"""

import hashlib
import math
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from numpy.typing import NDArray

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    AMINO_ACIDS,
    ContactNKTensor,
    CouplingTensor,
    EpistasisModel,
)
from mutascape.mutant import Mutant
from mutascape.nonlinearity import Identity, NonLinearity
from mutascape.nonlinearity import resolve as _resolve_nonlinearity
from mutascape.scoring import PLMScorer
from mutascape.validation.metrics import local_optima_from_tensor


def _impute_missing_single_sites(
    single_site_effects: dict[int, dict[str, float]],
    strategy: str = "position_mean",
) -> dict[int, dict[str, float]]:
    """Fill in (pos, aa) entries the DMS did not measure.

    A synthetic combinatorial landscape must produce a value for every
    variant it is asked to evaluate. Defaulting unmeasured (pos, aa) effects
    to 0 silently injects neutral mutations into multi-mutant predictions,
    which biases the synthetic landscape toward additive behaviour and
    distorts the candidate-pool fitness distribution.

    Strategies:
      - "position_mean": each missing AA at position i gets the mean of the
        measured non-WT effects at position i. WT stays anchored at 0.
        Assumes AAs at a position are roughly exchangeable a priori, which
        is the right baseline given only DMS information.
      - "raise": raise ValueError listing the missing entries.

    Returns a NEW dict with every (pos, aa) for the 20-letter alphabet
    populated at each position. Input is not mutated.
    """
    if strategy not in {"position_mean", "raise"}:
        raise ValueError(f"unknown missing_aa_prior strategy: {strategy!r}")

    out: dict[int, dict[str, float]] = {}
    n_imputed = 0
    for pos, effects in single_site_effects.items():
        complete = dict(effects)  # copy
        measured_aas = set(effects.keys())
        missing = [aa for aa in AMINO_ACIDS if aa not in measured_aas]
        if not missing:
            out[pos] = complete
            continue
        if strategy == "raise":
            raise ValueError(
                f"position {pos} missing single-site effects for AAs: "
                f"{','.join(missing)} (and missing_aa_prior='raise')"
            )
        # position_mean strategy
        non_wt = [(aa, eff) for aa, eff in effects.items() if eff != 0.0]
        if non_wt:
            prior = float(np.mean([eff for _, eff in non_wt]))
        else:
            # only WT measured at this position; no useful prior, fall back to 0
            prior = 0.0
        for aa in missing:
            complete[aa] = prior
            n_imputed += 1
        out[pos] = complete

    if n_imputed:
        warnings.warn(
            f"Mutascape: imputed {n_imputed} unmeasured (pos, aa) single-site "
            f"effects using strategy={strategy!r}. The synthetic landscape now "
            f"produces a value for every variant; no silent 0 fallback remains.",
            stacklevel=2,
        )

    return out


class Variant(NamedTuple):
    """A variant in a combinatorial fitness landscape."""

    mutations: Mutant
    sequence: str
    fitness: float


def _allocate_weighted_counts(
    total: int,
    weights: Mapping[int, float],
    capacities: Mapping[int, int],
) -> dict[int, int]:
    """Allocate an integer total by weight without exceeding depth capacities."""
    if total > sum(capacities[depth] for depth in weights):
        raise ValueError("sample plan exceeds the finite genotype space")
    counts = {depth: 0 for depth in weights}
    active = set(weights)
    remaining = total
    while remaining:
        weight_sum = sum(weights[depth] for depth in active)
        exact = {depth: remaining * weights[depth] / weight_sum for depth in active}
        allocated = {depth: math.floor(value) for depth, value in exact.items()}
        remainder = remaining - sum(allocated.values())
        order = sorted(
            active,
            key=lambda depth: (-(exact[depth] - allocated[depth]), depth),
        )
        for depth in order[:remainder]:
            allocated[depth] += 1

        capped = {depth for depth in active if allocated[depth] > capacities[depth] - counts[depth]}
        if not capped:
            for depth, value in allocated.items():
                counts[depth] += value
            break
        for depth in capped:
            available = capacities[depth] - counts[depth]
            counts[depth] += available
            remaining -= available
            active.remove(depth)
    return counts


def _pin_nk_wildtype(tensor: ContactNKTensor, wild_type: dict[int, str]) -> ContactNKTensor:
    """Shift each per-position NK function so f(all wildtype) = 0.

    Needed at Landscape construction time because the wildtype is tracked on
    the landscape rather than on the contact map. Keeps single-mutant
    predictions consistent with DMS data when the higher-order component is
    added on top of the additive h_i terms.
    """
    shifted: dict[int, NDArray[np.float64]] = {}
    for pos, f in tensor.per_position_functions.items():
        wt_idx = (
            tensor.aa_index[wild_type[pos]],
            *(tensor.aa_index[wild_type[n]] for n in tensor.neighbours_by_position[pos]),
        )
        shifted[pos] = f - float(f[wt_idx])
    return ContactNKTensor(
        per_position_functions=shifted,
        neighbours_by_position=tensor.neighbours_by_position,
        positions=tensor.positions,
        aa_index=tensor.aa_index,
    )


@dataclass
class Landscape:
    """Synthetic combinatorial protein fitness landscape.

    Attributes
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
        Multiplier for optional pairwise couplings. In the publication
        reference configs `sigma=0`, so this is retained for API compatibility
        but does not affect the headline landscape.
    noise_sigma : float
        Standard deviation of Gaussian measurement noise (0.0 = no noise).
    seed : int or None
        Random seed for noise and sampling.
    deterministic_noise : bool
        If True, the noise term is a reproducible function of genotype and
        seed. This makes the landscape a stable oracle: repeated queries for
        the same variant return the same value. If False, each query draws a
        fresh observation-noise sample from the landscape RNG.
    readout_texture_sigma : float
        Standard deviation of an optional post-readout texture term.
        This term is useful for breaking censored low-fitness plateaus into a
        stable neutral network without changing the latent biophysical score.
    """

    wild_type: dict[int, str]
    single_site_effects: dict[int, dict[str, float]]
    contact_map: ContactMap
    couplings: CouplingTensor
    alpha: float = 1.0
    noise_sigma: float = 0.0
    seed: int | None = None
    deterministic_noise: bool = True
    readout_texture_sigma: float = 0.0
    readout_texture_order: int = 0
    readout_texture_iid_fraction: float = 0.0
    readout_texture_floor: float | None = None
    readout_texture_floor_band: float = 1e-9
    readout_texture_seed: int | None = None
    wt_sequence: str | None = None
    nonlinearity: NonLinearity = field(default_factory=Identity)
    higher_order: ContactNKTensor | None = None
    alpha_ho: float = 0.0
    _rng: np.random.Generator = field(init=False, repr=False)
    _preimage_effects: dict[int, dict[str, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.wt_sequence is not None:
            self._validate_wt_sequence()
        self._cache_preimages()
        if self.higher_order is not None:
            self.higher_order = _pin_nk_wildtype(self.higher_order, self.wild_type)

    def _cache_preimages(self) -> None:
        """Precompute g^{-1}(h_i) for every single-site effect.

        Under the global-epistasis function model the additive component of
        fitness is summed in stability-equivalent units (the preimage) and then
        passed through g. Caching the preimages keeps per-evaluation cost equal
        to the identity-readout case.
        """
        self._preimage_effects = {
            pos: {aa: float(self.nonlinearity.inverse(v)) for aa, v in effects.items()}
            for pos, effects in self.single_site_effects.items()
        }

    def __repr__(self) -> str:
        positions = ", ".join(f"{pos}({self.wild_type[pos]})" for pos in self.positions)
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
        mutations : dict[int, str]
            Maps position -> mutant amino acid.

        Returns
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
        # Validate claimed wildtype matches actual wildtype
        if isinstance(mutations, Mutant):
            for pos, claimed_wt in mutations._wild_type.items():
                actual_wt = self.wild_type.get(pos)
                if actual_wt and claimed_wt != actual_wt:
                    raise ValueError(
                        f"Position {pos}: claimed wildtype '{claimed_wt}' "
                        f"but actual wildtype is '{actual_wt}'"
                    )
        return self.evaluate(mutations)

    def evaluate(self, mutations: dict[int, str]) -> float:
        """Compute fitness of a variant specified by its mutations.

        Fitness is relative to wild type: 0.0 means wild-type fitness,
        positive values are better than WT, negative values are worse.

        Parameters
        mutations : dict[int, str]
            Maps position -> mutant amino acid. Positions not included
            are assumed to be wild type.

        Returns
        float
            Fitness relative to wild type (WT = 0.0).

        Examples
        >>> landscape.evaluate({57: "G", 72: "V"})
        0.42
        """
        active_mutations = {
            int(pos): aa for pos, aa in mutations.items() if self.wild_type.get(int(pos)) != aa
        }

        # Additive component summed in preimage (stability-equivalent) space
        # so that the global-epistasis function g wraps the whole fitness function.
        additive_preimage = 0.0
        for pos, aa in active_mutations.items():
            effects = self._preimage_effects.get(pos, {})
            additive_preimage += effects.get(aa, 0.0)

        # Epistatic component: pairwise couplings between true non-WT positions.
        # Pairs with only one non-WT position cancel under the WT baseline
        # correction below, so iterating over mutant pairs is equivalent and
        # keeps full WT-context dictionaries canonical.
        epistatic = 0.0
        mutated_positions = set(active_mutations.keys())

        for pos_i, pos_j in combinations(sorted(mutated_positions), 2):
            if pos_i not in self.wild_type or pos_j not in self.wild_type:
                continue

            aa_i = active_mutations[pos_i]
            aa_j = active_mutations[pos_j]

            wt_i = self.wild_type[pos_i]
            wt_j = self.wild_type[pos_j]

            coupling = self.couplings.get_coupling(pos_i, pos_j, aa_i, aa_j)

            # Subtract the wild-type baseline couplings to keep effects relative
            # J(mut_i, mut_j) - J(wt_i, mut_j) - J(mut_i, wt_j) + J(wt_i, wt_j)
            coupling -= self.couplings.get_coupling(pos_i, pos_j, wt_i, aa_j)
            coupling -= self.couplings.get_coupling(pos_i, pos_j, aa_i, wt_j)
            coupling += self.couplings.get_coupling(pos_i, pos_j, wt_i, wt_j)

            epistatic += coupling

        # Higher-order (Contact-NK) contribution. Each position with structural
        # neighbours contributes its wildtype-pinned NK function if any involved
        # position is mutated. Summed with its own scale parameter so it composes
        # cleanly with the additive + pairwise terms.
        higher_order_contribution = 0.0
        if self.higher_order is not None and self.alpha_ho != 0.0:
            nk = self.higher_order
            for pos in nk.positions:
                neighbours = nk.neighbours_by_position.get(pos, ())
                involved = (pos, *neighbours)
                if not any(p in mutated_positions for p in involved):
                    continue
                aa_idx = nk.aa_index
                idx = (
                    aa_idx[active_mutations.get(pos, self.wild_type[pos])],
                    *(aa_idx[active_mutations.get(n, self.wild_type[n])] for n in neighbours),
                )
                higher_order_contribution += float(nk.per_position_functions[pos][idx])

        raw = additive_preimage + self.alpha * epistatic + self.alpha_ho * higher_order_contribution

        # Noise enters before the global-epistasis function. By default it
        # is pinned to genotype so complete-landscape metrics and MLDE oracles
        # see a stable node value rather than a fresh draw on every query.
        if self.noise_sigma > 0:
            raw += self._noise_for_mutations(active_mutations)

        fitness = float(self.nonlinearity.forward(raw))
        if self.readout_texture_sigma > 0 and self._readout_texture_applies(fitness):
            fitness += self.readout_texture_sigma * self._readout_texture_value(active_mutations)

        return fitness

    def _noise_for_mutations(self, mutations: dict[int, str]) -> float:
        """Return the observation-noise term for a genotype."""
        if not self.deterministic_noise:
            return float(self._rng.normal(0, self.noise_sigma))

        label = ";".join(f"{int(pos)}:{aa}" for pos, aa in sorted(mutations.items()))
        payload = f"{self.seed!r}|{label}".encode()
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        noise_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(noise_seed)
        return float(rng.normal(0, self.noise_sigma))

    def _readout_texture_applies(self, fitness: float) -> bool:
        """Whether the post-readout texture should perturb this variant."""
        floor = self.readout_texture_floor
        if floor is None:
            floor = getattr(self.nonlinearity, "floor_hard", None)
        if floor is None:
            return True
        return fitness <= float(floor) + self.readout_texture_floor_band

    def _readout_texture_value(self, mutations: dict[int, str]) -> float:
        """Stable high-order readout texture for censored neutral networks."""
        states = tuple((int(pos), aa) for pos, aa in sorted(mutations.items()))
        texture_seed = self.seed if self.readout_texture_seed is None else self.readout_texture_seed
        iid = self._hash_normal(("readout-iid", states), texture_seed)

        order = int(self.readout_texture_order)
        if order <= 0:
            return iid

        if len(states) < order:
            ordered = 0.0
        else:
            vals = [
                self._hash_normal(("readout-order", combo), texture_seed)
                for combo in combinations(states, order)
            ]
            ordered = float(np.sum(vals) / np.sqrt(len(vals)))

        mix = min(max(float(self.readout_texture_iid_fraction), 0.0), 1.0)
        if mix == 0.0:
            return ordered
        denom = np.sqrt((1.0 - mix) ** 2 + mix**2)
        if denom == 0.0:
            return 0.0
        return ((1.0 - mix) * ordered + mix * iid) / denom

    @staticmethod
    def _hash_normal(key: object, seed: int | None) -> float:
        payload = f"{seed!r}|{key!r}".encode()
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        normal_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(normal_seed)
        return float(rng.normal())

    def evaluate_batch(self, variants: list[dict[int, str]]) -> NDArray[np.float64]:
        """Evaluate fitness for a batch of variants.

        Parameters
        variants : list of dict
            Each dict maps position -> mutant amino acid.

        Returns
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
            self.single_site_effects.get(pos, {}).get(aa, 0.0) for pos, aa in mutations.items()
        )

    def _sample(
        self,
        n: int,
        n_mutations: int,
        exclude_wild_type_aa: bool = True,
    ) -> tuple[list[dict[int, str]], NDArray[np.float64]]:
        """Sample random multi-mutant variants and compute their fitness.

        Parameters
        n : int
            Number of variants to sample.
        n_mutations : int
            Number of mutations per variant.
        exclude_wild_type_aa : bool
            If True, sampled mutations cannot be the wild-type amino acid
            at that position (i.e. all positions are truly mutated).

        Returns
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

    def stats(self, n_samples: int = 5000, n_mutations: int = 2) -> dict[str, float]:
        """Compute summary statistics characterising the landscape.

        Samples random double (or n-fold) mutants and compares their
        actual fitness to the additive prediction to quantify epistasis.

        Parameters
        n_samples : int
            Number of random variants to sample.
        n_mutations : int
            Number of mutations per variant.

        Returns
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
        n_mutations : int or (min, max)
            Fixed mutation count, or inclusive range to sample uniformly from.
        n : int or None
            Number of variants to yield. None for an infinite generator.
        seed : int or None
            Random seed for this generator. Uses the landscape's RNG if None.

        Yields
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
                available = [aa for aa in self.single_site_effects.get(pos, {}) if aa != wt_aa]
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
        n_variants : int
            Target number of unique variants (may oversample to hit target
            after deduplication).
        n_mutations : int or (min, max)
            Fixed mutation count, or inclusive range.
        seed : int or None
            Random seed for reproducibility.

        Returns
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
            rows.append(
                {
                    "sequence": variant.sequence,
                    "variant": variant.mutations.label,
                    "fitness": variant.fitness,
                    "n_mutations": variant.mutations.depth,
                }
            )
            if len(rows) >= n_variants:
                break

        return pd.DataFrame(rows)

    def iter_sample_batches(
        self,
        n_variants: int,
        depth_weights: Mapping[int, float],
        seed: int,
        batch_size: int = 4096,
    ) -> Iterator[pd.DataFrame]:
        """Yield a weighted, unique sample in bounded DataFrame batches.

        ``depth_weights`` assigns a relative share to each mutation depth.
        Shares are converted to exact integer counts with largest-remainder
        allocation. Each selected genotype appears once across all batches.

        Parameters
        ----------
        n_variants : int
            Total variants to sample.
        depth_weights : Mapping[int, float]
            Relative sampling weights keyed by mutation depth.
        seed : int
            Seed for genotype sampling.
        batch_size : int, default 4096
            Maximum rows yielded in one batch.

        Yields
        ------
        pandas.DataFrame
            Columns are ``sequence``, ``variant``, ``fitness``, and
            ``n_mutations``.
        """
        if self.wt_sequence is None:
            raise ValueError(
                "wt_sequence is not set. Pass wt_fasta to Landscape.from_dms() "
                "or set wt_sequence directly."
            )
        if n_variants < 1:
            raise ValueError(f"n_variants must be positive, got {n_variants}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        weights: dict[int, float] = {}
        for raw_depth, raw_weight in depth_weights.items():
            depth = int(raw_depth)
            weight = float(raw_weight)
            if depth < 0 or depth > self.n_positions:
                raise ValueError(f"mutation depth {depth} must be in [0, {self.n_positions}]")
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"weight for depth {depth} must be finite and non-negative")
            if weight > 0:
                weights[depth] = weights.get(depth, 0.0) + weight
        if not weights:
            raise ValueError("depth_weights must contain at least one positive weight")

        spaces = {depth: self._depth_space_size(depth) for depth in weights}
        allocations = _allocate_weighted_counts(n_variants, weights, spaces)

        rng = np.random.default_rng(seed)
        pending: list[dict[int, str]] = []
        pending_depths: list[int] = []
        seen: set[tuple[tuple[int, str], ...]] = set()

        def make_batch() -> pd.DataFrame:
            fitness = self.evaluate_batch(pending)
            rows = []
            for mutations, depth, value in zip(pending, pending_depths, fitness):
                mutant = Mutant.from_dict(mutations, self.wild_type)
                rows.append(
                    {
                        "sequence": self.mutant_sequence(mutant),
                        "variant": mutant.label,
                        "fitness": float(value),
                        "n_mutations": depth,
                    }
                )
            pending.clear()
            pending_depths.clear()
            return pd.DataFrame.from_records(
                rows,
                columns=("sequence", "variant", "fitness", "n_mutations"),
            )

        for depth, target in sorted(allocations.items()):
            accepted = 0
            while accepted < target:
                mutation = self._draw_mutation_at_depth(depth, rng)
                key = tuple(sorted(mutation.items()))
                if key in seen:
                    continue
                seen.add(key)
                pending.append(mutation)
                pending_depths.append(depth)
                accepted += 1
                if len(pending) >= batch_size:
                    yield make_batch()

        if pending:
            yield make_batch()

    def _depth_space_size(self, depth: int) -> int:
        """Return the number of distinct genotypes at one mutation depth."""
        if depth == 0:
            return 1
        counts = [
            len([aa for aa in self.single_site_effects[pos] if aa != self.wild_type[pos]])
            for pos in self.positions
        ]
        total = 0
        for position_indices in combinations(range(self.n_positions), depth):
            total += math.prod(counts[index] for index in position_indices)
        return total

    def _draw_mutation_at_depth(
        self,
        depth: int,
        rng: np.random.Generator,
    ) -> dict[int, str]:
        """Draw one genotype uniformly within a fixed mutation depth."""
        if depth == 0:
            return {}
        chosen = rng.choice(self.positions, size=depth, replace=False)
        mutation: dict[int, str] = {}
        for raw_position in chosen:
            position = int(raw_position)
            alternatives = [
                aa for aa in self.single_site_effects[position] if aa != self.wild_type[position]
            ]
            mutation[position] = str(rng.choice(alternatives))
        return mutation

    def local_optima(
        self,
        positions: list[int] | None = None,
        alphabets: dict[int, list[str]] | None = None,
    ) -> dict:
        """Enumerate the complete combinatorial space and report local optima.

        A local optimum is a variant whose fitness is strictly greater than all
        its single-mutation neighbours over the product of observed alphabets.
        Reports the number of peaks, their basin sizes under greedy ascent, and
        the global peak.

        Parameters
        positions : list[int] or None
            Subset of landscape positions to enumerate over. Default is all.
        alphabets : dict[position -> list[aa]] or None
            Amino acids to consider at each position. Default is the union of
            the wild-type AA and all mutant AAs with measured effects.

        Returns dict with keys: n_optima, peak_indices, peak_fitnesses,
        global_peak, basin_sizes, positions, alphabets, tensor.
        """
        if positions is None:
            positions = self.positions
        if alphabets is None:
            alphabets = {}
            for pos in positions:
                wt = self.wild_type[pos]
                aas = set(self.single_site_effects.get(pos, {}).keys())
                aas.add(wt)
                alphabets[pos] = sorted(aas)

        shape = tuple(len(alphabets[p]) for p in positions)
        n = int(np.prod(shape))
        variants = []
        indices = []

        for combo in product(*(alphabets[p] for p in positions)):
            mutations = {pos: aa for pos, aa in zip(positions, combo) if aa != self.wild_type[pos]}
            variants.append(mutations)
            indices.append(tuple(alphabets[p].index(aa) for p, aa in zip(positions, combo)))

        fitness = self.evaluate_batch(variants)
        tensor = np.empty(shape, dtype=np.float64)
        for idx, f in zip(indices, fitness):
            tensor[idx] = f

        result = local_optima_from_tensor(tensor)
        result["positions"] = positions
        result["alphabets"] = alphabets
        result["tensor"] = tensor
        result["n_variants"] = n
        return result

    def plot_epistasis(
        self,
        depths: list[int] | None = None,
        n_samples: int = 500,
        seed: int | None = None,
    ):
        """Scatter plot of additive vs actual fitness at each mutation depth."""
        from mutascape.plotting import plot_epistasis_scatter  # cycle: plotting imports Landscape

        return plot_epistasis_scatter(
            self, depths=depths or [1, 2, 4], n_samples=n_samples, seed=seed
        )

    def plot_fitness(
        self,
        depths: list[int] | None = None,
        n_samples: int = 500,
    ):
        """Fitness distributions at different mutation depths."""
        from mutascape.plotting import (
            plot_fitness_distribution,  # cycle: plotting imports Landscape
        )

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
        from mutascape.plotting import plot_alpha_sweep  # cycle: plotting imports Landscape

        return plot_alpha_sweep(self, alphas=alphas, n_samples=n_samples, n_mutations=n_mutations)

    def plot_contacts(self):
        """Heatmap of the structural contact map."""
        from mutascape.plotting import plot_contact_map  # cycle: plotting imports Landscape

        return plot_contact_map(self)

    def plot_morphing(self, **kwargs):
        """Animated 3D landscape that morphs as alpha changes.

        See :func:`interactive.morphing_landscape` for full arguments.
        """
        from mutascape.interactive import morphing_landscape  # cycle: interactive imports Landscape

        return morphing_landscape(self, **kwargs)

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
        readout_texture_sigma: float = 0.0,
        readout_texture_order: int = 0,
        readout_texture_iid_fraction: float = 0.0,
        readout_texture_floor: float | None = None,
        readout_texture_floor_band: float = 1e-9,
        readout_texture_seed: int | None = None,
        nonlinearity: NonLinearity | str | None = None,
    ) -> Landscape:
        """Build a landscape from pLM-predicted single-site effects.

        Uses masked marginal log-likelihood ratios as fitness proxies
        for every position (or a subset) in the protein.

        Parameters
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
        nl = _resolve_nonlinearity(nonlinearity, single_site_effects)

        return cls(
            wild_type=wild_type,
            single_site_effects=single_site_effects,
            contact_map=restricted_contacts,
            couplings=couplings,
            alpha=alpha,
            noise_sigma=noise_sigma,
            seed=seed,
            readout_texture_sigma=readout_texture_sigma,
            readout_texture_order=readout_texture_order,
            readout_texture_iid_fraction=readout_texture_iid_fraction,
            readout_texture_floor=readout_texture_floor,
            readout_texture_floor_band=readout_texture_floor_band,
            readout_texture_seed=readout_texture_seed,
            wt_sequence=sequence,
            nonlinearity=nl,
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
        readout_texture_sigma: float = 0.0,
        readout_texture_order: int = 0,
        readout_texture_iid_fraction: float = 0.0,
        readout_texture_floor: float | None = None,
        readout_texture_floor_band: float = 1e-9,
        readout_texture_seed: int | None = None,
        wt_fasta: str | Path | None = None,
        wt_sequence: str | None = None,
        scorer: PLMScorer | None = None,
        position_col: str = "position",
        wildtype_col: str = "wildtype",
        mutant_col: str = "mutant",
        fitness_col: str = "fitness",
        nonlinearity: NonLinearity | str | None = None,
        missing_aa_prior: str = "position_mean",
    ) -> Landscape:
        """Construct a landscape from a DMS CSV file and structural data.

        Expected CSV format: one row per single-site variant, with columns
        for position, wild-type AA, mutant AA, and fitness effect.

        Parameters
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

        # Synthetic single-site imputation. For every position with at least
        # one measured AA, fill in every (pos, aa) pair the DMS did not measure
        # with a position-mean prior over the measured non-WT AAs. The WT
        # residue stays anchored at 0 by construction. The point: a synthetic
        # combinatorial landscape must produce a value for every variant it
        # is asked to evaluate, and silently defaulting unmeasured (pos, aa)
        # entries to 0 is wrong (it injects neutral mutations where the
        # synthetic model should predict something).
        single_site_effects = _impute_missing_single_sites(
            single_site_effects,
            strategy=missing_aa_prior,
        )

        # Restrict contact map to positions with effects
        all_effect_positions = sorted(wild_type.keys())
        available_positions = [p for p in all_effect_positions if p in set(contact_map.positions)]
        if len(available_positions) < len(all_effect_positions):
            n_missing = len(all_effect_positions) - len(available_positions)
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
        nl = _resolve_nonlinearity(nonlinearity, single_site_effects)

        return cls(
            wild_type=wild_type,
            single_site_effects=single_site_effects,
            contact_map=restricted_contacts,
            couplings=couplings,
            alpha=alpha,
            noise_sigma=noise_sigma,
            seed=seed,
            readout_texture_sigma=readout_texture_sigma,
            readout_texture_order=readout_texture_order,
            readout_texture_iid_fraction=readout_texture_iid_fraction,
            readout_texture_floor=readout_texture_floor,
            readout_texture_floor_band=readout_texture_floor_band,
            readout_texture_seed=readout_texture_seed,
            nonlinearity=nl,
            wt_sequence=wt_sequence,
        )
