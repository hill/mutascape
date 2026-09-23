"""Small singles-only landscape families for research comparisons."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.completion import dms_pairwise_unit_coefficient
from mutascape.domain import MutationDomain
from mutascape.mutant import VariantLike, parse_variant
from mutascape.nonlinearity import GlobalEpistasisSigmoid


@lru_cache(maxsize=200_000)
def _pair(first: tuple[int, str], second: tuple[int, str], sequence: str, seed: int) -> float:
    return dms_pairwise_unit_coefficient(first, second, sequence, seed) / 2.0


@dataclass(frozen=True)
class SimpleLandscape:
    """Preserve measured singles and add a fixed random pair field.

    Linear mode sums measured effects. Sigmoid mode adds inverse-transformed
    effects. Pool mode combines positive and negative effects by their norms.
    These are synthetic assumptions, with no target combination fitting.
    """

    domain: MutationDomain
    observations: dict[tuple, float]
    effects: dict[tuple[int, str], float]
    scale: float
    seed: int
    strength: float
    mode: str
    readout: GlobalEpistasisSigmoid | None
    contacts: frozenset[tuple[int, int]] | None

    @classmethod
    def from_singles(
        cls,
        sequence: str,
        singles: pd.DataFrame,
        *,
        seed: int = 0,
        strength: float = 0.3,
        mode: Literal["linear", "sigmoid", "pool"] = "sigmoid",
        margin: float = 0.05,
        contacts: set[tuple[int, int]] | None = None,
    ) -> Self:
        """Construct a landscape from WT and singles without enumerating variants.

        Strength scales unit-variance pair effects by the single-effect standard
        deviation. Contacts restrict which site pairs can interact.
        """
        if mode not in {"linear", "sigmoid", "pool"}:
            raise ValueError("mode must be linear, sigmoid or pool")
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and non-negative")
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("margin must be finite and positive")
        measured = {}
        for mutation, fitness in singles[["mutations", "fitness"]].itertuples(index=False):
            key = tuple(sorted(parse_variant(mutation, wild_type_sequence=sequence).items()))
            if len(key) > 1:
                raise ValueError("this generator accepts only wild type and single mutants")
            if key in measured:
                raise ValueError("duplicate input genotype")
            if not np.isfinite(fitness):
                raise ValueError("fitness must be finite")
            measured[key] = float(fitness)
        if () not in measured or len(measured) < 3:
            raise ValueError("supply wild type and at least two single mutants")
        wt = measured[()]
        effects = {key[0]: value - wt for key, value in measured.items() if key}
        values = np.array(list(effects.values()))
        observed = np.r_[0.0, values]
        span = float(np.ptp(observed))
        if span <= 1e-12:
            raise ValueError("single-mutant fitness must vary")
        readout = None
        if mode == "sigmoid":
            readout = GlobalEpistasisSigmoid(
                float(observed.min() - margin * span),
                float(observed.max() + margin * span),
                1.0,
            )
            effects = {key: float(readout.inverse(value)) for key, value in effects.items()}
        scale = float(np.std(list(effects.values())))
        domain = MutationDomain(sequence, tuple(effects))
        mask = None
        if contacts is not None:
            mask = frozenset(tuple(sorted((int(a), int(b)))) for a, b in contacts)
            if any(
                a == b or a not in domain.positions or b not in domain.positions for a, b in mask
            ):
                raise ValueError("contact pairs must join distinct supported positions")
        return cls(domain, measured, effects, scale, int(seed), strength, mode, readout, mask)

    def evaluate_batch(self, variants: list[VariantLike]) -> NDArray[np.float64]:
        """Score supported variants with the same pair effects in every query."""
        parsed = [self.domain.parse(v) for v in variants]
        if not parsed:
            return np.empty(0)
        positions = sorted({p for m in parsed for p in m})
        values = np.zeros((len(parsed), len(positions)))
        residues = np.full(values.shape, "", dtype="U1")
        index = {p: i for i, p in enumerate(positions)}
        for row, mutant in enumerate(parsed):
            for p, aa in mutant.items():
                values[row, index[p]] = self.effects[p, aa]
                residues[row, index[p]] = aa
        if self.mode == "pool":
            positive = np.sqrt(np.sum(np.maximum(values, 0) ** 2, axis=1))
            negative = np.sqrt(np.sum(np.minimum(values, 0) ** 2, axis=1))
            latent = positive - negative
        else:
            latent = values.sum(axis=1)
        if self.strength:
            encoded = [
                np.unique(residues[:, i], return_inverse=True) for i in range(len(positions))
            ]
            for i, j in combinations(range(len(positions)), 2):
                p, q = positions[i], positions[j]
                if self.contacts is not None and (p, q) not in self.contacts:
                    continue
                first, first_codes = encoded[i]
                second, second_codes = encoded[j]
                table = np.zeros((len(first), len(second)))
                for a, aa in enumerate(first):
                    if not aa:
                        continue
                    for b, bb in enumerate(second):
                        if not bb:
                            continue
                        table[a, b] = _pair(
                            (p, str(aa)), (q, str(bb)), self.domain.wild_type_sequence, self.seed
                        )
                latent += self.strength * self.scale * table[first_codes, second_codes]
        result = self.observations[()] + (
            self.readout.forward(latent) if self.readout is not None else latent
        )
        for row, mutant in enumerate(parsed):
            key = tuple(sorted(mutant.items()))
            if key in self.observations:
                result[row] = self.observations[key]
        return np.asarray(result)

    def __getitem__(self, variant: VariantLike) -> float:
        """Return fitness for one supported multi-mutant or measured variant."""
        return float(self.evaluate_batch([variant])[0])
