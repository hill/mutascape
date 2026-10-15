"""Synthetic combinatorial landscapes built from wild type and measured singles."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.completion import dms_pairwise_unit_coefficient
from mutascape.domain import MutationDomain
from mutascape.imputation import completion_effects
from mutascape.mutant import VariantLike, parse_variant
from mutascape.nonlinearity import GlobalEpistasisSigmoid

GENERATOR_ID = "dms-sigmoid-v1"


@lru_cache(maxsize=200_000)
def _pair(first: tuple[int, str], second: tuple[int, str], sequence: str, seed: int) -> float:
    return dms_pairwise_unit_coefficient(first, second, sequence, seed) / 2.0


def source_digest(sequence: str, observations: Mapping[tuple, float], completion: Mapping) -> str:
    """Return a stable digest of the sequence, observations and frozen completion."""
    payload: dict[str, Any] = {
        "wild_type_sequence": sequence,
        "observations": [
            {"mutations": list(genotype), "fitness": fitness}
            for genotype, fitness in sorted(observations.items())
        ],
    }
    if completion:
        payload["completion"] = dict(completion)
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=16,
    ).hexdigest()


@dataclass(frozen=True)
class DMSLandscape:
    """Preserve measured singles and add a fixed random pair field.

    Sigmoid mode adds inverse-transformed effects under a readout bounded by
    the measured singles. Linear mode sums measured effects. Pool mode combines
    positive and negative effects by their norms. No combination data enter
    the model.
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
    margin: float = 0.05
    imputed: dict[tuple[int, str], float] = field(default_factory=dict)
    completion: dict = field(default_factory=dict)

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
        maximum_depth: int | None = None,
        completion: Mapping | None = None,
    ) -> Self:
        """Construct a landscape from WT and singles without enumerating variants.

        Strength scales unit-variance pair effects by the standard deviation of
        the measured single effects. Contacts restrict which site pairs can
        interact. A frozen ``completion`` adds imputed singles, which share the
        readout fitted to measured singles.
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
        completion = dict(completion or {})
        imputed = completion_effects(sequence, effects, completion) if completion else {}
        readout = None
        if mode == "sigmoid":
            readout = GlobalEpistasisSigmoid(
                float(observed.min() - margin * span),
                float(observed.max() + margin * span),
                1.0,
            )
            effects = {key: float(readout.inverse(value)) for key, value in effects.items()}
        scale = float(np.std(list(effects.values())))
        effects |= {
            key: float(readout.inverse(value)) if readout is not None else value
            for key, value in imputed.items()
        }
        domain = MutationDomain(sequence, tuple(effects), maximum_depth=maximum_depth)
        mask = None
        if contacts is not None:
            mask = frozenset((min(int(a), int(b)), max(int(a), int(b))) for a, b in contacts)
            if any(
                a == b or a not in domain.positions or b not in domain.positions for a, b in mask
            ):
                raise ValueError("contact pairs must join distinct supported positions")
        return cls(
            domain,
            measured,
            effects,
            scale,
            int(seed),
            strength,
            mode,
            readout,
            mask,
            float(margin),
            imputed,
            completion,
        )

    @property
    def wild_type(self) -> dict[int, str]:
        return dict(enumerate(self.domain.wild_type_sequence, 1))

    @property
    def wild_type_fitness(self) -> float:
        return self.observations[()]

    @property
    def source_digest(self) -> str:
        return source_digest(self.domain.wild_type_sequence, self.observations, self.completion)

    def pair_scale(self, first: int, second: int) -> float:
        """Return the standard deviation of pair effects between two sites."""
        if self.contacts is not None and (first, second) not in self.contacts:
            return 0.0
        return self.strength * self.scale

    def evaluate_batch(self, variants: Sequence[VariantLike]) -> NDArray[np.float64]:
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
                weight = self.pair_scale(p, q)
                if not weight:
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
                latent += weight * table[first_codes, second_codes]
        result = self.observations[()] + (
            self.readout.forward(latent) if self.readout is not None else latent
        )
        for row, mutant in enumerate(parsed):
            key = tuple(sorted(mutant.items()))
            if key in self.observations:
                result[row] = self.observations[key]
            elif len(key) == 1 and key[0] in self.imputed:
                result[row] = self.observations[()] + self.imputed[key[0]]
        return np.asarray(result)

    def evaluate(self, variant: VariantLike) -> float:
        """Return fitness for one supported variant."""
        return float(self.evaluate_batch([variant])[0])

    def score(self, variant: VariantLike) -> float:
        """Score one variant supplied as a label, mapping, or ``Mutant``."""
        return self.evaluate(variant)

    def score_many(self, variants: Sequence[VariantLike]) -> pd.Series:
        """Score variants and return values indexed by canonical mutation labels."""
        mutants = [self.domain.parse(variant) for variant in variants]
        return pd.Series(
            self.evaluate_batch(mutants),
            index=[mutant.label for mutant in mutants],
            name="fitness",
        )

    def is_observed(self, variant: VariantLike) -> bool:
        """Return whether this exact genotype was supplied as an observation."""
        return tuple(sorted(self.domain.parse(variant).items())) in self.observations

    def provenance(self, variant: VariantLike) -> str:
        """Return measured, imputed, or synthetic for a supported query."""
        key = tuple(sorted(self.domain.parse(variant).items()))
        if key in self.observations:
            return "measured"
        if len(key) == 1 and key[0] in self.imputed:
            return "imputed"
        return "synthetic"

    def manifest(self) -> dict[str, Any]:
        """Return the fixed settings required to recreate this oracle."""
        try:
            package_version = version("mutascape")
        except PackageNotFoundError:
            package_version = "unknown"
        return {
            "mutascape_version": package_version,
            "generator": GENERATOR_ID,
            "seed": self.seed,
            "strength": self.strength,
            "mode": self.mode,
            "margin": self.margin,
            "lower": self.readout.floor if self.readout is not None else None,
            "upper": self.readout.ceiling if self.readout is not None else None,
            "contacts": sorted(map(list, self.contacts)) if self.contacts is not None else None,
            "source_digest": self.source_digest,
            "observations": len(self.observations),
            "wild_type_fitness": self.wild_type_fitness,
            "positions": list(self.domain.positions),
            "substitutions": len(self.domain.substitutions),
            "maximum_depth": self.domain.maximum_depth,
            **({"completion": self.completion} if self.completion else {}),
        }

    def save_manifest(self, path: str | PathLike[str]) -> None:
        """Write the oracle manifest as formatted JSON."""
        Path(path).write_text(json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n")

    def __getitem__(self, variant: VariantLike) -> float:
        return self.evaluate(variant)

    def __repr__(self) -> str:
        return (
            f'Oracle(generator="{GENERATOR_ID}", seed={self.seed}, '
            f"strength={self.strength}, positions={len(self.domain.positions)}, "
            f"substitutions={len(self.domain.substitutions)}, "
            f"maximum_depth={self.domain.maximum_depth})"
        )


@dataclass(frozen=True)
class DMSGenerator:
    """Fixed DMS inputs from which each seed draws one benchmark oracle."""

    template: DMSLandscape

    @property
    def domain(self) -> MutationDomain:
        return self.template.domain

    @property
    def wild_type_sequence(self) -> str:
        return self.template.domain.wild_type_sequence

    @property
    def observations(self) -> dict[tuple, float]:
        return self.template.observations

    @property
    def source_digest(self) -> str:
        return self.template.source_digest

    def draw(self, seed: int = 0, *, strength: float | None = None) -> DMSLandscape:
        """Return one fixed oracle. ``strength`` overrides the pair-field scale."""
        if strength is not None and (not np.isfinite(strength) or strength < 0):
            raise ValueError("strength must be finite and non-negative")
        return replace(
            self.template,
            seed=int(seed),
            strength=self.template.strength if strength is None else float(strength),
        )

    def __repr__(self) -> str:
        return (
            f'LandscapeGenerator(generator="{GENERATOR_ID}", '
            f"strength={self.template.strength}, "
            f"positions={len(self.domain.positions)}, "
            f"substitutions={len(self.domain.substitutions)})"
        )


SimpleLandscape = DMSLandscape
