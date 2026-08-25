"""Declared mutation domains for lazy benchmark oracles."""

from dataclasses import dataclass

import numpy as np

from mutascape.mutant import Mutant, VariantLike, parse_variant


@dataclass(frozen=True)
class MutationDomain:
    """Supported substitutions and depth limit for one target protein."""

    wild_type_sequence: str
    substitutions: tuple[tuple[int, str], ...]
    maximum_depth: int | None = None

    def __post_init__(self) -> None:
        canonical = tuple(
            sorted(
                {
                    (int(position), str(amino_acid).upper())
                    for position, amino_acid in self.substitutions
                }
            )
        )
        object.__setattr__(self, "wild_type_sequence", self.wild_type_sequence.upper())
        object.__setattr__(self, "substitutions", canonical)
        if self.maximum_depth is not None and self.maximum_depth < 0:
            raise ValueError("maximum depth must be non-negative")

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(sorted({position for position, _ in self.substitutions}))

    def substitutions_at(self, position: int) -> tuple[str, ...]:
        """Return supported non-wild-type residues at one position."""
        position = int(position)
        return tuple(amino_acid for site, amino_acid in self.substitutions if site == position)

    def parse(self, variant: VariantLike) -> Mutant:
        """Return one canonical, domain-checked mutant."""
        return parse_variant(
            variant,
            wild_type_sequence=self.wild_type_sequence,
            supported_substitutions=self.substitutions,
            maximum_depth=self.maximum_depth,
        )

    def contains(self, variant: VariantLike) -> bool:
        try:
            self.parse(variant)
        except ValueError:
            return False
        return True

    def count(self, *, depth: int | None = None) -> int:
        """Count variants at one depth or across the declared depth range."""
        sizes = [len(self.substitutions_at(position)) for position in self.positions]
        counts = [0] * (len(sizes) + 1)
        counts[0] = 1
        for size in sizes:
            for order in range(len(sizes), 0, -1):
                counts[order] += counts[order - 1] * size
        maximum = self.maximum_depth if self.maximum_depth is not None else len(sizes)
        maximum = min(maximum, len(sizes))
        if depth is None:
            return sum(counts[: maximum + 1])
        if depth < 0 or depth > maximum:
            return 0
        return counts[depth]

    def sample(self, *, depth: int, n: int, seed: int = 0) -> list[Mutant]:
        """Sample distinct supported variants at a fixed mutation depth."""
        if n < 0:
            raise ValueError("sample size must be non-negative")
        capacity = self.count(depth=depth)
        if capacity == 0 and n:
            raise ValueError(f"mutation depth {depth} is outside this domain")
        if n > capacity:
            raise ValueError(f"requested {n} variants but depth {depth} contains {capacity}")
        if n == 0:
            return []

        rng = np.random.default_rng(seed)
        positions = np.asarray(self.positions, dtype=int)
        wild_type = {position: self.wild_type_sequence[position - 1] for position in self.positions}
        sampled: list[Mutant] = []
        seen: set[tuple[tuple[int, str], ...]] = set()
        while len(sampled) < n:
            chosen = rng.choice(positions, size=depth, replace=False)
            mutations = {
                int(position): str(rng.choice(self.substitutions_at(int(position))))
                for position in chosen
            }
            key = tuple(sorted(mutations.items()))
            if key in seen:
                continue
            seen.add(key)
            sampled.append(Mutant(mutations, wild_type=wild_type))
        return sampled

    def __repr__(self) -> str:
        maximum = self.maximum_depth if self.maximum_depth is not None else len(self.positions)
        return (
            "MutationDomain("
            f"positions={len(self.positions)}, "
            f"substitutions={len(self.substitutions)}, "
            f"maximum_depth={maximum})"
        )
