"""Lightweight mutation representation for combinatorial landscapes."""

from __future__ import annotations

import re


class Mutant(dict[int, str]):
    """A set of amino acid substitutions relative to wild type.

    Behaves as a dict[int, str] mapping position -> mutant amino acid,
    so it works anywhere a plain dict does. Adds parsing from label
    strings, a nice repr, and +/- operators for combining mutations.

    Examples
    --------
    >>> m = Mutant("A5G+L10V")
    >>> m[5]
    'G'
    >>> m.depth
    2
    >>> Mutant("A5G") + Mutant("L10V")
    Mutant('A5G+L10V')
    """

    _wild_type: dict[int, str]

    def __init__(
        self,
        mutations: str | dict[int, str] = "",
        wild_type: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        self._wild_type = {}

        if isinstance(mutations, dict):
            self.update(mutations)
            if wild_type:
                self._wild_type = dict(wild_type)
            return

        label = mutations
        if not label or label.lower() == "wt":
            return

        for part in label.split("+"):
            part = part.strip()
            match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", part)
            if not match:
                raise ValueError(
                    f"Cannot parse mutation '{part}'. "
                    f"Expected format like 'A5G' (from_aa, position, to_aa)"
                )
            wt_aa, pos_str, mut_aa = match.groups()
            pos = int(pos_str)
            self[pos] = mut_aa
            self._wild_type[pos] = wt_aa

    @classmethod
    def from_dict(
        cls, mutations: dict[int, str], wild_type: dict[int, str] | None = None
    ) -> Mutant:
        """Create a Mutant from a plain dict. Alias for Mutant(dict, wild_type)."""
        return cls(mutations, wild_type)

    @property
    def label(self) -> str:
        if not self:
            return "wt"
        parts = []
        for pos in sorted(self):
            wt_aa = self._wild_type.get(pos, "?")
            parts.append(f"{wt_aa}{pos}{self[pos]}")
        return "+".join(parts)

    @property
    def depth(self) -> int:
        return len(self)

    @property
    def positions(self) -> list[int]:
        return sorted(self.keys())

    def __add__(self, other: Mutant) -> Mutant:
        """Combine two mutants. Raises on conflicting positions."""
        conflicts = set(self.keys()) & set(other.keys())
        if conflicts:
            raise ValueError(f"Conflicting positions: {conflicts}")
        merged = {**self, **other}
        wt = {**self._wild_type, **other._wild_type}
        return Mutant.from_dict(merged, wt)

    def __sub__(self, other: Mutant) -> Mutant:
        """Remove mutations present in other."""
        remaining = {pos: aa for pos, aa in self.items() if pos not in other}
        wt = {pos: aa for pos, aa in self._wild_type.items() if pos in remaining}
        return Mutant.from_dict(remaining, wt)

    def __repr__(self) -> str:
        return f"Mutant('{self.label}')"

    def __str__(self) -> str:
        return self.label
