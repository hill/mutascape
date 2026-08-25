"""Mutation parsing shared by Python, CSV, HTTP, and browser interfaces."""

import re
from collections.abc import Collection, Mapping


class Mutant(dict[int, str]):
    """A set of amino acid substitutions relative to wild type.

    Behaves as a dict[int, str] mapping position -> mutant amino acid,
    so it works anywhere a plain dict does. Adds parsing from label
    strings, a nice repr, and +/- operators for combining mutations.

    Examples
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
        mutations: str | Mapping[int | str, str] = "",
        wild_type: Mapping[int | str, str] | None = None,
    ) -> None:
        super().__init__()
        self._wild_type = {}

        if isinstance(mutations, Mapping):
            source_wild_type = getattr(mutations, "_wild_type", {})
            if wild_type is None and source_wild_type:
                wild_type = source_wild_type
            for raw_position, raw_amino_acid in mutations.items():
                position = _position(raw_position)
                amino_acid = _amino_acid(raw_amino_acid, position=position)
                if position in self:
                    raise ValueError(f"multiple substitutions supplied at position {position}")
                self[position] = amino_acid
            if wild_type:
                self._wild_type = {
                    _position(position): _amino_acid(amino_acid, position=_position(position))
                    for position, amino_acid in wild_type.items()
                }
                for position in list(self):
                    if self._wild_type.get(position) == self[position]:
                        del self[position]
                        self._wild_type.pop(position, None)
            return

        label = mutations.strip()
        if not label or label.lower() == "wt":
            return

        for part in re.split(r"[+:]", label):
            part = part.strip().upper()
            match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", part)
            if not match:
                raise ValueError(
                    f"Cannot parse mutation '{part}'. Expected a label such as 'A5G' or 'A5G+L10V'"
                )
            wt_aa, pos_str, mut_aa = match.groups()
            pos = _position(pos_str)
            if pos in self or pos in self._wild_type:
                raise ValueError(f"multiple substitutions supplied at position {pos}")
            if wt_aa == mut_aa:
                continue
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


type VariantLike = str | Mapping[int | str, str] | Mutant


def parse_variant(
    variant: VariantLike,
    *,
    wild_type_sequence: str | None = None,
    supported_substitutions: Collection[tuple[int, str]] | None = None,
    maximum_depth: int | None = None,
) -> Mutant:
    """Normalise and validate one variant against an optional mutation domain."""
    mutant = Mutant(variant)
    sequence = wild_type_sequence.upper() if wild_type_sequence is not None else None
    supported = (
        {
            (int(position), str(amino_acid).upper())
            for position, amino_acid in supported_substitutions
        }
        if supported_substitutions is not None
        else None
    )
    supported_positions = {position for position, _ in supported} if supported is not None else None
    active: dict[int, str] = {}
    wild_type: dict[int, str] = {}

    for position, amino_acid in mutant.items():
        if sequence is not None:
            if position > len(sequence):
                raise ValueError(
                    f"position {position} is outside the wild-type sequence of "
                    f"length {len(sequence)}"
                )
            actual_wild_type = sequence[position - 1]
            claimed_wild_type = mutant._wild_type.get(position)
            if claimed_wild_type is not None and claimed_wild_type != actual_wild_type:
                raise ValueError(
                    f"mutation at position {position} claims wild type {claimed_wild_type}, "
                    f"but the supplied sequence contains {actual_wild_type}"
                )
            if amino_acid == actual_wild_type:
                continue
            wild_type[position] = actual_wild_type
        else:
            actual_wild_type = mutant._wild_type.get(position, "?")
            if actual_wild_type != "?":
                wild_type[position] = actual_wild_type

        if supported_positions is not None and position not in supported_positions:
            raise ValueError(f"position {position} is not mutable in this oracle")
        if supported is not None and (position, amino_acid) not in supported:
            label = f"{actual_wild_type}{position}{amino_acid}"
            raise ValueError(
                f"missing single-mutant measurement for {label}. "
                "The variant has a missing constituent single"
            )
        active[position] = amino_acid

    if maximum_depth is not None:
        if maximum_depth < 0:
            raise ValueError("maximum depth must be non-negative")
        if len(active) > maximum_depth:
            raise ValueError(
                f"variant depth {len(active)} exceeds this oracle's maximum depth "
                f"of {maximum_depth}"
            )
    return Mutant(active, wild_type=wild_type)


def _position(value: int | str) -> int:
    if isinstance(value, bool):
        raise ValueError("mutation positions must be positive integers")
    try:
        position = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid mutation position {value!r}") from exc
    if position < 1:
        raise ValueError("mutation positions are one-based positive integers")
    return position


def _amino_acid(value: object, *, position: int) -> str:
    amino_acid = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z]", amino_acid):
        raise ValueError(f"invalid amino acid {value!r} at position {position}")
    return amino_acid
