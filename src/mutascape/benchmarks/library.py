"""Integer view of a mutation library, shared by every search method.

A complete factorial is a dense tensor, so a genotype is a tuple of residue
indices and its fitness on any arm is one array lookup. Every arm searches the
same rows in the same order, which is what makes a ranking difference
attributable to the landscape rather than to the domain.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mutascape.benchmarks.topology import CubeSpec


@dataclass(frozen=True)
class SearchSpace:
    """Rows of a complete cube that every arm is allowed to query."""

    codes: NDArray[np.int16]
    alphabet_sizes: tuple[int, ...]
    flat_rows: NDArray[np.int64]
    wild_type_row: int | None
    lookup: dict[tuple[int, ...], int]

    @property
    def size(self) -> int:
        return int(self.codes.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.codes.shape[1])

    @classmethod
    def from_cube(cls, cube: CubeSpec, mask: NDArray[np.bool_]) -> SearchSpace:
        """Take the rows of a cube the mask keeps, in flattened tensor order."""
        sizes = tuple(len(cube.alphabets[position]) for position in cube.positions)
        grids = np.indices(sizes).reshape(len(sizes), -1).T.astype(np.int16)
        rows = np.flatnonzero(mask.reshape(-1))
        codes = np.ascontiguousarray(grids[rows])
        lookup = {tuple(int(value) for value in code): index for index, code in enumerate(codes)}
        wild_type = tuple(
            cube.alphabets[position].index(cube.wild_type[position]) for position in cube.positions
        )
        return cls(codes, sizes, rows, lookup.get(wild_type), lookup)

    def arm_values(self, tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fitness of every searchable row under one arm's landscape."""
        flat = np.asarray(tensor, dtype=np.float64).reshape(-1)
        return flat[self.flat_rows]

    def one_hot(self) -> NDArray[np.float32]:
        """Binary site-by-residue design matrix."""
        offsets = np.concatenate([[0], np.cumsum(self.alphabet_sizes)[:-1]])
        columns = self.codes.astype(np.int64) + offsets[None, :]
        design = np.zeros((self.size, int(sum(self.alphabet_sizes))), dtype=np.float32)
        np.put_along_axis(design, columns, 1.0, axis=1)
        return design

    def neighbours(self, row: int) -> list[int]:
        """Rows one substitution away that the mask kept."""
        code = list(int(value) for value in self.codes[row])
        found: list[int] = []
        for site, size in enumerate(self.alphabet_sizes):
            original = code[site]
            for residue in range(size):
                if residue == original:
                    continue
                code[site] = residue
                index = self.lookup.get(tuple(code))
                if index is not None:
                    found.append(index)
            code[site] = original
        return found

    def start_row(self, rng: np.random.Generator) -> int:
        """Wild type where the mask kept it, otherwise a uniform row."""
        if self.wild_type_row is not None:
            return self.wild_type_row
        return int(rng.integers(self.size))


class BudgetedOracle:
    """Fitness lookup that refuses to answer beyond the measurement budget."""

    def __init__(self, values: NDArray[np.float64], budget: int) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self._values = np.asarray(values, dtype=np.float64)
        self.budget = int(budget)
        self.fallback_queries = 0
        self._rows: list[int] = []
        self._observed: list[float] = []
        self._seen: dict[int, float] = {}

    @property
    def spent(self) -> int:
        return len(self._rows)

    @property
    def remaining(self) -> int:
        return self.budget - self.spent

    @property
    def rows(self) -> NDArray[np.int64]:
        return np.asarray(self._rows, dtype=np.int64)

    @property
    def observed(self) -> NDArray[np.float64]:
        return np.asarray(self._observed, dtype=np.float64)

    def seen(self) -> set[int]:
        return set(self._seen)

    def query(self, rows) -> NDArray[np.float64]:
        """Spend budget on rows never queried before and return their fitness.

        A repeated row is free. A real campaign already holds that measurement,
        and charging for it would let a method waste budget by accident rather
        than by design.
        """
        requested = [int(row) for row in np.atleast_1d(rows)]
        for row in requested:
            if row in self._seen or self.remaining <= 0:
                continue
            value = float(self._values[row])
            self._seen[row] = value
            self._rows.append(row)
            self._observed.append(value)
        # A row the budget could not reach scores minus infinity, so a caller
        # taking an argmax can never select a variant it did not pay for.
        return np.asarray([self._seen.get(row, -np.inf) for row in requested], dtype=np.float64)

    def best(self) -> float:
        return float(max(self._observed)) if self._observed else float("nan")
