"""Named benchmark landscapes with a stable interface.

A complete cube is a dense tensor, so a whole landscape ships as a few hundred
kilobytes and answers any genotype by lookup. That is what a benchmark needs
and what a measured dataset cannot give, since no real assay reports every
combination.

    from mutascape.suite import load, available

    landscape = load("suite/GB1_posterior_0.npz")
    landscape.get_fitness(["MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"])
    landscape.optimum

Fitness is relative to wild type, where wild type is 0.0.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Benchmark:
    """One complete landscape over a defined mutation library."""

    name: str
    wild_type_sequence: str
    positions: tuple[int, ...]
    alphabets: tuple[tuple[str, ...], ...]
    values: NDArray[np.float64]
    source: str
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = tuple(len(residues) for residues in self.alphabets)
        if self.values.shape != expected:
            raise ValueError(f"values shape {self.values.shape} does not match library {expected}")

    @property
    def size(self) -> int:
        return int(self.values.size)

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    @property
    def optimum(self) -> float:
        """Best fitness in the library. A measured assay can rarely state this."""
        return float(np.nanmax(self.values))

    @property
    def optimum_sequence(self) -> str:
        index = np.unravel_index(int(np.nanargmax(self.values)), self.values.shape)
        return self._sequence_from_code(index)

    def _sequence_from_code(self, code) -> str:
        residues = list(self.wild_type_sequence)
        for site, position in enumerate(self.positions):
            residues[position - 1] = self.alphabets[site][int(code[site])]
        return "".join(residues)

    def _code_from_sequence(self, sequence: str) -> tuple[int, ...]:
        if len(sequence) != len(self.wild_type_sequence):
            raise ValueError(
                f"sequence length {len(sequence)} does not match wild type "
                f"{len(self.wild_type_sequence)}"
            )
        code = []
        for site, position in enumerate(self.positions):
            residue = sequence[position - 1]
            if residue not in self.alphabets[site]:
                raise ValueError(f"residue {residue} at position {position} is outside the library")
            code.append(self.alphabets[site].index(residue))
        return tuple(code)

    def get_fitness(self, sequences) -> NDArray[np.float64]:
        """Fitness of full-length sequences, in the order given."""
        return np.asarray(
            [self.values[self._code_from_sequence(sequence)] for sequence in sequences],
            dtype=np.float64,
        )

    def evaluate(self, mutations: dict[int, str]) -> float:
        """Fitness of a sparse one-based mutation map against wild type."""
        residues = list(self.wild_type_sequence)
        for position, amino_acid in mutations.items():
            residues[position - 1] = amino_acid
        return float(self.get_fitness(["".join(residues)])[0])

    def library(self) -> list[str]:
        """Every sequence in the cube, in flattened tensor order."""
        codes = np.indices(self.values.shape).reshape(self.n_sites, -1).T
        return [self._sequence_from_code(code) for code in codes]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            name=self.name,
            wild_type_sequence=self.wild_type_sequence,
            positions=np.asarray(self.positions, dtype=np.int64),
            alphabets=np.asarray(["".join(residues) for residues in self.alphabets]),
            values=self.values.astype(np.float32),
            source=self.source,
            metadata_keys=np.asarray(list(self.metadata)),
            metadata_values=np.asarray(list(self.metadata.values()), dtype=np.float64),
        )
        return path


def load(path: Path) -> Benchmark:
    """Read a landscape written by :meth:`Benchmark.save`."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        keys = [str(key) for key in bundle["metadata_keys"]]
        values = [float(value) for value in bundle["metadata_values"]]
        return Benchmark(
            name=str(bundle["name"]),
            wild_type_sequence=str(bundle["wild_type_sequence"]),
            positions=tuple(int(value) for value in bundle["positions"]),
            alphabets=tuple(tuple(str(group)) for group in bundle["alphabets"]),
            values=np.asarray(bundle["values"], dtype=np.float64),
            source=str(bundle["source"]),
            metadata=dict(zip(keys, values)),
        )


def available(root: Path) -> list[str]:
    """Names of every landscape bundled under a directory."""
    return sorted(path.stem for path in Path(root).glob("*.npz"))
