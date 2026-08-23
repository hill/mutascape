"""Round trip and lookup invariants for the shipped benchmark landscapes."""

from __future__ import annotations

import numpy as np
import pytest

from mutascape.suite import Benchmark, available, load


@pytest.fixture
def benchmark() -> Benchmark:
    values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    return Benchmark(
        name="toy",
        wild_type_sequence="AAAAA",
        positions=(2, 3, 5),
        alphabets=(("A", "C"), ("A", "D"), ("A", "E", "F")),
        values=values,
        source="posterior",
        metadata={"n_optima": 3.0},
    )


def test_shape_must_match_the_library() -> None:
    with pytest.raises(ValueError, match="does not match library"):
        Benchmark(
            name="toy",
            wild_type_sequence="AA",
            positions=(1, 2),
            alphabets=(("A", "C"), ("A", "D")),
            values=np.zeros((2, 3)),
            source="measured",
        )


def test_wild_type_sits_at_the_origin(benchmark: Benchmark) -> None:
    assert benchmark.get_fitness([benchmark.wild_type_sequence])[0] == 0.0
    assert benchmark.evaluate({}) == 0.0


def test_mutation_map_and_sequence_agree(benchmark: Benchmark) -> None:
    sequence = "ACAAF"
    assert benchmark.evaluate({2: "C", 5: "F"}) == benchmark.get_fitness([sequence])[0]


def test_optimum_sequence_scores_the_optimum(benchmark: Benchmark) -> None:
    assert benchmark.get_fitness([benchmark.optimum_sequence])[0] == benchmark.optimum


def test_library_enumerates_the_cube(benchmark: Benchmark) -> None:
    sequences = benchmark.library()
    assert len(sequences) == benchmark.size == 12
    assert len(set(sequences)) == 12
    assert np.array_equal(benchmark.get_fitness(sequences), benchmark.values.reshape(-1))


def test_a_residue_outside_the_library_is_refused(benchmark: Benchmark) -> None:
    with pytest.raises(ValueError, match="outside the library"):
        benchmark.get_fitness(["AWAAA"])


def test_a_wrong_length_sequence_is_refused(benchmark: Benchmark) -> None:
    with pytest.raises(ValueError, match="does not match wild type"):
        benchmark.get_fitness(["AAA"])


def test_save_and_load_round_trip(benchmark: Benchmark, tmp_path) -> None:
    path = benchmark.save(tmp_path / "toy.npz")
    restored = load(path)
    assert restored.name == benchmark.name
    assert restored.positions == benchmark.positions
    assert restored.alphabets == benchmark.alphabets
    assert restored.source == benchmark.source
    assert restored.metadata == benchmark.metadata
    assert np.allclose(restored.values, benchmark.values)
    assert available(tmp_path) == ["toy"]
