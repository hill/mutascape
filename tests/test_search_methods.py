"""Budget, domain and roughness invariants for the search-transfer benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from mutascape.benchmarks.library import BudgetedOracle, SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features, run_method
from mutascape.benchmarks.topology import CubeSpec


@pytest.fixture
def cube() -> CubeSpec:
    positions = (1, 2, 3)
    alphabets = {1: ["A", "C", "D"], 2: ["A", "E"], 3: ["A", "F", "G"]}
    return CubeSpec(positions, alphabets, {1: "A", 2: "A", 3: "A"})


@pytest.fixture
def space(cube: CubeSpec) -> SearchSpace:
    return SearchSpace.from_cube(cube, np.ones((3, 2, 3), dtype=bool))


def test_space_covers_the_whole_cube(space: SearchSpace) -> None:
    assert space.size == 18
    assert space.n_sites == 3
    assert space.alphabet_sizes == (3, 2, 3)
    assert space.wild_type_row == 0


def test_arm_values_follow_tensor_order(cube: CubeSpec, space: SearchSpace) -> None:
    tensor = np.arange(18, dtype=float).reshape(3, 2, 3)
    values = space.arm_values(tensor)
    for row, code in enumerate(space.codes):
        assert values[row] == tensor[tuple(int(value) for value in code)]


def test_mask_removes_rows_and_keeps_neighbours_consistent(cube: CubeSpec) -> None:
    mask = np.ones((3, 2, 3), dtype=bool)
    mask[2, 1, 2] = False
    space = SearchSpace.from_cube(cube, mask)
    assert space.size == 17
    for row in range(space.size):
        for neighbour in space.neighbours(row):
            difference = space.codes[row] != space.codes[neighbour]
            assert int(difference.sum()) == 1


def test_oracle_stops_at_the_budget(space: SearchSpace) -> None:
    values = np.arange(space.size, dtype=float)
    oracle = BudgetedOracle(values, budget=5)
    oracle.query(range(space.size))
    assert oracle.spent == 5
    assert oracle.best() == 4.0


def test_repeated_rows_cost_nothing(space: SearchSpace) -> None:
    values = np.arange(space.size, dtype=float)
    oracle = BudgetedOracle(values, budget=3)
    oracle.query([0, 0, 0, 0])
    assert oracle.spent == 1
    assert oracle.remaining == 2


def test_unaffordable_rows_never_win_an_argmax(space: SearchSpace) -> None:
    values = np.arange(space.size, dtype=float)
    oracle = BudgetedOracle(values, budget=1)
    returned = oracle.query([0, 17])
    assert returned[0] == 0.0
    assert returned[1] == -np.inf


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_method_respects_the_budget(space: SearchSpace, name: str) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=space.size)
    features = make_features(space)
    oracle = BudgetedOracle(values, budget=8)
    METHODS[name](space, oracle, features, 3)
    assert oracle.spent <= 8
    assert oracle.best() in set(values.tolist())


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_method_finds_the_peak_with_a_full_budget(space: SearchSpace, name: str) -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=space.size)
    features = make_features(space)
    best = run_method(name, space, values, features, budget=space.size, seed=5)
    assert best == pytest.approx(float(values.max()))


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_method_is_invariant_to_positive_affine_fitness(name: str) -> None:
    cube = CubeSpec(
        positions=(1, 2, 3, 4),
        alphabets={position: ["A", "C", "D"] for position in range(1, 5)},
        wild_type={position: "A" for position in range(1, 5)},
    )
    space = SearchSpace.from_cube(cube, np.ones((3, 3, 3, 3), dtype=bool))
    rng = np.random.default_rng(11)
    values = rng.normal(size=space.size)
    transformed = 17.0 + 8.5 * values
    first = BudgetedOracle(values, budget=48)
    second = BudgetedOracle(transformed, budget=48)

    METHODS[name](space, first, make_features(space), 13)
    METHODS[name](space, second, make_features(space), 13)

    np.testing.assert_array_equal(first.rows, second.rows)


@pytest.mark.filterwarnings("error::sklearn.exceptions.ConvergenceWarning")
def test_mlp_stops_without_a_convergence_warning() -> None:
    cube = CubeSpec(
        positions=(1, 2, 3, 4, 5),
        alphabets={position: ["A", "C", "D"] for position in range(1, 6)},
        wild_type={position: "A" for position in range(1, 6)},
    )
    space = SearchSpace.from_cube(cube, np.ones((3, 3, 3, 3, 3), dtype=bool))
    values = np.random.default_rng(17).normal(size=space.size)
    oracle = BudgetedOracle(values, budget=192)

    METHODS["mlp_ucb"](space, oracle, make_features(space), 19)

    assert oracle.spent == 192
