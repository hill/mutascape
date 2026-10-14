"""Regression checks for search behaviours found during the benchmark audit."""

from types import SimpleNamespace

import numpy as np
import pytest

from mutascape.benchmarks import methods
from mutascape.benchmarks.library import SearchSpace
from scripts.analyse_usefulness import crossed_medians
from scripts.audit_controlled_initialisation import fresh_initial_rows
from scripts.singles_warm_start import DMSOracle


def test_forest_keeps_equal_fitness_targets_equal(monkeypatch):
    targets = []

    class CapturedForest:
        def __init__(self, **kwargs):
            pass

        def fit(self, design, values):
            targets.append(values.copy())
            return self

        def predict(self, design):
            return np.zeros(len(design))

    monkeypatch.setattr(methods, "RandomForestRegressor", CapturedForest)
    features = SimpleNamespace(one_hot=lambda rows: np.eye(5)[rows])
    values = np.array([1.0, 1.0, 2.0])
    methods._forest_acquire("greedy")(
        np.array([3, 4]), np.array([0, 1, 2]), values, features, np.random.default_rng(2), 1
    )
    np.testing.assert_array_equal(targets[0], [0.25, 0.25, 1.0])


def test_forest_rank_targets_do_not_depend_on_tie_order(monkeypatch):
    targets = []

    class CapturedForest:
        def __init__(self, **kwargs):
            pass

        def fit(self, design, values):
            targets.append(values.copy())
            return self

        def predict(self, design):
            return np.zeros(len(design))

    monkeypatch.setattr(methods, "RandomForestRegressor", CapturedForest)
    features = SimpleNamespace(one_hot=lambda rows: np.eye(6)[rows])
    values = np.array([0.0, 2.0, 2.0, 1.0])
    order = np.array([2, 0, 3, 1])
    for requested in (np.arange(4), order):
        methods._forest_acquire("greedy")(
            np.array([4, 5]), requested, values[requested], features, np.random.default_rng(2), 1
        )
    np.testing.assert_array_equal(targets[1], targets[0][order])


def test_fidelity_bootstrap_keeps_shared_seed_columns_together():
    values = np.tile([0.0, 1.0, 2.0], (3, 1))
    calibration = np.array([[0, 1, 2], [2, 2, 0]])
    seeds = np.array([[0, 0, 2], [2, 2, 0]])
    np.testing.assert_array_equal(crossed_medians(values, calibration, seeds), [0.0, 2.0])


def test_controlled_initialisation_uses_only_new_combinations():
    codes = ((np.arange(128)[:, None] >> np.arange(7)) & 1).astype(np.int16)
    space = SimpleNamespace(codes=codes)
    rows = fresh_initial_rows(space, seed=23, size=32)
    assert len(np.unique(rows)) == 32
    assert (codes[rows].sum(axis=1) > 1).all()
    np.testing.assert_array_equal(rows, fresh_initial_rows(space, seed=23, size=32))


def test_model_initialisation_spends_32_new_measurements():
    codes = ((np.arange(128)[:, None] >> np.arange(7)) & 1).astype(np.int16)
    space = SearchSpace(
        codes, (2,) * 7, np.arange(128), 0, {tuple(c): i for i, c in enumerate(codes)}
    )
    known = np.flatnonzero(codes.sum(axis=1) <= 1)
    oracle = DMSOracle(np.arange(128, dtype=float), 32, known)

    def unexpected_fit(*args):
        pytest.fail("A 32-evaluation campaign must spend its whole budget on the initial design")

    methods._run_model_loop(space, oracle, methods.make_features(space), 23, unexpected_fit)
    assert oracle.spent == 32
    assert len(set(oracle.rows) - set(known)) == 32


@pytest.mark.parametrize("method", ["random", "annealing", "genetic"])
def test_search_spends_budget_with_known_rows_and_disconnected_candidates(method):
    codes = np.eye(40, dtype=np.int16)
    space = SearchSpace(
        codes, (2,) * 40, np.arange(40), 0, {tuple(c): i for i, c in enumerate(codes)}
    )
    oracle = DMSOracle(np.arange(40, dtype=float), 32, np.arange(5))
    methods.METHODS[method](space, oracle, methods.make_features(space), 23)
    assert oracle.spent == 32
    assert len(oracle.seen()) == 37
    if method == "annealing":
        assert oracle.fallback_queries == 32
    if method == "random":
        assert oracle.fallback_queries == 0
