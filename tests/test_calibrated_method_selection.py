"""Check experimental inputs and budget accounting for calibrated searches."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from mutascape.benchmarks.methods import METHODS, make_features
from scripts import analyse_calibrated_method_selection as analysis
from scripts.analyse_calibrated_method_selection import batch_tau, choice_weights
from scripts.calibrated_method_selection import (
    KnownOracle,
    condition_values,
    make_space,
    sample_calibration,
    seed_for,
)


def test_calibration_uses_only_distinct_multi_mutants():
    depth = np.r_[np.zeros(1), np.ones(12), np.full(400, 2)]
    a = sample_calibration(depth, 5)
    b = sample_calibration(depth, 6)
    assert len(a) == len(np.unique(a)) == 96
    assert (depth[a] > 1).all()
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, sample_calibration(depth, 5))


def test_conditioning_changes_only_observed_values():
    generated = np.arange(10.0)
    experimental = generated * -2
    known = np.array([0, 2, 9])
    actual = condition_values(generated, experimental, known)
    np.testing.assert_array_equal(actual[known], experimental[known])
    unknown = np.setdiff1d(np.arange(10), known)
    np.testing.assert_array_equal(actual[unknown], generated[unknown])
    np.testing.assert_array_equal(generated, np.arange(10.0))


def test_random_roles_have_separate_seeds():
    seeds = [
        seed_for(r, p, c, b)
        for r in range(1, 6)
        for p in ("GB1", "TEV")
        for c in range(10)
        for b in range(18)
    ]
    assert len(seeds) == len(set(seeds))


def test_recommendation_does_not_resolve_ties_using_experimental_fitness():
    prediction = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    weights = choice_weights(prediction)
    np.testing.assert_allclose(weights, [[0.5, 0.5, 0], [1 / 3, 1 / 3, 1 / 3]])
    experimental = np.array([0.9, 0.1, 0.0])
    np.testing.assert_allclose(weights @ experimental, [0.5, 1 / 3])


def test_tau_accounts_for_ties_and_undefined_rankings():
    a = np.array([[1, 1, 2, 3], [0, 0, 0, 0], [1, 2, 3, 4]])
    b = np.array([[1, 3, 2, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    expected = np.array([kendalltau(x, y).statistic for x, y in zip(a, b)])
    np.testing.assert_allclose(batch_tau(a, b), expected, equal_nan=True)


def test_fixed_recommendation_uses_training_proteins_only(monkeypatch):
    monkeypatch.setattr(analysis, "N_BOOT", 20)
    rows = []
    for protein in [*analysis.TRAINING, "GB1"]:
        for calibration in range(3):
            for trial in range(6):
                for method, value in [("a", 0.1), ("b", 0.8)]:
                    rows.append(
                        dict(
                            protein=protein,
                            calibration=calibration,
                            budget=96,
                            model="measured",
                            block=0,
                            trial=trial,
                            method=method,
                            regret=value if protein != "GB1" else 1 - value,
                        )
                    )
    table = pd.DataFrame(rows)
    prediction, bootstrap = analysis.fixed_predictions(
        table, 96, ["a", "b"], np.random.default_rng(3)
    )
    np.testing.assert_allclose(prediction, [0.1, 0.8])
    np.testing.assert_allclose(bootstrap, np.tile([0.1, 0.8], (20, 1)))


@pytest.mark.parametrize("method", list(METHODS))
def test_search_spends_full_budget_after_calibration(method):
    shape = np.array([3, 3, 3, 3])
    codes = np.indices(shape).reshape(4, -1).T.astype(np.int16)
    values = np.random.default_rng(0).normal(size=len(codes))
    known = np.r_[np.flatnonzero(np.count_nonzero(codes, axis=1) <= 1), [20, 30, 40, 50]]
    space = make_space(codes, shape, known, values)
    oracle = KnownOracle(values, 48, known)
    METHODS[method](space, oracle, make_features(space), 42)
    assert oracle.spent == 48
    assert not np.intersect1d(oracle.rows[len(known) :], known).size
    assert len(np.unique(oracle.rows)) == len(known) + 48
    assert oracle.best() >= values[known].max()
