"""Check method selection and uncertainty calculations."""

import numpy as np

from scripts.analyse_search_selection import (
    choice_weights,
    selected_performance,
    synthetic_bootstrap,
    weighted_disagreement,
)
from scripts.audit_search_rankings import batch_tau


def test_tied_selection_is_uniform_after_rounding():
    predictions = np.array([[1.0, 1.0 + 1e-14, 2.0], [2.0, 1.0, 3.0]])
    truth = np.array([0.4, 0.2, 0.0])
    np.testing.assert_allclose(choice_weights(predictions), [[0.5, 0.5, 0], [0, 1, 0]])
    np.testing.assert_allclose(selected_performance(predictions, truth), [0.3, 0.2])


def test_pair_disagreement_accounts_for_ties_and_effect_size():
    truth = np.array([0.0, 0.1, 1.0])
    assert weighted_disagreement(truth, truth) == 0
    assert weighted_disagreement(-truth, truth) == 1
    assert weighted_disagreement(np.zeros(3), truth) == 0.5
    np.testing.assert_allclose(weighted_disagreement(np.array([0.1, 0.0, 1]), truth), 0.05)
    assert np.isnan(weighted_disagreement(truth, np.ones(3)))


def test_bootstrap_includes_landscape_sample_uncertainty():
    values = np.repeat(np.array([0.0, 0.5, 1.0])[:, None, None], 20, axis=1)
    values = np.concatenate([values, 1 - values], axis=2)
    sampled = synthetic_bootstrap(values, np.random.default_rng(761), replicates=4000)
    np.testing.assert_allclose(sampled.mean(axis=0), [0.5, 0.5], atol=0.015)
    np.testing.assert_allclose(sampled.sum(axis=1), 1)
    assert sampled[:, 0].std() > 0.2


def test_tau_remains_undefined_for_identical_method_means():
    assert np.isnan(batch_tau(np.ones(4), np.arange(4)))
    assert batch_tau(np.arange(4), np.arange(4)) == 1
