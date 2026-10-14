"""Check benchmark efficacy metrics against small examples and SciPy."""

import numpy as np
from scipy.stats import kendalltau, spearmanr

from scripts.analyse_benchmark_efficacy import metrics, spearman


def test_method_ranks_retain_ties_and_match_scipy():
    predictions = np.array([[0.1, 0.1, 0.4, 0.2], [0.4, 0.2, 0.1, 0.1]])
    actual = np.array([[0.1, 0.2, 0.3, 0.3], [0.3, 0.3, 0.2, 0.1]])
    result = metrics(predictions, actual, actual)
    np.testing.assert_allclose(
        result["spearman"],
        [spearmanr(a, b).statistic for a, b in zip(predictions, actual)],
    )
    np.testing.assert_allclose(
        result["tau"],
        [kendalltau(a, b).statistic for a, b in zip(predictions, actual)],
    )


def test_mspd_and_selected_regret_score_tied_choices_equally():
    prediction = np.array([0.1, 0.1, 0.4])
    actual = np.array([0.2, 0.3, 0.5])
    result = metrics(prediction, actual, 2 * actual)
    np.testing.assert_allclose(result["mspd"], 0.02)
    np.testing.assert_allclose(result["selected_regret"], 0.25)
    np.testing.assert_allclose(result["remaining_regret"], 0.5)


def test_constant_method_ranking_stays_undefined():
    result = spearman(np.ones((2, 4)), np.arange(8).reshape(2, 4))
    assert np.isnan(result).all()


def test_numerical_noise_does_not_break_method_ties():
    first = np.array([0.1, 0.1 + 1e-14, 0.4])
    second = np.array([0.1, 0.1, 0.4])
    np.testing.assert_allclose(spearman(first, second), 1)
