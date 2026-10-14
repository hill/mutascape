"""Check the full-distribution evaluation domain and scale."""

import numpy as np
import pytest

from scripts.recompute_review_distributions import distribution_metrics, evaluation_rows


def test_evaluation_excludes_training_but_keeps_unsampled_multi_mutants():
    np.testing.assert_array_equal(evaluation_rows([0, 1, 2, 3, 4, 2], [3]), [2, 4, 5])


def test_full_wasserstein_retains_location_error_and_uses_evaluation_scale():
    real = np.array([999.0, 0.0, 2.0])
    generated = np.array([-999.0, 3.0, 5.0])
    result = distribution_metrics(real, generated, [1, 2])
    assert result == dict(w1=3.0, w1_raw=3.0, experimental_std=1.0, distribution_n=2)


def test_full_wasserstein_rejects_partial_generated_arrays():
    with pytest.raises(ValueError, match="missing fitness"):
        distribution_metrics([0.0, 1.0], [np.nan, 1.0], [0, 1])
