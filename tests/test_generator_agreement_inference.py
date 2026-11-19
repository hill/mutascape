import numpy as np
import pandas as pd
import pytest

from scripts.analyse_generator_agreement import (
    contrast,
    holm_adjust,
    paired_differences,
    sign_flip_pvalue,
)


def test_pairing_uses_calibration_ids_and_keeps_zero_agreement():
    frame = pd.DataFrame(
        {
            "protein": ["GB1"] * 4,
            "budget": [48] * 4,
            "calibration": [1, 0, 0, 1],
            "arm": ["sigmoid", "calibrated", "sigmoid", "calibrated"],
            "eligible_pairs": [10] * 4,
            "matching_pairs": [8, 0, 5, 9],
            "kendall_tau": [0.5, np.nan, 0.4, 0.8],
        }
    )
    agreement = paired_differences(frame, "agreement", 48)
    np.testing.assert_allclose(agreement.difference, [-0.5, 0.1])
    tau = paired_differences(frame, "kendall_tau", 48)
    assert tau.calibration.tolist() == [1]
    assert tau.difference.iloc[0] == pytest.approx(0.3)


def test_exact_two_sided_sign_flip_counts_both_extremes():
    pvalue, kind = sign_flip_pvalue([np.ones(4)], np.random.default_rng(1), 100)
    assert pvalue == 2 / 16
    assert kind == "exact paired sign-flip"


def test_zero_contrast_has_pvalue_one():
    pvalue, _ = sign_flip_pvalue([np.zeros(4)], np.random.default_rng(1), 100)
    assert pvalue == 1.0


def test_overall_contrast_weights_proteins_equally():
    result = contrast([np.ones(4), np.zeros(10)], np.random.default_rng(1), 100)
    assert result["difference"] == 0.5
    assert result["ci_low"] == result["ci_high"] == 0.5


def test_holm_adjustment_restores_input_order():
    actual = holm_adjust(np.array([0.04, 0.001, 0.03]))
    np.testing.assert_allclose(actual, [0.06, 0.003, 0.06])
