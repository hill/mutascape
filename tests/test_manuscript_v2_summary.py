import pandas as pd
import pytest

from scripts.manuscript_v2 import aggregate_search_summary


def test_agreement_gives_calibration_samples_equal_weight():
    summary = pd.DataFrame(
        {
            "protein": ["GB1", "GB1"],
            "budget": [48, 48],
            "arm": ["calibrated", "calibrated"],
            "calibration": [0, 1],
            "kendall_tau": [0.4, 0.8],
            "tied": [False, False],
            "matching_pairs": [8, 45],
            "eligible_pairs": [10, 50],
        }
    )
    result = aggregate_search_summary(summary).iloc[0]
    assert result.agreement == pytest.approx(0.85)
    assert result.agreement != pytest.approx(53 / 60)
    assert result.agreement_sd == pytest.approx(0.1 / 2**0.5)
    assert result.agreement_n == 2
    assert result.matching == 53
    assert result.eligible == 60
    assert result.kendall_tau == pytest.approx(0.6)
