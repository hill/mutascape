"""Regression coverage for the frozen publication reference configs."""

import pytest

from mutascape.validation.reference_configs import (
    GB1_HEADLINE,
    TRPB_HEADLINE,
    evaluate_reference_config,
)


def test_gb1_headline_reference_metrics_regression():
    result = evaluate_reference_config(GB1_HEADLINE)

    assert result.walsh_l1 == pytest.approx(0.317, abs=0.02)
    assert result.fitness_w1 == pytest.approx(0.036, abs=0.01)
    assert result.synth_walsh[1] == pytest.approx(0.304, abs=0.03)
    assert result.synth_walsh[2] == pytest.approx(0.470, abs=0.03)
    assert result.synth_walsh[3] == pytest.approx(0.155, abs=0.03)
    assert result.synth_walsh[4] == pytest.approx(0.072, abs=0.02)


def test_trpb_headline_reference_metrics_regression():
    result = evaluate_reference_config(TRPB_HEADLINE)

    assert result.walsh_l1 == pytest.approx(0.107, abs=0.03)
    assert result.fitness_w1 == pytest.approx(0.028, abs=0.01)
    assert result.synth_walsh[1] == pytest.approx(0.242, abs=0.03)
    assert result.synth_walsh[2] == pytest.approx(0.410, abs=0.03)
    assert result.synth_walsh[3] == pytest.approx(0.201, abs=0.03)
    assert result.synth_walsh[4] == pytest.approx(0.148, abs=0.03)
