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
    assert result.synth_n_optima == pytest.approx(result.real_n_optima, rel=0.15)
    # Richer spectral metrics. The order-2 cosine is low (~0.34): at sigma=0 the
    # global-epistasis term reproduces the order budget but not which pairs
    # actually interact, a structural mismatch the scalar walsh_l1 hides.
    assert result.walsh_order_w1 == pytest.approx(0.282, abs=0.05)
    assert result.walsh_cos_order2 == pytest.approx(0.336, abs=0.08)
    assert result.walsh_cos_ge2 == pytest.approx(0.267, abs=0.08)


def test_trpb_headline_reference_metrics_regression():
    result = evaluate_reference_config(TRPB_HEADLINE)

    assert result.walsh_l1 == pytest.approx(0.107, abs=0.03)
    assert result.fitness_w1 == pytest.approx(0.028, abs=0.01)
    assert result.synth_walsh[1] == pytest.approx(0.242, abs=0.03)
    assert result.synth_walsh[2] == pytest.approx(0.410, abs=0.03)
    assert result.synth_walsh[3] == pytest.approx(0.201, abs=0.03)
    assert result.synth_walsh[4] == pytest.approx(0.148, abs=0.03)
    assert result.synth_n_optima == pytest.approx(result.real_n_optima, rel=0.15)
    # Richer spectral metrics. TrpB reproduces the order-2 interaction pattern
    # markedly better than GB1 (cos ~0.62 vs ~0.34), consistent with its much
    # larger and cleaner MSA.
    assert result.walsh_order_w1 == pytest.approx(0.061, abs=0.03)
    assert result.walsh_cos_order2 == pytest.approx(0.622, abs=0.08)
    assert result.walsh_cos_ge2 == pytest.approx(0.466, abs=0.08)
