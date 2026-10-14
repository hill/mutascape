"""Check external generator definitions and SLIP score conventions."""

import numpy as np
import pytest

from scripts.external_generator_models import (
    nk_values,
    pair_effect,
    potts_components,
    rmf_values,
    slip_scales,
)


def test_nk_zero_has_no_pairwise_epistasis():
    codes = np.indices((3, 4)).reshape(2, -1).T
    values = nk_values(codes, (3, 4), 0, 7).reshape(3, 4)
    np.testing.assert_allclose(values - values[:, :1] - values[:1, :] + values[0, 0], 0, atol=1e-15)


def test_nk_scores_are_independent_of_query_order():
    codes = np.indices((2, 3, 2)).reshape(3, -1).T
    expected = nk_values(codes, (2, 3, 2), 1, 7)
    np.testing.assert_array_equal(nk_values(codes[::-1], (2, 3, 2), 1, 7), expected[::-1])
    with pytest.raises(ValueError):
        nk_values(codes, (2, 3, 2), 3, 7)


def test_rmf_zero_noise_is_additive_and_subset_stable():
    codes = np.indices((2, 3, 2)).reshape(3, -1).T
    values = rmf_values(codes, (2, 3, 2), 8, noise=0).reshape(2, 3, 2)
    np.testing.assert_allclose(
        values[1, :, 1] - values[1, :, 0] - values[0, :, 1] + values[0, :, 0], 0
    )
    expected = rmf_values(codes, (2, 3, 2), 8)
    np.testing.assert_array_equal(rmf_values(codes[::2], (2, 3, 2), 8), expected[::2])


def test_potts_reduction_keeps_fixed_background_and_sign():
    rng = np.random.default_rng(11)
    weight = rng.normal(size=(4, 3, 4, 3))
    weight = (weight + weight.transpose(2, 3, 0, 1)) / 2
    for i in range(4):
        weight[i, :, i, :] = 0
    bias = rng.normal(size=(4, 3))
    wt = np.array([0, 1, 2, 1])
    field = potts_components(weight, bias, wt)

    def direct(seq):
        return sum(bias[i, a] for i, a in enumerate(seq)) + sum(
            weight[i, seq[i], j, seq[j]] for i in range(4) for j in range(i + 1, 4)
        )

    for a in range(3):
        for b in range(3):
            seq = wt.copy()
            seq[[0, 2]] = [a, b]
            reduced = field[0, a] + field[2, b] + pair_effect(weight, wt, 0, 2)[a, b]
            assert reduced == pytest.approx(direct(seq) - direct(wt))
    np.testing.assert_allclose(field[np.arange(4), wt], 0, atol=1e-14)


def test_slip_tuning_reaches_requested_horizon():
    weight = np.zeros((3, 2, 3, 2))
    for i in range(3):
        for j in range(i + 1, 3):
            weight[i, 1, j, 1] = weight[j, 1, i, 1] = -0.1
    field = np.array([[0, 1.0], [0, 2.0], [0, 3.0]])
    sf, se, report = slip_scales(weight, field, np.zeros(3, dtype=int), 4)
    assert 1 - 2 * sf * report["mean_adaptive_single"] / (
        se * report["mean_adaptive_pair"]
    ) == pytest.approx(4)
