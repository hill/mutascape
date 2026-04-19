"""Tests for tensor-level and distributional comparison scalars."""

import numpy as np
import pytest

from mutascape.validation.comparisons import (
    fitness_wasserstein,
    tensor_pearson_r,
)


def test_tensor_pearson_r_perfect_self_correlation():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(4, 4, 4))
    assert tensor_pearson_r(tensor, tensor) == pytest.approx(1.0)


def test_tensor_pearson_r_anti_correlation():
    rng = np.random.default_rng(1)
    tensor = rng.normal(size=(3, 3, 3))
    assert tensor_pearson_r(tensor, -tensor) == pytest.approx(-1.0)


def test_tensor_pearson_r_ignores_nans_in_either_side():
    real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    synth = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    # Matched finite entries are [1,2,4,5] vs [1,2,4,5] -> r = 1.
    assert tensor_pearson_r(real, synth) == pytest.approx(1.0)


def test_tensor_pearson_r_zero_variance_returns_nan():
    real = np.array([1.0, 2.0, 3.0])
    synth = np.array([7.0, 7.0, 7.0])
    assert np.isnan(tensor_pearson_r(real, synth))


def test_tensor_pearson_r_shape_mismatch_raises():
    a = np.zeros((2, 3))
    b = np.zeros((3, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        tensor_pearson_r(a, b)


def test_tensor_pearson_r_insufficient_overlap_returns_nan():
    real = np.array([1.0, np.nan, np.nan])
    synth = np.array([np.nan, 2.0, np.nan])
    # Zero matched finite entries.
    assert np.isnan(tensor_pearson_r(real, synth))


def test_fitness_wasserstein_zero_for_identical():
    rng = np.random.default_rng(2)
    tensor = rng.normal(size=(5, 5))
    assert fitness_wasserstein(tensor, tensor) == pytest.approx(0.0)


def test_fitness_wasserstein_shift_invariance():
    rng = np.random.default_rng(3)
    t = rng.normal(size=1000)
    shifted = t + 2.5
    # Wasserstein-1 between a distribution and its translate equals the shift.
    assert fitness_wasserstein(t, shifted) == pytest.approx(2.5, rel=0.05)


def test_fitness_wasserstein_handles_nans():
    real = np.array([0.0, 1.0, 2.0, np.nan])
    synth = np.array([0.0, 1.0, 2.0, np.nan])
    assert fitness_wasserstein(real, synth) == pytest.approx(0.0)
