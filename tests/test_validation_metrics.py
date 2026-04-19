"""Tests for the complete-space and scalar metrics in validation/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from mutascape.validation.metrics import (
    accessible_paths,
    fitness_distance_correlation,
    fitness_distance_correlation_from_tensor,
    local_optima_from_tensor,
    roughness_slope_ratio,
)


def test_roughness_slope_ratio_perfect_additive():
    df = pd.DataFrame({
        "fitness": [0.1, 0.2, 0.3, -0.1],
        "additive_prediction": [0.1, 0.2, 0.3, -0.1],
    })
    # Residuals are zero -> r/s = 0.
    assert roughness_slope_ratio(df) == pytest.approx(0.0)


def test_roughness_slope_ratio_known_value():
    df = pd.DataFrame({
        "fitness": [1.0, 2.0, 3.0],
        "additive_prediction": [0.5, 1.5, 2.5],
    })
    # Residuals all 0.5 -> RMSD = 0.5; slope = mean(|add|) = 1.5; r/s = 1/3.
    assert roughness_slope_ratio(df) == pytest.approx(0.5 / 1.5)


def test_fitness_distance_correlation_negative_when_farther_worse():
    df = pd.DataFrame({
        "depth": [1, 2, 3, 4, 5],
        "fitness": [0.5, 0.4, 0.3, 0.2, 0.1],
    })
    # Perfect monotonic decrease with depth -> r == -1.
    assert fitness_distance_correlation(df) == pytest.approx(-1.0)


def test_fitness_distance_correlation_from_tensor_negative_when_peak_is_funnel():
    # 2D tensor where fitness decreases with Hamming distance from peak at (0,0).
    tensor = np.array([
        [3.0, 2.0, 2.0],
        [2.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
    ])
    r = fitness_distance_correlation_from_tensor(tensor, peak_idx=(0, 0))
    assert r < 0


def test_local_optima_single_peak():
    tensor = np.array([
        [0.0, 1.0, 0.5],
        [0.8, 2.0, 1.2],
        [0.3, 1.5, 0.9],
    ])
    result = local_optima_from_tensor(tensor)
    assert result["n_optima"] == 1
    assert result["global_peak"] == (1, 1)
    assert result["peak_fitnesses"][0] == pytest.approx(2.0)
    # Basin of the single peak covers the whole tensor.
    assert sum(result["basin_sizes"]) == tensor.size


def test_local_optima_two_peaks():
    # Construct a tensor with two isolated peaks at (0,0) and (2,2).
    tensor = np.array([
        [3.0, 1.0, 0.0],
        [1.0, 0.5, 1.0],
        [0.0, 1.0, 3.0],
    ])
    result = local_optima_from_tensor(tensor)
    assert result["n_optima"] == 2
    assert sum(result["basin_sizes"]) == tensor.size


def test_local_optima_monotonic_tensor_single_peak():
    # Tensor where fitness = sum of indices, strictly monotonic -> single peak
    # at (top, top, top).
    shape = (4, 4, 4)
    tensor = np.zeros(shape)
    for idx in np.ndindex(*shape):
        tensor[idx] = sum(idx)
    result = local_optima_from_tensor(tensor)
    assert result["n_optima"] == 1
    assert result["global_peak"] == tuple(s - 1 for s in shape)


def test_accessible_paths_fully_accessible_on_monotonic_tensor():
    # Monotonic tensor -> every direct path from any source to the max is
    # strictly increasing, so accessible fraction is 1.
    shape = (3, 3, 3)
    tensor = np.zeros(shape)
    for idx in np.ndindex(*shape):
        tensor[idx] = sum(idx)
    result = accessible_paths(tensor, n_rsign_samples=500, seed=0)
    assert result["mean_accessible_fraction"] == pytest.approx(1.0)
    # Monotonic tensor has no reciprocal sign epistasis.
    assert result["reciprocal_sign_frac"] == pytest.approx(0.0)


def test_accessible_paths_peak_defaults_to_argmax():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(3, 3, 3))
    result = accessible_paths(tensor, n_sources=100, n_rsign_samples=200, seed=0)
    expected_peak = np.unravel_index(int(np.argmax(tensor)), tensor.shape)
    assert result["peak_idx"] == tuple(int(x) for x in expected_peak)


def test_accessible_paths_on_reciprocal_sign_square_is_zero():
    # A 2x2 sub-landscape designed so that neither monotonic path to the peak
    # is accessible: both single mutations decrease fitness from WT, but both
    # together increase fitness. Peak is (1, 1); sources at (0, 1) and (1, 0)
    # both have lower fitness than the direct paths would require, and (0, 0)
    # has both steps blocked.
    tensor = np.array([
        [1.0, 0.5],
        [0.5, 2.0],
    ])
    result = accessible_paths(tensor, n_rsign_samples=100, seed=0)
    # From (0,0) both paths must pass through fitness 0.5 which is lower
    # than the start (1.0), so no paths are accessible.
    # From (0,1) and (1,0) there's a single step to peak that strictly
    # increases, so those contribute 1.0 each.
    # Mean over the 3 sources with differing > 0: (0 + 1 + 1) / 3.
    assert result["mean_accessible_fraction"] == pytest.approx(2.0 / 3.0)
