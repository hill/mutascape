"""Checks for distribution calibration and fixed landscape queries."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mutascape.experimental.quantile import QuantileCalibratedLandscape, empirical_quantiles
from tests.test_c2st_alternatives import calibration_data


def test_calibration_preserves_measurements_order_and_serialisation():
    singles, pilot = calibration_data()
    model = QuantileCalibratedLandscape.fit("AAAA", singles, pilot)
    draw = model.draw(73)
    variants = [{i + 1: "C" for i, b in enumerate(bits) if b} for bits in product((0, 1), repeat=4)]
    keys = [tuple(sorted(m.items())) for m in variants]
    expected = model.reference.draw(73).evaluate_batch(variants)
    assert_allclose(draw.base_values(keys)[5:], expected[5:], atol=1e-12)
    assert_array_equal(draw.evaluate_batch(singles.mutations.tolist()), singles.fitness)
    values = draw.evaluate_batch(variants)
    assert_allclose(draw.evaluate_batch(variants[::-1]), values[::-1], atol=1e-12)
    assert_allclose(draw.evaluate_batch(variants[:3]), values[:3], atol=1e-12)
    restored = QuantileCalibratedLandscape.from_dict(model.to_dict()).draw(73)
    assert_allclose(restored.evaluate_batch(variants), values, atol=1e-12)
    eligible = np.array([len(m) > 1 for m in variants])
    order = np.argsort(expected[eligible])
    assert np.all(np.diff(values[eligible][order]) >= -1e-12)
    assert draw.evaluate_batch([]).shape == (0,)
    with pytest.raises(ValueError, match="maximum_variants"):
        model.draw(1, maximum_variants=8)


def test_quantile_map_retains_recorded_mass_without_flat_artificial_tails():
    probabilities = np.linspace(0.001, 0.999, 1000)
    values = empirical_quantiles(np.array([-2.0, 0.0, 0.0, 0.0, 2.0]), probabilities)
    assert np.all(np.diff(values) >= 0)
    assert values[0] < -2 and values[-1] > 2
    assert np.count_nonzero(values == 0) == 600
    assert len(np.unique(values[values > 0])) == np.count_nonzero(values > 0)
