"""Checks for distribution calibration and fixed landscape queries."""

from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mutascape.calibrated import (
    DEFAULT_REFERENCE_SIZE,
    QuantileCalibratedLandscape,
    _sample_reference_batches,
    empirical_quantiles,
)
from mutascape.domain import MutationDomain
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


def test_sampled_map_is_fixed_and_keeps_measurements():
    singles, pilot = calibration_data()
    model = QuantileCalibratedLandscape.fit("AAAA", singles, pilot)
    landscape = model.draw(73, reference_size=8, reference_seed=19)
    assert landscape.reference_mode == "sampled"
    assert landscape.reference_size == 8
    assert landscape.reference_seed == 19
    variants = singles.mutations.tolist() + pilot.mutations.tolist()
    values = landscape.evaluate_batch(variants)
    assert_array_equal(values[: len(singles)], singles.fitness)
    assert_allclose(landscape.evaluate_batch(variants[::-1]), values[::-1], atol=1e-12)
    assert_allclose([landscape[v] for v in variants], values, atol=1e-12)
    restored = QuantileCalibratedLandscape.from_dict(model.to_dict()).draw(
        73, reference_size=8, reference_seed=19
    )
    assert_array_equal(restored.score_knots, landscape.score_knots)
    assert_array_equal(restored.evaluate_batch(variants), values)
    other_sample = model.draw(73, reference_size=8, reference_seed=20)
    assert not np.array_equal(other_sample.score_knots, landscape.score_knots)
    assert_array_equal(other_sample.weights, landscape.weights)
    base = landscape.base_values([tuple(sorted(v.items())) for v in pilot.mutations])
    assert np.all(np.diff(values[len(singles) :][np.argsort(base)]) >= -1e-12)


def test_small_spaces_keep_the_exact_map():
    singles, pilot = calibration_data()
    model = QuantileCalibratedLandscape.fit("AAAA", singles, pilot)
    automatic = model.draw(73)
    exact = model.draw(73, reference_size=None)
    assert automatic.reference_mode == exact.reference_mode == "exact"
    assert automatic.reference_size == exact.reference_size == 11
    assert automatic.reference_seed is None
    assert_array_equal(automatic.score_knots, exact.score_knots)
    assert_array_equal(automatic.probabilities, exact.probabilities)


def test_reference_sampling_is_uniform_across_unequal_alphabets():
    domain = MutationDomain("AAA", ((1, "C"), (2, "C"), (2, "D"), (3, "C"), (3, "D"), (3, "E")))
    counts = Counter(
        key for batch in _sample_reference_batches(domain, 34_000, 12) for key in batch
    )
    assert sum(counts.values()) == 34_000
    assert len(counts) == domain.count() - len(domain.substitutions) - 1 == 17
    assert all(len(key) >= 2 and domain.contains(dict(key)) for key in counts)
    assert_allclose(np.array(list(counts.values())) / 34_000, 1 / 17, atol=0.007)


@pytest.mark.parametrize("sites,reference_size", [(21, DEFAULT_REFERENCE_SIZE), (70, 64)])
def test_large_search_spaces_do_not_need_enumeration(sites, reference_size, monkeypatch):
    singles, pilot = calibration_data()
    extra = pd.DataFrame(
        {
            "mutations": [{p: "C"} for p in range(5, sites + 1)],
            "fitness": [2.0 + 0.01 * p for p in range(5, sites + 1)],
        }
    )
    singles = pd.concat([singles, extra], ignore_index=True)
    model = QuantileCalibratedLandscape.fit("A" * sites, singles, pilot)

    def forbidden_enumeration(*args, **kwargs):
        raise AssertionError("Large spaces must use sampling")

    monkeypatch.setattr("mutascape.calibrated.product", forbidden_enumeration)
    options = {} if reference_size == DEFAULT_REFERENCE_SIZE else {"reference_size": reference_size}
    landscape = model.draw(42, **options)
    assert landscape.domain.count() == 2**sites
    assert landscape.reference_mode == "sampled"
    assert landscape.reference_size == reference_size
    variants = [{1: "C", sites: "C"}, {p: "C" for p in range(1, sites + 1)}]
    assert np.isfinite(landscape.evaluate_batch(variants)).all()
    assert_array_equal(landscape.evaluate_batch(singles.mutations.tolist()), singles.fitness)


@pytest.mark.parametrize(
    "options",
    [
        {"reference_size": 0},
        {"reference_size": 1},
        {"reference_size": 2.5},
        {"reference_size": True},
        {"reference_seed": -1},
        {"reference_seed": None},
        {"maximum_variants": 0},
    ],
)
def test_invalid_reference_settings_fail_before_generation(options):
    singles, pilot = calibration_data()
    model = QuantileCalibratedLandscape.fit("AAAA", singles, pilot)
    with pytest.raises(ValueError, match=next(iter(options))):
        model.draw(42, **options)
