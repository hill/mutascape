"""Checks for the pilot-calibrated burden model used in the CreiLOV assessment."""

import importlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mutascape.completion import dms_pairwise_unit_coefficient


@pytest.fixture
def prototype(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("assess_creilov_pilot")


@pytest.mark.parametrize("burden", [False, True])
def test_corrections_preserve_wild_type_and_singles(prototype, burden):
    effects = {(1, "C"): 0.2, (2, "D"): -0.1}
    variants = [{}, {1: "C"}, {2: "D"}]
    base, corrections = prototype.pair_design(variants, effects, burden=burden)
    np.testing.assert_allclose(base, [0, 0.2, -0.1])
    np.testing.assert_array_equal(corrections, np.zeros((3, 2)))
    np.testing.assert_array_equal(prototype.pair_noise(variants, "AA", 42), [0, 0, 0])


def test_vectorised_noise_matches_scalar_pair_terms(prototype):
    variants = [{}, {1: "C", 2: "D"}, {1: "C", 2: "D", 3: "E"}]
    expected = [
        sum(
            dms_pairwise_unit_coefficient(a, b, "AAA", 42) / 2
            for a, b in combinations(sorted(m.items()), 2)
        )
        for m in variants
    ]
    actual = prototype.pair_noise(variants, "AAA", 42)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(prototype.pair_noise(variants[::-1], "AAA", 42), actual[::-1])


def test_pair_normalisation_has_unit_marginal_variance(prototype):
    values = [prototype.pair_noise([{1: "C", 2: "D"}], "AA", seed)[0] for seed in range(1000)]
    assert np.var(values) == pytest.approx(1, abs=0.15)


def test_pilot_fit_is_repeatable(prototype):
    effects = {(1, "C"): 0.2, (2, "D"): -0.1, (3, "E"): 0.05}
    variants = [{1: "C", 2: "D"}, {1: "C", 3: "E"}, {2: "D", 3: "E"}, {1: "C", 2: "D", 3: "E"}]
    pilot = pd.DataFrame(
        dict(mutations=variants, depth=list(map(len, variants)), fitness=[-0.1, 0.03, -0.2, -0.3])
    )
    first, scale = prototype.fit_pilot(pilot, effects)
    second, second_scale = prototype.fit_pilot(pilot.copy(), effects)
    np.testing.assert_array_equal(first[-1].coef_, second[-1].coef_)
    assert scale == second_scale
    assert np.isfinite(scale) and scale > 0
