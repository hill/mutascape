"""Checks for singles-only inputs and repeatable synthetic interactions."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mutascape.experimental.singles import SimpleLandscape, _pair
from scripts.investigate_singles_only import local_geometry
from scripts.singles_warm_start import DMSOracle


def measurements():
    return pd.DataFrame(
        {"mutations": [{}, {1: "C"}, {2: "D"}, {3: "E"}], "fitness": [1.0, 1.2, 0.8, 1.5]}
    )


@pytest.mark.parametrize("mode", ["linear", "sigmoid", "pool"])
def test_preserved_singles_and_stable_arbitrary_queries(mode):
    singles = measurements()
    model = SimpleLandscape.from_singles("AAA", singles, mode=mode, seed=73)
    np.testing.assert_array_equal(model.evaluate_batch(singles.mutations.tolist()), singles.fitness)
    variants = ["A1C+A2D", "A1C+A3E", "A1C+A2D+A3E"]
    values = model.evaluate_batch(variants)
    np.testing.assert_allclose(model.evaluate_batch(variants[::-1]), values[::-1], atol=1e-12)
    assert model[variants[0]] == values[0]
    assert model[{1: "C", 2: "D"}] == values[0]
    changed = SimpleLandscape.from_singles("AAA", singles, mode=mode, seed=74)
    assert not np.allclose(values, changed.evaluate_batch(variants))


def test_contacts_restrict_pair_terms_without_changing_singles():
    singles = measurements()
    model = SimpleLandscape.from_singles(
        "AAA", singles, mode="linear", seed=3, strength=0.5, contacts={(1, 3)}
    )
    assert model["A1C+A2D"] == pytest.approx(1.0)
    expected = 1.0 + 0.2 - 0.2 + 0.5 + 0.5 * model.scale * _pair((1, "C"), (3, "E"), "AAA", 3)
    assert model["A1C+A2D+A3E"] == pytest.approx(expected)
    no_pairs = SimpleLandscape.from_singles(
        "AAA", singles, mode="linear", strength=10, contacts=set()
    )
    assert no_pairs["A1C+A2D+A3E"] == pytest.approx(1.5)


def test_large_library_queries_do_not_enumerate_genotypes():
    singles = pd.DataFrame(
        {
            "mutations": [{}, *[{i: "C"} for i in range(1, 101)]],
            "fitness": [1.0, *np.linspace(0.2, 1.5, 100)],
        }
    )
    model = SimpleLandscape.from_singles("A" * 100, singles)
    assert model.domain.count() == 2**100
    assert np.isfinite(model["A1C+A99C+A100C"])


def test_combination_labels_cannot_enter_model_construction():
    singles = measurements()
    extra = pd.DataFrame({"mutations": [{1: "C", 2: "D"}], "fitness": [99.0]})
    with pytest.raises(ValueError, match="only wild type and single"):
        SimpleLandscape.from_singles("AAA", pd.concat([singles, extra]))
    model = SimpleLandscape.from_singles("AAA", singles)
    with pytest.raises(ValueError, match="missing single-mutant"):
        model["A1D+A2D"]
    with pytest.raises(ValueError, match="claims wild type"):
        model["G1C"]


def test_local_spectra_separate_additive_and_pair_variance():
    bits = np.array([[bool(i & (1 << j)) for j in range(3)] for i in range(8)])
    spins = 1 - 2 * bits.astype(float)
    cubes = np.arange(8)[None, :]
    np.testing.assert_allclose(local_geometry(spins.sum(axis=1), cubes)[0], [1, 0, 0])
    pairs = sum(spins[:, i] * spins[:, j] for i, j in combinations(range(3), 2))
    np.testing.assert_allclose(local_geometry(pairs, cubes)[0], [0, 1, 0])
    np.testing.assert_allclose(local_geometry(np.ones(8), cubes)[0], [0, 0, 0])


def test_dms_campaign_charges_only_new_combinations():
    oracle = DMSOracle(np.arange(6, dtype=float), budget=2, known=[0, 1, 2])
    assert oracle.spent == 0
    assert oracle.remaining == 2
    np.testing.assert_array_equal(oracle.observed, [0, 1, 2])
    np.testing.assert_array_equal(oracle.query([1, 3, 3, 4, 5]), [1, 3, 3, 4, -np.inf])
    assert oracle.spent == 2
    assert oracle.remaining == 0
    assert oracle.best() == 4
