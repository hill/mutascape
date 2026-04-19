"""Tests for Walsh-Hadamard decomposition and fitness tensor builders."""

import numpy as np
import pandas as pd
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import RandomEpistasis
from mutascape.landscape import Landscape
from mutascape.validation.fourier import (
    fitness_tensor_from_landscape,
    fitness_tensor_from_real,
    observed_alphabets,
    variance_fraction_comparison,
    walsh_decompose,
    walsh_l1_distance,
)


def test_walsh_pure_additive_2site_binary():
    tensor = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = walsh_decompose(tensor)
    assert result[1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(0.0)


def test_walsh_pure_interaction_xor():
    tensor = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = walsh_decompose(tensor)
    assert result[1] == pytest.approx(0.0, abs=1e-12)
    assert result[2] == pytest.approx(1.0)


def test_walsh_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    for shape in [(3, 3), (4, 4, 4), (2, 3, 4, 5)]:
        tensor = rng.normal(size=shape)
        result = walsh_decompose(tensor)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-10)
        assert set(result.keys()) == set(range(1, len(shape) + 1))
        for v in result.values():
            assert v >= -1e-12


def test_walsh_scale_invariance():
    rng = np.random.default_rng(1)
    tensor = rng.normal(size=(4, 4, 4))
    a = walsh_decompose(tensor)
    b = walsh_decompose(tensor * 7.5 + 2.0)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-12)


def test_walsh_mixed_additive_perturbed():
    rng = np.random.default_rng(2)
    h1 = rng.normal(size=5)
    h2 = rng.normal(size=5)
    J = rng.normal(size=(5, 5)) * 0.2
    tensor = h1[:, None] + h2[None, :] + J
    result = walsh_decompose(tensor)
    assert result[1] > result[2]
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-12)


def test_walsh_handles_nan_by_mean_imputation():
    tensor = np.array([[0.0, 1.0], [2.0, 3.0]])
    with_nan = tensor.astype(float).copy()
    with_nan[0, 1] = np.nan
    # Should not error; decomposition should still be a valid distribution.
    result = walsh_decompose(with_nan)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-12)


def test_walsh_zero_variance_tensor():
    tensor = np.full((3, 3), 5.0)
    result = walsh_decompose(tensor)
    # All orders should be zero when there's no variance.
    for v in result.values():
        assert v == pytest.approx(0.0)


@pytest.fixture
def tiny_combinatorial_df():
    # Two positions, 3 observed AAs each, fully specified.
    rows = [
        {"mutations": {1: "A"}, "fitness": 0.1, "depth": 1},
        {"mutations": {1: "G"}, "fitness": 0.2, "depth": 1},
        {"mutations": {2: "L"}, "fitness": 0.3, "depth": 1},
        {"mutations": {2: "V"}, "fitness": -0.1, "depth": 1},
        {"mutations": {1: "A", 2: "L"}, "fitness": 0.5, "depth": 2},
        {"mutations": {1: "A", 2: "V"}, "fitness": 0.0, "depth": 2},
        {"mutations": {1: "G", 2: "L"}, "fitness": 0.6, "depth": 2},
        {"mutations": {1: "G", 2: "V"}, "fitness": 0.15, "depth": 2},
    ]
    return pd.DataFrame(rows)


def test_observed_alphabets_includes_wt(tiny_combinatorial_df):
    wt = {1: "M", 2: "K"}
    alphabets = observed_alphabets(tiny_combinatorial_df, (1, 2), wt)
    assert "M" in alphabets[1]
    assert "K" in alphabets[2]
    assert set(alphabets[1]) == {"M", "A", "G"}
    assert set(alphabets[2]) == {"K", "L", "V"}


def test_fitness_tensor_from_real_wt_is_zero_and_coverage(tiny_combinatorial_df):
    wt = {1: "M", 2: "K"}
    alphabets = {1: ["A", "G", "M"], 2: ["K", "L", "V"]}
    tensor, coverage = fitness_tensor_from_real(
        tiny_combinatorial_df, (1, 2), wt, alphabets
    )
    # WT cell (M, K) is set to 0 by convention.
    assert tensor[alphabets[1].index("M"), alphabets[2].index("K")] == 0.0
    # Known single-mutant entries match.
    assert tensor[alphabets[1].index("A"), alphabets[2].index("K")] == pytest.approx(0.1)
    assert tensor[alphabets[1].index("M"), alphabets[2].index("L")] == pytest.approx(0.3)
    # Known double-mutant entry matches.
    assert tensor[alphabets[1].index("A"), alphabets[2].index("L")] == pytest.approx(0.5)
    # Coverage: we filled 1 WT + 4 singles + 4 doubles = 9 of 9 cells.
    assert coverage == pytest.approx(1.0)


def test_fitness_tensor_from_landscape_matches_evaluate(tmp_path):
    distances = np.array([[0.0, 5.0], [5.0, 0.0]])
    cm = ContactMap.from_distance_matrix(distances, [10, 20], threshold=8.0)
    dms = pd.DataFrame({
        "position": [10, 10, 20, 20],
        "wildtype": ["A", "A", "L", "L"],
        "mutant": ["G", "V", "I", "F"],
        "fitness": [0.3, -0.1, 0.2, -0.4],
    })
    csv_path = tmp_path / "dms.csv"
    dms.to_csv(csv_path, index=False)
    ls = Landscape.from_dms(
        dms_path=csv_path,
        contact_map=cm,
        epistasis=RandomEpistasis(sigma=0.3, seed=0),
        alpha=1.0,
    )
    alphabets = {10: ["A", "G", "V"], 20: ["F", "I", "L"]}
    tensor = fitness_tensor_from_landscape(ls, [10, 20], alphabets)
    # Cell for WT combination should equal landscape evaluation with empty mutations.
    wt_idx = (alphabets[10].index("A"), alphabets[20].index("L"))
    assert tensor[wt_idx] == pytest.approx(ls.evaluate({}))
    # A specific mutant cell matches direct evaluation.
    mutant = {10: "G", 20: "I"}
    cell = (alphabets[10].index("G"), alphabets[20].index("I"))
    assert tensor[cell] == pytest.approx(ls.evaluate(mutant))


def test_variance_fraction_comparison_identical_tensors():
    rng = np.random.default_rng(3)
    tensor = rng.normal(size=(3, 3, 3))
    cmp = variance_fraction_comparison(tensor, tensor)
    assert cmp["max_abs_diff"] == pytest.approx(0.0)
    assert cmp["mean_abs_diff"] == pytest.approx(0.0)
    assert cmp["l1_abs_diff"] == pytest.approx(0.0)
    for k in cmp["real"]:
        assert cmp["real"][k] == pytest.approx(cmp["synth"][k])


def test_walsh_l1_distance_zero_for_identical():
    spectrum = {1: 0.5, 2: 0.3, 3: 0.15, 4: 0.05}
    assert walsh_l1_distance(spectrum, dict(spectrum)) == pytest.approx(0.0)


def test_walsh_l1_distance_sums_abs_differences():
    real = {1: 0.5, 2: 0.3, 3: 0.2}
    synth = {1: 0.4, 2: 0.5, 3: 0.1}
    # |0.1| + |-0.2| + |0.1| = 0.4
    assert walsh_l1_distance(real, synth) == pytest.approx(0.4)


def test_walsh_l1_distance_intersects_shared_orders():
    real = {1: 0.5, 2: 0.3, 3: 0.2}
    synth = {1: 0.5, 2: 0.4}  # missing order 3
    # Only orders 1 and 2 compared: 0 + 0.1 = 0.1
    assert walsh_l1_distance(real, synth) == pytest.approx(0.1)
