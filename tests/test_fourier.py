"""Tests for Walsh-Hadamard decomposition and fitness tensor builders."""

import numpy as np
import pandas as pd
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import RandomEpistasis
from mutascape.landscape import Landscape
from mutascape.validation.fourier import (
    binary_cube_spectral_metrics,
    binary_walsh_coefficients,
    fitness_tensor_from_landscape,
    fitness_tensor_from_real,
    index_orders,
    observed_alphabets,
    order_wasserstein,
    spectral_cosine,
    truncate_fractions,
    variance_fraction_comparison,
    walsh_decompose,
    walsh_l1_distance,
    walsh_signed_effects,
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


# --- order_wasserstein ---------------------------------------------------


def test_order_wasserstein_zero_for_identical():
    spectrum = {1: 0.8, 2: 0.15, 3: 0.05}
    assert order_wasserstein(spectrum, dict(spectrum)) == pytest.approx(0.0)


def test_order_wasserstein_penalises_distant_orders_more():
    base = {1: 0.8, 2: 0.2, 3: 0.0}
    adjacent = {1: 0.8, 2: 0.1, 3: 0.1}  # moved 0.1 from O2 to O3
    distant = {1: 0.7, 2: 0.2, 3: 0.1}  # moved 0.1 from O1 to O3
    w_adj = order_wasserstein(base, adjacent)
    w_dist = order_wasserstein(base, distant)
    assert w_adj == pytest.approx(0.1)
    assert w_dist == pytest.approx(0.2)
    assert w_dist > w_adj


def test_order_wasserstein_matches_cdf_formula():
    real = {1: 0.6, 2: 0.4}
    synth = {1: 0.4, 2: 0.6}
    # CDFs: real (0.6, 1.0), synth (0.4, 1.0). |0.2| + |0| = 0.2
    assert order_wasserstein(real, synth) == pytest.approx(0.2)


def test_order_wasserstein_renormalises_inputs():
    # Un-normalised raw squared sums should give the same answer as fractions.
    raw = {1: 6.0, 2: 4.0}
    frac = {1: 0.6, 2: 0.4}
    assert order_wasserstein(raw, frac) == pytest.approx(0.0)


# --- spectral_cosine -----------------------------------------------------


def test_spectral_cosine_identical_is_one():
    v = np.array([1.0, -2.0, 0.5, 3.0])
    assert spectral_cosine(v, v) == pytest.approx(1.0)


def test_spectral_cosine_is_scale_invariant():
    # The documented property: cosine cannot select magnitude, so it is the
    # structural companion to the magnitude-sensitive per-order spectra.
    v = np.array([1.0, -2.0, 0.5, 3.0])
    assert spectral_cosine(v, 7.5 * v) == pytest.approx(1.0)


def test_spectral_cosine_sign_flip_is_minus_one():
    v = np.array([1.0, -2.0, 0.5, 3.0])
    assert spectral_cosine(v, -v) == pytest.approx(-1.0)


def test_spectral_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 1.0])
    assert spectral_cosine(a, b) == pytest.approx(0.0)


def test_spectral_cosine_zero_norm_returns_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert spectral_cosine(v, np.zeros(3)) == 0.0


# --- walsh_signed_effects / binary_walsh_coefficients --------------------


def test_walsh_signed_effects_grouped_matches_decompose():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(3, 2, 4))
    coeffs, orders = walsh_signed_effects(tensor)
    total = float(np.var(tensor)) * tensor.size
    L = tensor.ndim
    regrouped = {
        k: float((coeffs[orders == k] ** 2).sum()) / total for k in range(1, L + 1)
    }
    reference = walsh_decompose(tensor)
    for k in reference:
        assert regrouped[k] == pytest.approx(reference[k])


def test_walsh_signed_effects_pure_additive_has_no_high_order():
    # An additive tensor (sum of per-axis effects) carries variance only at
    # order 1; the order >= 2 coefficient norm must be ~0.
    h0 = np.array([0.0, 1.0, -1.0])[:, None, None]
    h1 = np.array([0.0, 0.5])[None, :, None]
    h2 = np.array([0.0, 2.0, -1.0, 0.3])[None, None, :]
    tensor = h0 + h1 + h2
    coeffs, orders = walsh_signed_effects(tensor)
    assert np.linalg.norm(coeffs[orders >= 2]) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(coeffs[orders == 1]) > 0.0


def test_walsh_signed_effects_xor_has_no_order_one():
    # 2x2 XOR is pure second-order interaction: no order-1 variance.
    tensor = np.array([[0.0, 1.0], [1.0, 0.0]])
    coeffs, orders = walsh_signed_effects(tensor)
    assert np.linalg.norm(coeffs[orders == 1]) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(coeffs[orders == 2]) > 0.0


def test_binary_walsh_coefficients_preserves_energy():
    # Orthonormal transform: Parseval, sum of squares preserved.
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(2, 2, 2, 2))
    coeffs = binary_walsh_coefficients(cube)
    assert float((coeffs**2).sum()) == pytest.approx(float((cube**2).sum()))


def test_index_orders_counts_hamming_weight():
    orders = index_orders(3)
    assert orders.tolist() == [0, 1, 1, 2, 1, 2, 2, 3]


# --- truncate_fractions / max_order --------------------------------------


def test_truncate_fractions_none_is_identity():
    fracs = {1: 0.8, 2: 0.15, 3: 0.05}
    assert truncate_fractions(fracs, None) == fracs


def test_truncate_fractions_keeps_and_renormalises():
    fracs = {1: 0.8, 2: 0.15, 3: 0.05}
    out = truncate_fractions(fracs, 2)
    assert set(out) == {1, 2}
    assert sum(out.values()) == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.8 / 0.95)


def test_truncate_fractions_zero_total_is_safe():
    out = truncate_fractions({1: 0.0, 2: 0.0}, 1)
    assert out == {1: 0.0}


def test_binary_cube_spectral_metrics_max_order_changes_only_per_order():
    # max_order truncates the per-order distances but leaves the cosines, which
    # are computed on the coefficient vectors, untouched.
    rng = np.random.default_rng(3)
    L = 6
    n = 4000
    cell_idx = rng.integers(0, 2**L, n)
    real = rng.normal(size=n)
    synth = 0.8 * real + 0.4 * rng.normal(size=n)
    full = binary_cube_spectral_metrics(real, synth, cell_idx, L, max_order=None)
    trunc = binary_cube_spectral_metrics(real, synth, cell_idx, L, max_order=2)
    assert trunc["cos_order2"] == pytest.approx(full["cos_order2"])
    assert trunc["cos_ge2"] == pytest.approx(full["cos_ge2"])
    # Truncating away the noisy tail changes the per-order distances.
    assert trunc["order_w1"] != pytest.approx(full["order_w1"])
