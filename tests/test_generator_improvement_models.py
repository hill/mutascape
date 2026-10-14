"""Check exact single-site conditioning and query-order consistency."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from scripts.generator_improvement_models import (
    MODELS,
    candidate_values,
    replace_single_effects,
    single_site_decomposition,
)


def example():
    codes = np.asarray(list(product(range(2), repeat=3)))
    effects = codes * [0.2, -0.4, 0.5]
    values = 0.7 + effects.sum(axis=1)
    variants = [{i + 1: "G" for i, state in enumerate(row) if state} for row in codes]
    known = np.flatnonzero(codes.sum(axis=1) <= 1)
    singles = pd.DataFrame({"mutations": [variants[i] for i in known], "fitness": values[known]})
    return codes, values, variants, known, singles


def test_single_site_decomposition_recovers_additive_landscape():
    codes, values, _, known, _ = example()
    wt, effects, result = single_site_decomposition(codes, values[known])
    assert wt == 0.7
    np.testing.assert_allclose(result, values)
    np.testing.assert_allclose(effects, codes * [0.2, -0.4, 0.5])


@pytest.mark.parametrize("model", MODELS)
def test_candidates_preserve_observations_and_query_order(model):
    codes, values, variants, known, singles = example()
    prior = 2 + 3 * codes.sum(axis=1) + codes[:, 0] + codes[:, 0] * codes[:, 2]
    actual, _ = candidate_values(model, codes, singles, variants, "AAA", 42, prior)
    np.testing.assert_array_equal(actual[known], values[known])
    permutation = np.random.default_rng(1).permutation(len(codes))
    permuted_known = np.flatnonzero(codes[permutation].sum(axis=1) <= 1)
    permuted_singles = pd.DataFrame(
        {
            "mutations": [variants[permutation[i]] for i in permuted_known],
            "fitness": values[permutation][permuted_known],
        }
    )
    permuted, _ = candidate_values(
        model,
        codes[permutation],
        permuted_singles,
        [variants[i] for i in permutation],
        "AAA",
        42,
        prior[permutation],
    )
    np.testing.assert_allclose(permuted, actual[permutation])


def test_msa_correction_preserves_only_interaction_residual():
    codes, values, _, known, _ = example()
    prior = 3 + 2 * (values - 0.7) + 0.4 * codes[:, 0] * codes[:, 2]
    actual, metadata = replace_single_effects(codes, values[known], prior)
    assert metadata["interaction_scale"] == pytest.approx(0.5)
    np.testing.assert_allclose(actual, values + 0.2 * codes[:, 0] * codes[:, 2])


def test_decomposition_rejects_combination_labels():
    codes, values, _, _, _ = example()
    with pytest.raises(ValueError, match="only for WT"):
        single_site_decomposition(codes, values)


def test_negative_msa_single_site_association_removes_interactions():
    codes, values, _, known, _ = example()
    prior = 5 - values + codes[:, 0] * codes[:, 2]
    actual, metadata = replace_single_effects(codes, values[known], prior, regression=True)
    assert metadata["interaction_scale"] == 0
    np.testing.assert_allclose(actual, values)


def test_uniform_control_matches_expected_pair_variance():
    from scripts.replicate_generator_improvements import matched_strength

    _, _, _, _, singles = example()
    strength = matched_strength(singles)
    expected = np.mean([abs(0.2 * -0.4), abs(0.2 * 0.5), abs(-0.4 * 0.5)])
    assert (strength * np.std([0.2, -0.4, 0.5])) ** 2 == pytest.approx(expected)


def test_configuration_selection_excludes_target_protein():
    from scripts.analyse_generator_improvements import excluded_protein_choice

    proteins, candidates = ["A", "B", "C"], ["first", "second"]
    losses = {(p, m): float(m == "second") for p in proteins for m in candidates}
    assert excluded_protein_choice("A", proteins, candidates, losses) == 0
    losses["A", "first"] = 1e9
    losses["A", "second"] = -1e9
    assert excluded_protein_choice("A", proteins, candidates, losses) == 0
