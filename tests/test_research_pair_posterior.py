"""Checks for the exploratory pair-field conditioning calculation."""

import numpy as np
from scipy.linalg import cho_factor

from mutascape.calibrated import PilotCalibratedLandscape
from mutascape.domain import MutationDomain
from mutascape.nonlinearity import GlobalEpistasisSigmoid
from scripts.research_pair_posterior import ResearchPairPosterior, pair_kernel
from scripts.test_usefulness import quantile_map


def test_pair_kernel_counts_pairs_and_zeros_singles():
    variants = [{}, {1: "A"}, {1: "A", 2: "C"}, {1: "A", 2: "C", 3: "D"}]
    kernel = pair_kernel(variants, variants)
    np.testing.assert_array_equal(np.diag(kernel), [0, 0, 1, 3])
    np.testing.assert_array_equal(kernel[:2], 0)
    assert kernel[2, 3] == 1


def test_pair_kernel_includes_shared_wild_type_table_entries():
    variants = [{1: "A", 2: "C"}, {1: "A", 2: "D"}, {1: "B", 2: "D"}]
    expected = [[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]]
    np.testing.assert_array_equal(pair_kernel(variants, variants), expected)
    assert np.linalg.eigvalsh(expected).min() > 0


def test_quantile_map_handles_unsorted_scores_and_ties():
    values = quantile_map(np.array([1, 0, -1, 0]), np.arange(4), np.array([2, 4, 6, 8]))
    np.testing.assert_array_equal(values, [8, 5, 2, 5])


def test_posterior_samples_match_gaussian_conditioning():
    substitutions = ((1, "C"), (1, "D"), (2, "C"), (2, "D"))
    model = PilotCalibratedLandscape(
        MutationDomain("AA", substitutions),
        {(): 0.0, **{(k,): 0.0 for k in substitutions}},
        dict.fromkeys(substitutions, 0.0),
        np.zeros(2),
        1.0,
        GlobalEpistasisSigmoid(-2, 2, 1),
        (),
    )
    train = [{1: "C", 2: "C"}, {1: "C", 2: "D"}]
    test = [{1: "D", 2: "C"}, {1: "D", 2: "D"}]
    residual = np.array([0.8, -0.4])
    noise = 0.3
    covariance = pair_kernel(train, train) + noise * np.eye(2)
    posterior = ResearchPairPosterior(
        model, train, residual, cho_factor(covariance, lower=True), noise, noise
    )
    cross = pair_kernel(test, train)
    expected_mean = cross @ np.linalg.solve(covariance, residual)
    expected_covariance = pair_kernel(test, test) - cross @ np.linalg.solve(covariance, cross.T)
    np.testing.assert_allclose(posterior.latent(test, None), expected_mean)
    samples = np.array([posterior.latent(test, seed) for seed in range(1500)])
    np.testing.assert_allclose(samples.mean(axis=0), expected_mean, atol=0.06)
    np.testing.assert_allclose(np.cov(samples.T), expected_covariance, atol=0.08)
