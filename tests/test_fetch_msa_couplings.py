"""Unit tests for the MSA-coupling helper functions in scripts/fetch_msa_couplings.py.

The script lives outside the package, so we add the scripts directory to
``sys.path`` to import the pure functions for testing. The fetch_homologs
and pairwise-aligner paths are NOT tested here (they require UniProt
network access); only the deterministic numerical helpers are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the mutascape scripts/ directory to sys.path so we can import
# fetch_msa_couplings as a normal module.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from fetch_msa_couplings import (  # noqa: E402
    ALPHABET,
    compute_couplings_reweighted,
    hamming_reweighting,
    zero_sum_gauge,
)

K = len(ALPHABET)  # 20


class TestHammingReweighting:
    """Per-sequence weights to downweight phylogenetic redundancy.

    Each sequence's weight is 1 / cluster_size where cluster_size counts
    sequences within (1 - identity_threshold) Hamming distance INCLUDING
    self. So a singleton cluster (no near-neighbours) has weight 1.0, a
    cluster of 5 identical sequences has weight 1/5 each.
    """

    def test_all_identical_sequences_form_one_cluster(self):
        # 5 identical sequences, length 10
        idx = np.zeros((5, 10), dtype=np.int64)
        weights = hamming_reweighting(idx, identity_threshold=0.8)
        np.testing.assert_allclose(weights, np.full(5, 1 / 5))

    def test_all_distinct_sequences_form_singletons(self):
        # 4 sequences, each maximally distant from the others (length 10)
        # Make each sequence have a different AA at every position
        idx = np.array(
            [
                [0] * 10,
                [1] * 10,
                [2] * 10,
                [3] * 10,
            ],
            dtype=np.int64,
        )
        weights = hamming_reweighting(idx, identity_threshold=0.8)
        np.testing.assert_allclose(weights, np.ones(4))

    def test_two_clusters(self):
        # 5 near-identical sequences + 3 distinct sequences
        # The 5 share AA 0 everywhere; the 3 use AAs 1, 2, 3
        idx = np.array(
            [
                [0] * 10,  # cluster A
                [0] * 10,
                [0] * 10,
                [0] * 10,
                [0] * 10,
                [1] * 10,  # singletons
                [2] * 10,
                [3] * 10,
            ],
            dtype=np.int64,
        )
        weights = hamming_reweighting(idx, identity_threshold=0.8)
        np.testing.assert_allclose(weights[:5], np.full(5, 1 / 5))
        np.testing.assert_allclose(weights[5:], np.ones(3))

    def test_identity_threshold_lowered_groups_more(self):
        # Two sequences differing at 3/10 positions = 70% identity
        # At identity_threshold=0.8 they should NOT cluster (require >80% identity)
        # At identity_threshold=0.6 they SHOULD cluster (require >60% identity)
        idx = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # 70% identical
            ],
            dtype=np.int64,
        )
        w_strict = hamming_reweighting(idx, identity_threshold=0.8)
        np.testing.assert_allclose(w_strict, np.ones(2))  # singletons

        w_loose = hamming_reweighting(idx, identity_threshold=0.6)
        np.testing.assert_allclose(w_loose, np.full(2, 0.5))  # one cluster of 2


class TestZeroSumGauge:
    """Projects pairwise coupling tensor to the Ising (zero-sum) gauge.

    Required property: after gauge fixing, for every (i, j) pair the
    20x20 sub-matrix has row sums and column sums approximately zero.
    """

    @pytest.fixture
    def J(self):
        rng = np.random.default_rng(123)
        return rng.normal(0.0, 0.5, (4, 4, K, K))

    def test_row_sums_zero_after_gauge(self, J):
        J_gauge = zero_sum_gauge(J)
        # sum over axis -2 (the "a" index) should be ~0 for every (i, j, b)
        row_sums = J_gauge.sum(axis=-2)
        assert np.max(np.abs(row_sums)) < 1e-12

    def test_col_sums_zero_after_gauge(self, J):
        J_gauge = zero_sum_gauge(J)
        col_sums = J_gauge.sum(axis=-1)
        assert np.max(np.abs(col_sums)) < 1e-12

    def test_idempotent(self, J):
        J1 = zero_sum_gauge(J)
        J2 = zero_sum_gauge(J1)
        np.testing.assert_allclose(J1, J2, atol=1e-14)

    def test_shape_preserved(self, J):
        J_gauge = zero_sum_gauge(J)
        assert J_gauge.shape == J.shape

    def test_grand_mean_correction(self):
        """gauge(J)[a,b] = J[a,b] - row_mean[b] - col_mean[a] + grand_mean."""
        rng = np.random.default_rng(7)
        sub = rng.normal(0.0, 1.0, (K, K))
        # Wrap in (1, 1, K, K) so zero_sum_gauge sees pair axes
        J = sub[None, None]
        J_gauge = zero_sum_gauge(J)
        # Each (a, b) entry should equal the expected formula
        row_mean = sub.mean(axis=0, keepdims=True)
        col_mean = sub.mean(axis=1, keepdims=True)
        grand = sub.mean()
        expected = sub - row_mean - col_mean + grand
        np.testing.assert_allclose(J_gauge[0, 0], expected, atol=1e-12)


class TestComputeCouplingsReweighted:
    """End-to-end coupling computation: MSA matrix → weighted log-odds + zero-sum gauge."""

    @pytest.fixture
    def msa_idx(self):
        """20 sequences over 3 positions. Half are clustered (same AA at every
        position), the other half are random. Just enough structure that the
        log-odds are non-trivial."""
        rng = np.random.default_rng(0)
        idx = np.zeros((20, 3), dtype=np.int64)
        idx[:10] = 0  # cluster
        idx[10:] = rng.integers(0, K, size=(10, 3))
        return idx

    def test_output_shape(self, msa_idx):
        J, n_eff = compute_couplings_reweighted(msa_idx)
        assert J.shape == (3, 3, K, K)
        assert isinstance(n_eff, float)
        assert n_eff > 0
        assert n_eff <= 20  # at most N

    def test_symmetry(self, msa_idx):
        J, _ = compute_couplings_reweighted(msa_idx)
        # J[i, j] should equal J[j, i].T
        for i in range(3):
            for j in range(i + 1, 3):
                np.testing.assert_allclose(J[i, j], J[j, i].T, atol=1e-12)

    def test_zero_sum_after_gauge(self, msa_idx):
        J, _ = compute_couplings_reweighted(msa_idx, apply_gauge=True)
        # Row sums and col sums of each (i,j) block ~ 0
        assert np.max(np.abs(J.sum(axis=-2))) < 1e-10
        assert np.max(np.abs(J.sum(axis=-1))) < 1e-10

    def test_no_gauge_skips_projection(self, msa_idx):
        """When apply_gauge=False the row/col sums are NOT generally zero."""
        J, _ = compute_couplings_reweighted(msa_idx, apply_gauge=False)
        # At least one pair should fail the zero-sum constraint
        assert np.max(np.abs(J.sum(axis=-2))) > 1e-6

    def test_n_eff_decreases_when_clustered(self, msa_idx):
        """A clustered MSA should give n_eff < N because cluster members
        share the same weight."""
        _, n_eff = compute_couplings_reweighted(msa_idx)
        assert n_eff < 20  # N=20; clustered first 10 share weight 1/10

    def test_gap_mask_excludes_position(self):
        """With one sequence gapped at a position, that sequence should
        contribute to the marginals/joints at OTHER positions but not at
        the gapped position."""
        idx_full = np.zeros((10, 3), dtype=np.int64)
        idx_full[5:] = 1  # split into 2 clusters
        gap_mask = np.ones((10, 3), dtype=bool)
        gap_mask[0, 0] = False  # sequence 0 has a gap at position 0

        J_no_gap, _ = compute_couplings_reweighted(idx_full, gap_mask=None, apply_gauge=False)
        J_gap, _ = compute_couplings_reweighted(idx_full, gap_mask=gap_mask, apply_gauge=False)

        # The (0, 1) and (0, 2) pair couplings should differ because seq 0
        # is excluded from joints involving position 0
        diff_01 = float(np.abs(J_no_gap[0, 1] - J_gap[0, 1]).max())
        assert diff_01 > 1e-3

        # The (1, 2) pair couplings should be identical because seq 0 is
        # still valid at positions 1 and 2
        np.testing.assert_allclose(J_no_gap[1, 2], J_gap[1, 2], atol=1e-12)

    def test_all_valid_gap_mask_matches_none(self, msa_idx):
        gap_mask = np.ones(msa_idx.shape, dtype=bool)
        J_none, n_none = compute_couplings_reweighted(msa_idx, gap_mask=None)
        J_full, n_full = compute_couplings_reweighted(msa_idx, gap_mask=gap_mask)
        np.testing.assert_allclose(J_none, J_full, atol=1e-12)
        assert n_none == pytest.approx(n_full)
