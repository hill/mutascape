"""Tests for the epistasis module."""

import numpy as np
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    RandomEpistasis,
    SignConsistentEpistasis,
)


@pytest.fixture
def small_contact_map():
    """3-position contact map: (10, 20) in contact, (20, 30) in contact, (10, 30) not."""
    distances = np.array(
        [
            [0.0, 6.0, 15.0],
            [6.0, 0.0, 7.0],
            [15.0, 7.0, 0.0],
        ]
    )
    return ContactMap.from_distance_matrix(distances, [10, 20, 30], threshold=8.0)


@pytest.fixture
def single_site_effects():
    """Simple single-site effects for positions 10, 20, 30."""
    return {
        10: {"A": 0.0, "G": 0.5, "V": -0.3},   # A is wt; G beneficial, V deleterious
        20: {"L": 0.0, "I": 0.2, "F": -0.5},    # L is wt; I beneficial, F deleterious
        30: {"K": 0.0, "R": 0.1, "E": -0.2},    # K is wt; R beneficial, E deleterious
    }


class TestCouplingTensor:
    def test_get_coupling_existing_pair(self, small_contact_map):
        model = RandomEpistasis(sigma=0.5, seed=42)
        tensor = model.generate(small_contact_map)

        # Pairs (10,20) and (20,30) should have couplings
        val = tensor.get_coupling(10, 20, "G", "I")
        assert isinstance(val, float)

    def test_get_coupling_nonexistent_pair(self, small_contact_map):
        model = RandomEpistasis(sigma=0.5, seed=42)
        tensor = model.generate(small_contact_map)

        # Pair (10, 30) is not in contact
        val = tensor.get_coupling(10, 30, "G", "R")
        assert val == 0.0

    def test_n_pairs(self, small_contact_map):
        model = RandomEpistasis(sigma=0.5, seed=42)
        tensor = model.generate(small_contact_map)
        assert tensor.n_pairs == 2  # (10,20) and (20,30)

    def test_get_coupling_order_invariant(self, small_contact_map):
        model = RandomEpistasis(sigma=0.5, seed=42)
        tensor = model.generate(small_contact_map)

        # J(10,20,G,I) should equal J(20,10,I,G) due to symmetry
        val_forward = tensor.get_coupling(10, 20, "G", "I")
        val_reverse = tensor.get_coupling(20, 10, "I", "G")
        assert val_forward == pytest.approx(val_reverse)


class TestRandomEpistasis:
    def test_reproducibility(self, small_contact_map):
        model1 = RandomEpistasis(sigma=0.3, seed=42)
        model2 = RandomEpistasis(sigma=0.3, seed=42)

        t1 = model1.generate(small_contact_map)
        t2 = model2.generate(small_contact_map)

        for key in t1.couplings:
            np.testing.assert_array_equal(t1.couplings[key], t2.couplings[key])

    def test_different_seeds_differ(self, small_contact_map):
        model1 = RandomEpistasis(sigma=0.3, seed=42)
        model2 = RandomEpistasis(sigma=0.3, seed=99)

        t1 = model1.generate(small_contact_map)
        t2 = model2.generate(small_contact_map)

        # At least one coupling matrix should differ
        any_diff = False
        for key in t1.couplings:
            if not np.allclose(t1.couplings[key], t2.couplings[key]):
                any_diff = True
                break
        assert any_diff

    def test_coupling_matrix_symmetric(self, small_contact_map):
        model = RandomEpistasis(sigma=0.5, seed=42)
        tensor = model.generate(small_contact_map)

        for matrix in tensor.couplings.values():
            np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_coupling_scale(self, small_contact_map):
        """Couplings should have std approximately equal to sigma."""
        model = RandomEpistasis(sigma=0.5, seed=42)
        tensor = model.generate(small_contact_map)

        all_values = np.concatenate([m.ravel() for m in tensor.couplings.values()])
        # Due to symmetrisation, actual std will be sigma/sqrt(2) for off-diag
        # Just check it's in the right ballpark
        assert 0.1 < np.std(all_values) < 1.0

    def test_zero_sigma_gives_zero_couplings(self, small_contact_map):
        model = RandomEpistasis(sigma=0.0, seed=42)
        tensor = model.generate(small_contact_map)

        for matrix in tensor.couplings.values():
            np.testing.assert_array_equal(matrix, np.zeros_like(matrix))


class TestSignConsistentEpistasis:
    def test_requires_single_site_effects(self, small_contact_map):
        model = SignConsistentEpistasis(sigma=0.3, seed=42)
        with pytest.raises(ValueError, match="requires single_site_effects"):
            model.generate(small_contact_map, single_site_effects=None)

    def test_sign_consistency(self, small_contact_map, single_site_effects):
        """Both-beneficial pair should have positive coupling."""
        model = SignConsistentEpistasis(sigma=0.3, seed=42)
        tensor = model.generate(small_contact_map, single_site_effects)

        # G at pos 10 is beneficial (+0.5), I at pos 20 is beneficial (+0.2)
        # Their coupling should be positive (cooperative)
        j_beneficial = tensor.get_coupling(10, 20, "G", "I")
        assert j_beneficial >= 0

        # V at pos 10 is deleterious (-0.3), F at pos 20 is deleterious (-0.5)
        # Both negative * negative = positive sign -> positive coupling
        j_both_del = tensor.get_coupling(10, 20, "V", "F")
        assert j_both_del >= 0

        # G at pos 10 is beneficial (+0.5), F at pos 20 is deleterious (-0.5)
        # Positive * negative = negative -> antagonistic coupling
        j_mixed = tensor.get_coupling(10, 20, "G", "F")
        assert j_mixed <= 0

    def test_wild_type_coupling_is_zero(self, small_contact_map, single_site_effects):
        """Wild-type amino acids (effect = 0) should have zero coupling."""
        model = SignConsistentEpistasis(sigma=0.3, seed=42)
        tensor = model.generate(small_contact_map, single_site_effects)

        # A is wt at pos 10, so coupling with anything should be 0
        j_wt = tensor.get_coupling(10, 20, "A", "I")
        assert j_wt == pytest.approx(0.0)

    def test_reproducibility(self, small_contact_map, single_site_effects):
        model1 = SignConsistentEpistasis(sigma=0.3, seed=42)
        model2 = SignConsistentEpistasis(sigma=0.3, seed=42)

        t1 = model1.generate(small_contact_map, single_site_effects)
        t2 = model2.generate(small_contact_map, single_site_effects)

        for key in t1.couplings:
            np.testing.assert_array_equal(t1.couplings[key], t2.couplings[key])
