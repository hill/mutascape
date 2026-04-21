"""Tests for the epistasis module."""

import numpy as np
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    AMINO_ACIDS,
    ContactNKEpistasis,
    ContactNKTensor,
    MSACouplingsEpistasis,
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
    """Simple single-site effects for positions 10, 20, 30.

    Magnitudes are chosen so no two-mutant linear sum is exactly zero, so the
    SignConsistentEpistasis tie-at-zero case (sign(sum) = 0) never fires here.
    """
    return {
        10: {"A": 0.0, "G": 0.6, "V": -0.3},  # A is wt; G beneficial, V deleterious
        20: {"L": 0.0, "I": 0.2, "F": -0.5},  # L is wt; I beneficial, F deleterious
        30: {"K": 0.0, "R": 0.1, "E": -0.2},  # K is wt; R beneficial, E deleterious
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

    def test_sign_rule_follows_sum(self, small_contact_map, single_site_effects):
        """Coupling sign tracks sign(h_a + h_b), amplifying the linear sum."""
        model = SignConsistentEpistasis(sigma=0.3, seed=42)
        tensor = model.generate(small_contact_map, single_site_effects)

        # Both beneficial (h_G=+0.6, h_I=+0.2, sum=+0.8) -> J > 0, reinforces benefit
        assert tensor.get_coupling(10, 20, "G", "I") > 0

        # Both deleterious (h_V=-0.3, h_F=-0.5, sum=-0.8) -> J < 0, reinforces damage
        assert tensor.get_coupling(10, 20, "V", "F") < 0

        # Mixed, net-beneficial (h_G=+0.6, h_F=-0.5, sum=+0.1) -> J > 0
        assert tensor.get_coupling(10, 20, "G", "F") > 0

        # Mixed, net-deleterious (h_V=-0.3, h_I=+0.2, sum=-0.1) -> J < 0
        assert tensor.get_coupling(10, 20, "V", "I") < 0

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


class TestContactNKEpistasis:
    def test_returns_tensor_with_expected_shapes(self, small_contact_map):
        model = ContactNKEpistasis(sigma=0.3, seed=0)
        tensor = model.generate(small_contact_map)
        assert isinstance(tensor, ContactNKTensor)
        assert set(tensor.positions) == {10, 20, 30}

        # Position 10 contacts {20}; position 20 contacts {10, 30};
        # position 30 contacts {20}. Per-position tensor shape is
        # (20,) * (1 + n_neighbours).
        assert tensor.per_position_functions[10].shape == (20, 20)
        assert tensor.per_position_functions[20].shape == (20, 20, 20)
        assert tensor.per_position_functions[30].shape == (20, 20)

    def test_reproducibility(self, small_contact_map):
        m1 = ContactNKEpistasis(sigma=0.3, seed=42)
        m2 = ContactNKEpistasis(sigma=0.3, seed=42)
        t1 = m1.generate(small_contact_map)
        t2 = m2.generate(small_contact_map)
        for pos in t1.per_position_functions:
            np.testing.assert_array_equal(
                t1.per_position_functions[pos],
                t2.per_position_functions[pos],
            )

    def test_max_neighbours_caps_tensor_order(self, small_contact_map):
        # Position 20 has 2 neighbours; capping at 1 drops one.
        model = ContactNKEpistasis(sigma=0.3, max_neighbours=1, seed=0)
        tensor = model.generate(small_contact_map)
        assert tensor.per_position_functions[20].shape == (20, 20)
        assert len(tensor.neighbours_by_position[20]) == 1

    def test_zero_sigma_gives_zero_functions(self, small_contact_map):
        model = ContactNKEpistasis(sigma=0.0, seed=0)
        tensor = model.generate(small_contact_map)
        for f in tensor.per_position_functions.values():
            np.testing.assert_array_equal(f, np.zeros_like(f))

    def test_different_seeds_differ(self, small_contact_map):
        t1 = ContactNKEpistasis(sigma=0.3, seed=1).generate(small_contact_map)
        t2 = ContactNKEpistasis(sigma=0.3, seed=2).generate(small_contact_map)
        assert not np.array_equal(
            t1.per_position_functions[10], t2.per_position_functions[10]
        )

    def test_aa_index_matches_standard(self, small_contact_map):
        tensor = ContactNKEpistasis(sigma=0.3, seed=0).generate(small_contact_map)
        assert tensor.aa_index["A"] == 0
        assert tensor.aa_index["Y"] == len(AMINO_ACIDS) - 1


class TestMSACouplingsEpistasis:
    def _mock_matrix(self, L=3, seed=0):
        rng = np.random.default_rng(seed)
        J = rng.normal(0, 1.0, size=(L, L, 20, 20))
        # Enforce symmetry on position axes for a well-formed Potts matrix.
        for i in range(L):
            for j in range(L):
                if i > j:
                    J[i, j] = J[j, i].T
                if i == j:
                    J[i, j] = 0.0
        return J

    def test_shape_check_raises(self, small_contact_map):
        with pytest.raises(ValueError, match="shape"):
            MSACouplingsEpistasis(
                coupling_matrix=np.zeros((3, 3, 20, 19)),
                position_order=(10, 20, 30),
            )

    def test_generate_restricts_to_contacts(self, small_contact_map):
        J = self._mock_matrix()
        model = MSACouplingsEpistasis(
            coupling_matrix=J, position_order=(10, 20, 30),
        )
        tensor = model.generate(small_contact_map)
        # (10, 30) is NOT in contact in the fixture; should be absent.
        assert set(tensor.couplings.keys()) == {(10, 20), (20, 30)}

    def test_generate_without_contact_restriction(self, small_contact_map):
        J = self._mock_matrix()
        model = MSACouplingsEpistasis(
            coupling_matrix=J,
            position_order=(10, 20, 30),
            restrict_to_contacts=False,
        )
        tensor = model.generate(small_contact_map)
        assert set(tensor.couplings.keys()) == {(10, 20), (10, 30), (20, 30)}

    def test_scale_multiplies_couplings(self, small_contact_map):
        J = self._mock_matrix()
        base = MSACouplingsEpistasis(
            coupling_matrix=J, position_order=(10, 20, 30),
        ).generate(small_contact_map)
        scaled = MSACouplingsEpistasis(
            coupling_matrix=J, position_order=(10, 20, 30), scale=2.0,
        ).generate(small_contact_map)
        for key in base.couplings:
            np.testing.assert_allclose(
                scaled.couplings[key], base.couplings[key] * 2.0, atol=1e-12,
            )

    def test_missing_position_raises(self, small_contact_map):
        J = np.zeros((2, 2, 20, 20))
        model = MSACouplingsEpistasis(
            coupling_matrix=J, position_order=(10, 20),
        )
        with pytest.raises(ValueError, match="missing"):
            model.generate(small_contact_map)

    def test_alphabet_reordering(self, small_contact_map):
        # Construct a matrix in REVERSED alphabet order, assert that the
        # generated couplings match the canonical-order version.
        J_canonical = self._mock_matrix()
        canonical_alphabet = "".join(AMINO_ACIDS)
        reversed_alphabet = canonical_alphabet[::-1]

        reorder = np.array([canonical_alphabet.index(a) for a in reversed_alphabet])
        J_reversed = J_canonical[:, :, reorder][:, :, :, reorder]

        canon = MSACouplingsEpistasis(
            coupling_matrix=J_canonical,
            position_order=(10, 20, 30),
            alphabet=canonical_alphabet,
        ).generate(small_contact_map)
        rev = MSACouplingsEpistasis(
            coupling_matrix=J_reversed,
            position_order=(10, 20, 30),
            alphabet=reversed_alphabet,
        ).generate(small_contact_map)
        for key in canon.couplings:
            np.testing.assert_allclose(
                canon.couplings[key], rev.couplings[key], atol=1e-12,
            )

    def test_from_npz_roundtrip(self, tmp_path, small_contact_map):
        J = self._mock_matrix()
        positions = np.array([10, 20, 30])
        path = tmp_path / "mock.npz"
        np.savez(path, coupling_matrix=J, positions=positions)

        loaded = MSACouplingsEpistasis.from_npz(path)
        tensor = loaded.generate(small_contact_map)
        assert tensor.n_pairs == 2  # (10,20) and (20,30)

    def test_from_msa_fasta_produces_valid_couplings(self, tmp_path):
        # Build a small synthetic MSA where positions 0 and 1 co-vary perfectly
        # and position 2 is independent. Log-odds coupling at (0,1) should be
        # much larger in magnitude than (0,2) or (1,2).
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        seqs = []
        rng = np.random.default_rng(0)
        for i in range(500):
            pair = rng.choice(["AA", "GG", "LL", "KK"])
            ind = rng.choice(list("ACDEFGHIKLMNPQRSTVWY"))
            seqs.append(SeqRecord(Seq(pair + ind), id=f"s{i}"))

        path = tmp_path / "msa.fasta"
        SeqIO.write(seqs, str(path), "fasta")

        model = MSACouplingsEpistasis.from_msa_fasta(
            path, positions=(0, 1, 2), restrict_to_contacts=False,
        )
        # Pairs (0,1), (0,2), (1,2) relative magnitudes.
        abs_01 = np.abs(model.coupling_matrix[0, 1]).mean()
        abs_02 = np.abs(model.coupling_matrix[0, 2]).mean()
        abs_12 = np.abs(model.coupling_matrix[1, 2]).mean()
        assert abs_01 > 2.0 * abs_02
        assert abs_01 > 2.0 * abs_12
