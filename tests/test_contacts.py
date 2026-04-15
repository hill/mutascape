"""Tests for the contacts module."""

import numpy as np
import pytest

from mutascape.contacts import ContactMap


@pytest.fixture
def simple_distance_matrix():
    """A 5-residue distance matrix with known contact structure.

    Positions 1-5, with contacts:
    - (1,2): 5.0 A  -> contact at 8.0 threshold
    - (1,3): 12.0 A -> no contact
    - (2,3): 7.0 A  -> contact
    - (3,4): 4.0 A  -> contact
    - (4,5): 9.0 A  -> no contact at 8.0
    """
    distances = np.array(
        [
            [0.0, 5.0, 12.0, 15.0, 20.0],
            [5.0, 0.0, 7.0, 10.0, 18.0],
            [12.0, 7.0, 0.0, 4.0, 11.0],
            [15.0, 10.0, 4.0, 0.0, 9.0],
            [20.0, 18.0, 11.0, 9.0, 0.0],
        ]
    )
    positions = [1, 2, 3, 4, 5]
    return distances, positions


@pytest.fixture
def contact_map(simple_distance_matrix):
    distances, positions = simple_distance_matrix
    return ContactMap.from_distance_matrix(distances, positions, threshold=8.0)


class TestContactMapFromDistanceMatrix:
    def test_basic_construction(self, contact_map):
        assert contact_map.n_positions == 5
        assert contact_map.threshold == 8.0

    def test_contact_pairs(self, contact_map):
        pairs = contact_map.contact_pairs()
        assert (1, 2) in pairs
        assert (2, 3) in pairs
        assert (3, 4) in pairs
        assert (1, 3) not in pairs  # 12A > 8A
        assert (4, 5) not in pairs  # 9A > 8A

    def test_n_contacts(self, contact_map):
        assert contact_map.n_contacts == 3  # (1,2), (2,3), (3,4)

    def test_density(self, contact_map):
        # 3 contacts out of 5*4/2 = 10 possible pairs
        assert contact_map.density == pytest.approx(0.3)

    def test_no_self_contacts(self, contact_map):
        for i in range(5):
            assert not contact_map.matrix[i, i]

    def test_symmetry(self, contact_map):
        np.testing.assert_array_equal(contact_map.matrix, contact_map.matrix.T)


class TestContactMapSubmap:
    def test_submap_preserves_contacts(self, contact_map):
        sub = contact_map.submap([1, 2, 3])
        assert sub.n_positions == 3
        pairs = sub.contact_pairs()
        assert (1, 2) in pairs
        assert (2, 3) in pairs

    def test_submap_removes_external_contacts(self, contact_map):
        sub = contact_map.submap([3, 4, 5])
        pairs = sub.contact_pairs()
        assert (3, 4) in pairs
        assert (4, 5) not in pairs  # still not in contact
        assert sub.n_contacts == 1

    def test_submap_invalid_positions(self, contact_map):
        with pytest.raises(ValueError, match="not found"):
            contact_map.submap([1, 2, 99])

    def test_submap_retains_distances(self, contact_map):
        sub = contact_map.submap([1, 2])
        assert sub.distances is not None
        assert sub.distances[0, 1] == pytest.approx(5.0)


class TestContactMapValidation:
    def test_shape_mismatch_raises(self):
        matrix = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="does not match"):
            ContactMap(matrix=matrix, positions=(1, 2), threshold=8.0)

    def test_empty_positions(self):
        matrix = np.zeros((0, 0), dtype=bool)
        cm = ContactMap(matrix=matrix, positions=(), threshold=8.0)
        assert cm.n_positions == 0
        assert cm.n_contacts == 0


class TestContactMapFromDistanceThreshold:
    def test_different_thresholds(self, simple_distance_matrix):
        distances, positions = simple_distance_matrix

        # Very tight threshold - only the closest pair
        tight = ContactMap.from_distance_matrix(distances, positions, threshold=4.5)
        assert tight.n_contacts == 1  # only (3,4) at 4.0A

        # Loose threshold - most pairs in contact
        loose = ContactMap.from_distance_matrix(distances, positions, threshold=15.0)
        assert loose.n_contacts >= 5
