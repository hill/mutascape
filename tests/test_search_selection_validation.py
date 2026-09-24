"""Check input handling for the frozen search-selection evaluation."""

import numpy as np
import pandas as pd

from scripts.search_selection_validation import parse_substitutions, retain_covered_positions
from scripts.singles_warm_start import DMSOracle


def test_archive_coordinates_and_variant_filters():
    assert parse_substitutions("A126G,D128V", "ACD", 126) == {1: "G", 3: "V"}
    assert parse_substitutions("A0G,D2V", "ACD", 0) == {1: "G", 3: "V"}
    assert parse_substitutions("C0G", "ACD", 0) is None
    assert parse_substitutions("A0*", "ACD", 0) is None
    assert parse_substitutions("A0G,A0V", "ACD", 0) is None
    assert parse_substitutions("A3G", "ACD", 0) is None


def test_position_selection_ignores_fitness_and_unsupported_combinations():
    mutations = [{1: "G"}, {2: "G"}, {3: "G"}, {1: "G", 2: "G"}, {1: "G", 3: "G"}, {2: "A", 3: "G"}]
    frame = pd.DataFrame(
        dict(mutations=mutations, depth=list(map(len, mutations)), fitness=[1, 2, 3, 4, 5, 999])
    )
    first, positions, count = retain_covered_positions(frame, limit=2)
    changed = frame.assign(fitness=-frame.fitness)
    second, other_positions, other_count = retain_covered_positions(changed, limit=2)
    assert positions == other_positions == [1, 2]
    assert count == other_count == 5
    assert first.mutations.to_list() == second.mutations.to_list()
    assert first.depth.to_list() == [1, 1, 2]


def test_known_singles_do_not_spend_combination_budget():
    oracle = DMSOracle(np.array([0.0, 1.0, 0.5, 2.0, 3.0]), 2, [0, 1, 2])
    oracle.query([0, 3, 1, 4])
    assert oracle.spent == 2
    assert oracle.best() == 3.0


def test_lexicographic_order_matches_flat_indices_without_cartesian_allocation():
    rng = np.random.default_rng(41)
    codes = np.array(list(np.ndindex(3, 4, 2)))
    rng.shuffle(codes)
    np.testing.assert_array_equal(
        np.lexsort(codes.T[::-1]), np.argsort(np.ravel_multi_index(codes.T, (3, 4, 2)))
    )
    large = np.vstack([np.ones(16, dtype=int), np.zeros(16, dtype=int)])
    np.testing.assert_array_equal(np.lexsort(large.T[::-1]), [1, 0])
