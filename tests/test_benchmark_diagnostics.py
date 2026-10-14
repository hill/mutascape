"""Checks for degenerate search domains and known-observation limits."""

import numpy as np

from mutascape.benchmarks.diagnostics import fitness_headroom, library_connectivity
from mutascape.benchmarks.library import SearchSpace
from scripts.benchmark_efficacy import connected_rows


def test_connectivity_reports_isolated_candidates():
    codes = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=np.int16)
    space = SearchSpace(
        codes, (2, 2, 2), np.arange(3), 0, {tuple(c): i for i, c in enumerate(codes)}
    )
    result = library_connectivity(space)
    assert result["connected_components"] == 2
    assert result["isolated_candidates"] == 1


def test_headroom_detects_when_known_measurements_already_attain_the_optimum():
    result = fitness_headroom(np.array([2.0, 1.0, 0.0, 2.0]), np.array([0, 1]))
    assert not result["can_improve"]
    assert result["headroom"] == 0
    assert result["improving_candidates"] == 0


def test_connected_domain_contains_known_rows_and_retains_paths():
    codes = ((np.arange(128)[:, None] >> np.arange(7)) & 1).astype(np.int16)
    known = np.flatnonzero(codes.sum(axis=1) <= 1)
    rows = connected_rows(codes, (2,) * 7, known, 40, 73)
    assert set(known) <= set(rows)
    selected = codes[rows]
    space = SearchSpace(selected, (2,) * 7, rows, 0, {tuple(c): i for i, c in enumerate(selected)})
    assert library_connectivity(space)["connected_components"] == 1
    np.testing.assert_array_equal(rows, connected_rows(codes, (2,) * 7, known, 40, 73))
