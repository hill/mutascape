"""Checks for whether a declared library can support a search comparison."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mutascape.benchmarks.library import SearchSpace


def library_connectivity(space: SearchSpace) -> dict[str, int | float]:
    """Count components and isolated candidates in the available mutation graph."""
    edges = [(row, other) for row in range(space.size) for other in space.neighbours(row)]
    if edges:
        first, second = np.asarray(edges, dtype=int).T
    else:
        first = second = np.empty(0, dtype=int)
    graph = coo_matrix((np.ones(len(first)), (first, second)), shape=(space.size, space.size))
    count, _ = connected_components(graph, directed=False)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    return {
        "candidates": space.size,
        "connected_components": int(count),
        "isolated_candidates": int((degree == 0).sum()),
        "mean_degree": float(degree.mean()) if space.size else float("nan"),
    }


def fitness_headroom(
    values: NDArray[np.float64], known: NDArray[np.int64]
) -> dict[str, int | float | bool]:
    """Measure available improvement beyond the best supplied observation."""
    values = np.asarray(values, dtype=float)
    known = np.asarray(known, dtype=int)
    if values.ndim != 1 or not np.isfinite(values).all() or not len(known):
        raise ValueError("supply finite fitness values and at least one known observation")
    if known.ndim != 1 or (known < 0).any() or (known >= len(values)).any():
        raise ValueError("known rows must index the fitness array")
    unknown = np.setdiff1d(np.arange(len(values)), known)
    best_known = float(values[known].max())
    headroom = float(values.max() - best_known)
    return {
        "known": int(len(np.unique(known))),
        "unknown": len(unknown),
        "best_known": best_known,
        "headroom": headroom,
        "improving_candidates": int((values[unknown] > best_known).sum()),
        "improving_fraction": float((values[unknown] > best_known).mean())
        if len(unknown)
        else float("nan"),
        "fitness_range": float(np.ptp(values)),
        "can_improve": bool(headroom > 0),
    }
