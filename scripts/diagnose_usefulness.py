"""Record search headroom and candidate connectivity for the usefulness study."""

import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mutascape.benchmarks.library import SearchSpace
from scripts import test_usefulness as experiment
from scripts.audit_usefulness_code import OUT


def headroom(values, known, test):
    """Summarise the upper tail relative to the best supplied measurement."""
    best_known = values[known].max()
    upper = values[test]
    return dict(
        combinations=len(test),
        improving_combinations=int((upper > best_known).sum()),
        fraction_above_known=float((upper > best_known).mean()),
        remaining_headroom=float(max(upper.max() - best_known, 0.0)),
        upper_quantile=float(np.quantile(upper, 0.99)),
        upper_spread=float(upper.max() - np.quantile(upper, 0.99)),
        unique_fitness=int(len(np.unique(upper))),
    )


def main():
    experiment.OUT = OUT
    experiment.verify()
    rows, networks = [], []
    for protein in experiment.PROTEINS:
        frame, design, _ = experiment.load(protein)
        known, test = design["known"], design["test"]
        rows.append(
            dict(
                protein=protein,
                model="measured",
                pilot=-1,
                sample=-1,
                **headroom(frame.fitness.to_numpy(), known, test),
            )
        )
        fits = json.loads((OUT / protein / "fits_complete.json").read_text())
        for fit in fits:
            values = np.load(OUT / protein / fit["file"])
            rows.append(dict(**fit, **headroom(values, known, test)))
        selected = np.r_[known, test]
        codes = design["codes"][selected]
        lookup = {tuple(code): index for index, code in enumerate(codes)}
        assert len(lookup) == len(codes)
        space = SearchSpace(codes, tuple(design["shape"]), selected, None, lookup)
        edges = [(row, other) for row in range(space.size) for other in space.neighbours(row)]
        first, second = np.asarray(edges, dtype=int).T
        graph = coo_matrix((np.ones(len(edges)), (first, second)), shape=(space.size, space.size))
        count, _ = connected_components(graph, directed=False)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        networks.append(
            dict(
                protein=protein,
                candidates=space.size,
                connected_components=count,
                isolated_combinations=int((degree[len(known) :] == 0).sum()),
                mean_combination_degree=float(degree[len(known) :].mean()),
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "headroom.csv", index=False)
    pd.DataFrame(networks).to_csv(OUT / "candidate_connectivity.csv", index=False)


if __name__ == "__main__":
    main()
