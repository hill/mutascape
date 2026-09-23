"""Test method transfer when the campaign already holds WT and DMS singles."""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks.library import BudgetedOracle, SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features
from scripts.investigate_singles_only import CONFIRMATION, DEVELOPMENT, OUT, SEEDS, data


class DMSOracle(BudgetedOracle):
    """Start with known observations and charge only for new combinations."""

    def __init__(self, values, budget, known):
        super().__init__(values, budget)
        self._known_count = len(known)
        self._rows = list(map(int, known))
        self._observed = [float(values[i]) for i in known]
        self._seen = dict(zip(self._rows, self._observed))

    @property
    def spent(self):
        return len(self._rows) - self._known_count


def run(name, label, seed):
    folder = OUT / name
    destination = folder / f"warm_{label}_{seed}.csv"
    if destination.exists():
        return
    frame, design, _ = data(name)
    values = (
        frame.fitness.to_numpy()
        if label == "measured"
        else np.load(folder / f"fitness_{label}_{seed}.npy")
    )
    rows = warm_rows(frame, design, values, (48, 96))
    rows = [dict(protein=name, model=label, seed=seed, **row) for row in rows]
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"Searched with DMS {name} {label} {seed}", flush=True)


def warm_rows(frame, design, values, budgets):
    """Run every method from a campaign that already holds WT and singles."""
    flat = np.ravel_multi_index(design["codes"].T, design["shape"])
    selected = np.argsort(flat)
    known = np.flatnonzero(frame.depth.to_numpy()[selected] <= 1)
    unknown = np.flatnonzero(frame.depth.to_numpy()[selected] > 1)
    values = np.asarray(values)[selected]
    np.testing.assert_array_equal(values[known], frame.fitness.to_numpy()[selected][known])
    codes = design["codes"][selected]
    incumbent = int(known[np.argmax(values[known])])
    space = SearchSpace(
        codes,
        tuple(design["shape"]),
        selected,
        incumbent,
        {tuple(c): i for i, c in enumerate(codes)},
    )
    features = make_features(space)
    rows = []
    for budget in budgets:
        for method, function in METHODS.items():
            for trial in range(10):
                search_seed = 2026092400 + 137 * trial
                oracle = DMSOracle(values, budget, known)
                if method == "random":
                    rng = np.random.default_rng(search_seed)
                    oracle.query(rng.choice(unknown, size=budget, replace=False))
                else:
                    function(space, oracle, features, search_seed)
                assert oracle.spent <= budget
                rows.append(
                    dict(
                        budget=budget,
                        method=method,
                        trial=trial,
                        spent=oracle.spent,
                        regret=(values.max() - oracle.best()) / max(np.ptp(values), 1e-12),
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    tasks = [(name, "measured", 0) for name in (*DEVELOPMENT, *CONFIRMATION)]
    tasks += [
        (name, label, seed)
        for name in CONFIRMATION
        for label in ("additive", "linear", "sigmoid", "main_singles")
        for seed in (SEEDS[:1] if label == "additive" else SEEDS)
    ]
    Parallel(n_jobs=args.jobs)(delayed(run)(*task) for task in tasks)


if __name__ == "__main__":
    main()
