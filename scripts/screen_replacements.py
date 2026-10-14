"""Screen candidate evaluation proteins for experimental search headroom.

A useful evaluation protein needs search methods to differ on its experimental
fitness. Each campaign starts from WT, singles and 96 calibration genotypes,
as in the manuscript search, so the screen asks how often a method beats that
start and how far apart the methods finish. Floor counts flag landscapes whose
single-site measurements carry no information.

    uv run python -m scripts.screen_replacements --jobs 9
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from mutascape.benchmarks.datasets import COMBINGYM_MANIFEST, _graphfla_spec, load_landscape
from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features
from scripts.manuscript_v2 import COMBINGYM, GRAPHFLA, KnownOracle, supported

OUT = Path("results/manuscript_v2_20261015/replacement_screen.csv")
CANDIDATES = {
    "PhoQ (control)": ("graphfla", "Podgornaia2015_PhoQ.csv", "PhoQ", "AVST"),
    "CR9114-h1": ("graphfla", "Phillips2021_CR9114_h1.csv", "CR9114", "auto"),
    "CR9114-h3": ("graphfla", "Phillips2021_CR9114_h3.csv", "CR9114", "auto"),
    "CR9114-fluB": ("graphfla", "Phillips2021_CR9114_fluB.csv", "CR9114", "auto"),
    "CR6261-h1": ("graphfla", "Phillips2021_CR6261_h1.csv", "CR6261", "auto"),
    "H1-MA90": ("graphfla", "Phillips2023_MA90.csv", "H1", "auto"),
    "H1-G189E": ("graphfla", "Phillips2023_G189E.csv", "H1", "auto"),
    "H1-SI06": ("graphfla", "Phillips2023_SI06.csv", "H1", "auto"),
    "RBD-ACE2": ("graphfla", "Moulana2022_ACE2.csv", "RBD", "auto"),
    "RBD-S309": ("graphfla", "Moulana2023_S309.csv", "RBD", "auto"),
    "RBD-CB6": ("graphfla", "Moulana2023_CB6.csv", "RBD", "auto"),
    "RBD-CoV555": ("graphfla", "Moulana2023_CoV555.csv", "RBD", "auto"),
    "RBD-REGN10987": ("graphfla", "Moulana2023_REGN10987.csv", "RBD", "auto"),
    "COV107-23": ("graphfla", "Schulz2025.csv", "COV107-23", "auto"),
    "Hsp90": ("graphfla", "Bank2016a.csv", "Hsp90", "QFGFSAAAE"),
    "RhlA": ("combingym", "RhlA_RLs", "RhlA", None),
    "SaCas9": ("combingym", "SaCas9_activity", "SaCas9", None),
    "HIV-1": ("combingym", "HIV1_activity", "HIV-1", None),
}
TRIALS = 5
SETS = 3


def reference(filename: str) -> str:
    """All-ancestral genotype of a GraphFLA presence-absence table."""
    columns = pd.read_csv(GRAPHFLA / "data/BioSequence" / filename, nrows=0).columns
    return "0" * sum(c.startswith("pos") and c[3:].isdigit() for c in columns)


def load(name: str):
    kind, key, protein, ref = CANDIDATES[name]
    if kind == "graphfla":
        if ref == "auto":
            ref = reference(key)
        root, spec = GRAPHFLA, _graphfla_spec(key, protein, ref)
    else:
        root, spec = COMBINGYM, next(s for s in COMBINGYM_MANIFEST if s.name == key)
    dataset = load_landscape(root, spec, require_complete=False)
    frame = dataset.variants.reset_index(drop=True)[["mutations", "depth", "fitness"]]
    return supported(frame, dataset.wild_type_sequence)


def screen(name: str) -> dict:
    warnings.filterwarnings("ignore")
    frame, codes, shape = load(name)
    f = frame.fitness.to_numpy()
    depth = frame.depth.to_numpy()
    multi = np.flatnonzero(depth > 1)
    lookup = {tuple(int(v) for v in code): i for i, code in enumerate(codes)}
    improved = {48: [], 96: []}
    spread = {48: [], 96: []}
    for c in range(SETS):
        rng = np.random.default_rng([20261016, c])
        calibration = rng.choice(multi, min(96, len(multi) // 4), replace=False)
        known = np.sort(np.r_[np.flatnonzero(depth <= 1), calibration])
        start = f[known].max()
        space = SearchSpace(
            codes, shape, np.arange(len(codes)), int(known[np.argmax(f[known])]), lookup
        )
        features = make_features(space)
        for budget in (48, 96):
            means = []
            for method, function in METHODS.items():
                best = []
                for trial in range(TRIALS):
                    oracle = KnownOracle(f, budget, known)
                    function(space, oracle, features, 1000 * c + trial)
                    best.append(oracle.best())
                    improved[budget].append(oracle.best() > start + 1e-12)
                means.append(np.mean(best))
            spread[budget].append((max(means) - min(means)) / np.ptp(f))
    return dict(
        candidate=name,
        sites=len(shape),
        genotypes=len(frame),
        singles=int((depth == 1).sum()),
        # A floored landscape with floored singles gives Mutascape no single-site signal.
        at_floor=float(np.mean(np.isclose(f, f.min()))),
        singles_at_floor=int(np.isclose(f[depth == 1], f.min()).sum()),
        improved_48=np.mean(improved[48]),
        improved_96=np.mean(improved[96]),
        method_spread_48=np.mean(spread[48]),
        method_spread_96=np.mean(spread[96]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=9)
    args = parser.parse_args()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=args.jobs)(delayed(screen)(name) for name in CANDIDATES)
    table = pd.DataFrame(rows).sort_values("improved_48", ascending=False)
    table.to_csv(OUT, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
