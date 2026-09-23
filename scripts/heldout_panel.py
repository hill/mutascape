"""Freeze a held-out protein panel before any generator scoring runs on it."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.datasets import (
    COMBINGYM_MANIFEST,
    SSMULA_MANIFEST,
    _graphfla_spec,
    load_landscape,
)
from scripts.investigate_singles_only import digest, sample_cubes

OUT = Path("results/heldout_panel_20260923")
SSMULA = Path(".mutascape/benchmark-data/ssmula")
COMBINGYM = Path(".mutascape/benchmark-data/combingym")
GRAPHFLA = Path(".mutascape/worktrees/posterior-landscapes-v1/.mutascape/benchmark-data/graphfla")
MINIMUM_COMBINATIONS = 1000
TEST_SIZE = 1000
CUBES = 256
SEED_BASE = 2026092400
SEARCH_SEEDS = (2026092411, 2026092412, 2026092413)
BUDGETS = (48, 96, 192)

# Families already used to develop or confirm the generator. Reusing any of them
# would let a choice made on development data score itself.
USED_FAMILIES = ("DHFR", "eqFP611", "ParB", "ParD", "CreiLOV", "GB1", "TrpB", "T7", "TEV")

# Ordered within each family, most measured combinations first. The first
# candidate of a family that passes the other rules is the one kept.
CANDIDATES = (
    ("PhoQ", "graphfla", "Podgornaia2015_PhoQ.csv", "AVST"),
    ("CR9114", "graphfla", "Phillips2021_CR9114_h1.csv", "0" * 16),
    ("CR9114", "graphfla", "Phillips2021_CR9114_h3.csv", "0" * 16),
    ("CR9114", "combingym", "CR9114_H1_binding", None),
    ("SARS-CoV-2-RBD", "graphfla", "Moulana2022_ACE2.csv", "0" * 15),
    ("SARS-CoV-2-RBD", "graphfla", "Moulana2023_CB6.csv", "0" * 15),
    ("H1-haemagglutinin", "graphfla", "Phillips2023_SI06.csv", "0" * 16),
    ("RhlA", "combingym", "RhlA_RLs", None),
    ("CR6261", "graphfla", "Phillips2021_CR6261_h1.csv", "0" * 11),
    ("CR6261", "combingym", "CR6261_H1_binding", None),
    ("SaCas9", "combingym", "SaCas9_activity", None),
    ("HIV-1", "combingym", "HIV1_activity", None),
    ("SpCas9", "combingym", "SpCas9_activity", None),
    ("ParB", "graphfla", "Jalal2020_parS.csv", "RTAG"),
    ("ParD", "ssmula", "ParD3", None),
    ("TrpB", "ssmula", "TrpB3A", None),
)


def spec_for(family, source, key, reference):
    if source == "graphfla":
        return GRAPHFLA, _graphfla_spec(key, family, reference)
    if source == "combingym":
        return COMBINGYM, next(s for s in COMBINGYM_MANIFEST if s.name == key)
    return SSMULA, next(s for s in SSMULA_MANIFEST if s.name == key)


def load(family, source, key, reference):
    """Return the measured library restricted to substitutions with a measured single."""
    root, spec = spec_for(family, source, key, reference)
    dataset = load_landscape(root, spec, require_complete=False)
    sequence = dataset.wild_type_sequence
    frame = dataset.variants.reset_index(drop=True)
    supported = {item for m in frame.loc[frame.depth == 1].mutations for item in m.items()}
    frame = frame.loc[frame.mutations.map(lambda m: set(m.items()) <= supported)]
    frame = frame.reset_index(drop=True)
    positions = sorted({p for p, _ in supported})
    alphabets = {
        p: sorted({sequence[p - 1]} | {a for q, a in supported if p == q}) for p in positions
    }
    codes = np.array(
        [
            [alphabets[p].index(m.get(p, sequence[p - 1])) for p in positions]
            for m in frame.mutations
        ],
        dtype=np.int16,
    )
    shape = tuple(len(alphabets[p]) for p in positions)
    return frame, sequence, codes, shape, root / spec.relative_path, spec


def screen(seed):
    """Apply the exclusion rules in order and return every candidate's verdict."""
    verdicts, kept_families = [], set()
    for number, (family, source, key, reference) in enumerate(CANDIDATES):
        record = dict(family=family, source=source, dataset=key)
        frame, sequence, codes, shape, path, _ = load(family, source, key, reference)
        singles = frame.loc[frame.depth == 1]
        combinations = int((frame.depth > 1).sum())
        record.update(
            combinations=combinations,
            sites=len(shape),
            singles=len(singles),
            singles_sd=round(float(singles.fitness.std()), 6),
        )
        if family in USED_FAMILIES:
            record["excluded"] = "family already used in development or confirmation"
        elif combinations < MINIMUM_COMBINATIONS:
            record["excluded"] = f"fewer than {MINIMUM_COMBINATIONS} measured combinations"
        elif record["singles_sd"] < 1e-6:
            record["excluded"] = "single-mutant fitness carries no variance"
        elif family in kept_families:
            record["excluded"] = "another assay of this family is already in the panel"
        else:
            try:
                cubes = sample_cubes(codes, shape, seed + number)
            except ValueError as error:
                record["excluded"] = str(error)
            else:
                record["cubes"] = len(cubes)
                kept_families.add(family)
        verdicts.append(record)
    return verdicts


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    verdicts = screen(SEED_BASE)
    accepted = [v for v in verdicts if "excluded" not in v]
    inventory = []
    for number, (family, source, key, reference) in enumerate(CANDIDATES):
        if not any(v["dataset"] == key for v in accepted):
            continue
        frame, sequence, codes, shape, path, spec = load(family, source, key, reference)
        folder = OUT / family
        folder.mkdir(exist_ok=True)
        combinations = np.flatnonzero(frame.depth > 1)
        test = np.random.default_rng(SEED_BASE + number).choice(
            combinations, min(TEST_SIZE, len(combinations)), replace=False
        )
        cubes = sample_cubes(codes, shape, SEED_BASE + number)
        frame.to_pickle(folder / "frame.pkl")
        np.savez(folder / "design.npz", codes=codes, shape=shape, test=test, cubes=cubes)
        (folder / "metadata.json").write_text(
            json.dumps(
                dict(
                    family=family,
                    dataset=key,
                    source=spec.source,
                    source_url=spec.source_url,
                    source_revision=spec.source_revision,
                    path=str(path),
                    path_sha256=digest(path),
                    sequence=sequence,
                ),
                indent=2,
            )
            + "\n"
        )
        inventory.append(
            dict(
                family=family,
                dataset=key,
                measured=len(frame),
                combinations=int((frame.depth > 1).sum()),
                singles=int((frame.depth == 1).sum()),
                sites=len(shape),
                declared=int(np.prod(shape, dtype=float)),
                cubes=len(cubes),
            )
        )
    panel = dict(
        purpose=(
            "Held-out panel for the preregistered test of whether Mutascape transfers "
            "method rankings from DMS singles alone"
        ),
        frozen_before_scoring=True,
        panel=[record["family"] for record in accepted],
        rules=dict(
            order="Candidates are screened in the listed order, one rule at a time",
            used_families=USED_FAMILIES,
            minimum_combinations=MINIMUM_COMBINATIONS,
            singles_variance="Single-mutant fitness standard deviation above 1e-6",
            one_per_family="Keep the first passing assay of each protein family",
            cubes=f"At least 64 complete three-site subcubes, target {CUBES}",
            nucleotide_libraries="Not considered. The generator takes a protein sequence",
        ),
        candidates=verdicts,
        analysis=dict(
            primary=(
                "Kendall tau between the synthetic and measured warm-start method ranking, "
                "averaged over panel proteins"
            ),
            budgets=BUDGETS,
            search_seeds=SEARCH_SEEDS,
            search_trials=10,
            test_size=TEST_SIZE,
            cubes=CUBES,
            arms=[
                "mutascape",
                "shuffled_singles",
                "structure_only",
                "nk_k1",
                "nk_k2",
                "additive",
                "potts",
                "rough_mount_fuji",
                "fixed_ranking",
            ],
            criterion=(
                "Mutascape beats the shuffled-singles and structure-only controls on mean "
                "Kendall tau, and the paired protein bootstrap 95 per cent interval for each "
                "difference excludes zero. The fixed ranking is reported alongside as a "
                "reference, not as a gate"
            ),
            secondary=[
                "w1 over measured standard deviation",
                "three-site local Walsh L1",
                "sequence-aware classifier two-sample test",
                "probability of reaching the measured top 1 per cent within budget",
            ],
        ),
        source_sha256={
            p: digest(p)
            for p in (
                __file__,
                "src/mutascape/model.py",
                "src/mutascape/benchmarks/datasets.py",
                "scripts/singles_warm_start.py",
            )
        },
    )
    path = OUT / "panel.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(panel)), "panel changed"
    else:
        path.write_text(json.dumps(panel, indent=2) + "\n")
    table = pd.DataFrame(inventory)
    table.to_csv(OUT / "inventory.csv", index=False)
    print(pd.DataFrame(verdicts).to_string(index=False))
    print()
    print(table.to_string(index=False))


def data(family):
    folder = OUT / family
    return (
        pd.read_pickle(folder / "frame.pkl"),
        dict(np.load(folder / "design.npz")),
        json.loads((folder / "metadata.json").read_text()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["screen", "freeze"])
    args = parser.parse_args()
    if args.phase == "screen":
        print(pd.DataFrame(screen(SEED_BASE)).to_string(index=False))
    else:
        freeze()


if __name__ == "__main__":
    main()
