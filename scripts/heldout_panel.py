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
TEST_SIZE = 1000
CUBES = 256
SEED_BASE = 2026092400
SEARCH_SEEDS = (2026092411, 2026092412, 2026092413)
BUDGETS = (48, 96, 192)
# A campaign that buys a tenth of its own library is not a search problem, so
# each landscape keeps only the budgets it can support.
BUDGET_FRACTION = 0.1

# Proteins already used to develop or confirm the generator. A library from one
# of these is kept, but its scores are reported apart from the held-out mean.
USED_PROTEINS = ("DHFR", "eqFP611", "ParB-Noc", "ParD2", "CreiLOV", "GB1", "TrpB4", "T7", "TEV")

# Libraries sharing a genotype space differ only in the assayed target, so the
# panel bootstrap resamples spaces rather than rows of this list.
CANDIDATES = (
    ("PhoQ", "PhoQ", "graphfla", "Podgornaia2015_PhoQ.csv", "AVST"),
    ("CR9114", "CR9114-h1", "graphfla", "Phillips2021_CR9114_h1.csv", "auto"),
    ("CR9114", "CR9114-h3", "graphfla", "Phillips2021_CR9114_h3.csv", "auto"),
    ("CR9114", "CR9114-fluB", "graphfla", "Phillips2021_CR9114_fluB.csv", "auto"),
    ("CR6261", "CR6261-h1", "graphfla", "Phillips2021_CR6261_h1.csv", "auto"),
    ("CR6261", "CR6261-h9", "graphfla", "Phillips2021_CR6261_h9.csv", "auto"),
    ("SARS-CoV-2-RBD", "RBD-ACE2", "graphfla", "Moulana2022_ACE2.csv", "auto"),
    ("SARS-CoV-2-RBD", "RBD-S309", "graphfla", "Moulana2023_S309.csv", "auto"),
    ("SARS-CoV-2-RBD", "RBD-CB6", "graphfla", "Moulana2023_CB6.csv", "auto"),
    ("SARS-CoV-2-RBD", "RBD-CoV555", "graphfla", "Moulana2023_CoV555.csv", "auto"),
    ("SARS-CoV-2-RBD", "RBD-REGN10987", "graphfla", "Moulana2023_REGN10987.csv", "auto"),
    ("H1-haemagglutinin", "H1-SI06", "graphfla", "Phillips2023_SI06.csv", "auto"),
    ("H1-haemagglutinin", "H1-MA90", "graphfla", "Phillips2023_MA90.csv", "auto"),
    ("H1-haemagglutinin", "H1-G189E", "graphfla", "Phillips2023_G189E.csv", "auto"),
    ("H3-haemagglutinin", "H3-HK68", "graphfla", "Wu2020_HK68.csv", "EEFDFA"),
    ("H3-haemagglutinin", "H3-Bk79", "graphfla", "Wu2020_Bk79.csv", "EEFDFA"),
    ("COV107-23", "COV107-23", "graphfla", "Schulz2025.csv", "auto"),
    ("Hsp90", "Hsp90", "graphfla", "Bank2016a.csv", "QFGFSAAAE"),
    ("IMDH", "IMDH", "graphfla", "Lunzer2005_fitness.csv", "DDFAGR"),
    ("phage-lambda-J", "lambda-J-LamB", "graphfla", "Michael2024_LamB.csv", "auto"),
    ("phage-lambda-J", "lambda-J-OmpF", "graphfla", "Michael2024_OmpF.csv", "auto"),
    ("RhlA", "RhlA-RLs", "combingym", "RhlA_RLs", None),
    ("RhlA", "RhlA-product", "combingym", "RhlA_product", None),
    ("SaCas9", "SaCas9", "combingym", "SaCas9_activity", None),
    ("SpCas9", "SpCas9", "combingym", "SpCas9_activity", None),
    ("HIV-1", "HIV-1", "combingym", "HIV1_activity", None),
    ("ParB-Noc", "ParB-parS", "graphfla", "Jalal2020_parS.csv", "RTAG"),
    ("ParD2", "ParD3", "ssmula", "ParD3", None),
    ("TrpB4", "TrpB3A", "ssmula", "TrpB3A", None),
    ("TrpB4", "TrpB3F", "ssmula", "TrpB3F", None),
)


def spec_for(family, source, key, reference):
    if source == "graphfla":
        return GRAPHFLA, _graphfla_spec(key, family, reference)
    if source == "combingym":
        return COMBINGYM, next(s for s in COMBINGYM_MANIFEST if s.name == key)
    return SSMULA, next(s for s in SSMULA_MANIFEST if s.name == key)


def binary_reference(key):
    """Return the all-ancestral genotype of a GraphFLA presence-absence table."""
    columns = pd.read_csv(GRAPHFLA / "data" / "BioSequence" / key, nrows=0).columns
    return "0" * sum(c.startswith("pos") and c[3:].isdigit() for c in columns)


def load(family, source, key, reference):
    """Return the measured library restricted to substitutions with a measured single."""
    if reference == "auto":
        reference = binary_reference(key)
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
    """Measure every candidate and drop only those the protocol cannot run on."""
    verdicts = []
    for number, (family, name, source, key, reference) in enumerate(CANDIDATES):
        record = dict(landscape=name, genotype_space=family, source=source, dataset=key)
        frame, _, codes, shape, _, _ = load(family, source, key, reference)
        singles = frame.loc[frame.depth == 1]
        combinations = int((frame.depth > 1).sum())
        budgets = [b for b in BUDGETS if b <= BUDGET_FRACTION * combinations]
        record.update(
            held_out=family not in USED_PROTEINS,
            combinations=combinations,
            sites=len(shape),
            singles=len(singles),
            singles_sd=round(float(singles.fitness.std()), 6),
            budgets=budgets,
        )
        if record["singles_sd"] < 1e-6:
            record["excluded"] = "singles are all equal, so the model has nothing to fit"
        elif not budgets:
            record["excluded"] = f"too small for a budget of {min(BUDGETS)}"
        else:
            try:
                record["cubes"] = len(sample_cubes(codes, shape, seed + number))
            except ValueError as error:
                record["excluded"] = str(error)
        verdicts.append(record)
    return verdicts


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    verdicts = screen(SEED_BASE)
    accepted = [v for v in verdicts if "excluded" not in v]
    inventory = []
    for number, (family, name, source, key, reference) in enumerate(CANDIDATES):
        if not any(v["landscape"] == name for v in accepted):
            continue
        frame, sequence, codes, shape, path, spec = load(family, source, key, reference)
        folder = OUT / name
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
                    landscape=name,
                    genotype_space=family,
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
                landscape=name,
                genotype_space=family,
                held_out=family not in USED_PROTEINS,
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
            "Panel for the preregistered test of whether Mutascape transfers method "
            "rankings from DMS singles alone"
        ),
        frozen_before_scoring=True,
        panel=[record["landscape"] for record in accepted],
        held_out=[record["landscape"] for record in accepted if record["held_out"]],
        rules=dict(
            singles_variance="Drop a library whose singles all share one fitness",
            budgets=(
                f"Keep the budgets in {BUDGETS} that cost at most "
                f"{BUDGET_FRACTION:g} of the measured combinations"
            ),
            cubes=f"Drop a library with fewer than 64 three-site subcubes, target {CUBES}",
            used_proteins=(
                "Libraries from these proteins stay in the panel and are scored, but the "
                "headline mean covers the held-out ones only"
            ),
            nucleotide_libraries="Not considered. The generator takes a protein sequence",
        ),
        used_proteins=USED_PROTEINS,
        candidates=verdicts,
        analysis=dict(
            primary=(
                "Kendall tau between the synthetic and measured warm-start method ranking, "
                "averaged over held-out landscapes"
            ),
            budgets=BUDGETS,
            budget_fraction=BUDGET_FRACTION,
            search_seeds=SEARCH_SEEDS,
            search_trials=10,
            test_size=TEST_SIZE,
            cubes=CUBES,
            bootstrap=(
                "Paired bootstrap over genotype spaces, not over landscapes, since several "
                "libraries assay different targets on one genotype space"
            ),
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
                "Kendall tau, and the paired bootstrap 95 per cent interval for each "
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
