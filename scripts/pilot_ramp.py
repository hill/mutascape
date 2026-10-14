"""Measure what each extra measured combination buys the generator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.calibrated import QuantileCalibratedLandscape
from mutascape.model import DMSLandscape
from scripts.heldout_panel import OUT as PANEL
from scripts.heldout_panel import data as panel_data
from scripts.heldout_test import campaign_rows, ranking
from scripts.investigate_singles_only import CONFIRMATION, DEVELOPMENT, diagnostics, digest
from scripts.investigate_singles_only import data as study_data

OUT = Path("results/pilot_ramp_20260923")
SIZES = (0, 24, 48, 96, 192)
REPLICATES = (0, 1, 2)
DRAW_SEED = 2026092431
PILOT_SEED = 2026092441
BUDGETS = (96, 192)
# One exclusion set per landscape, the union of every pilot this study can draw.
# Holding the search domain fixed means the ramp varies only the pilot the
# generator saw, not the genotypes a campaign is allowed to buy.
HELD_BACK = max(SIZES) * len(REPLICATES)


def source(name):
    return "panel" if (PANEL / name).exists() else "study"


def load(name):
    return panel_data(name) if source(name) == "panel" else study_data(name)


def landscapes():
    plan = json.loads((PANEL / "panel.json").read_text())
    return (*DEVELOPMENT, *CONFIRMATION, *plan["panel"])


def budgets_for(name, frame):
    combinations = int((frame.depth > 1).sum()) - HELD_BACK
    return [b for b in BUDGETS if b <= 0.1 * combinations]


def design_for(name):
    """Return the frame, design, metadata, pilot pool and searchable rows."""
    frame, design, meta = load(name)
    reserved = np.unique(np.r_[design["test"], design["cubes"].ravel()])
    eligible = np.setdiff1d(np.flatnonzero(frame.depth > 1), reserved)
    # Python's string hash is salted per process, so derive the offset from the name.
    offset = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "big")
    rng = np.random.default_rng(PILOT_SEED + offset % 100000)
    pool = rng.choice(eligible, min(HELD_BACK, len(eligible)), replace=False)
    searchable = np.setdiff1d(np.arange(len(frame)), pool)
    return frame, design, meta, pool, searchable


def pilot_rows(pool, size, replicate):
    return pool[replicate * max(SIZES) : replicate * max(SIZES) + size]


def restrict(frame, design, searchable):
    """Drop the pilot pool from the domain every campaign searches."""
    small = frame.iloc[searchable].reset_index(drop=True)
    kept = dict(
        codes=design["codes"][searchable],
        shape=design["shape"],
        test=design["test"],
        cubes=design["cubes"],
    )
    return small, kept


def model_for(name, meta, frame, pilot):
    singles = frame.loc[frame.depth <= 1]
    if len(pilot) == 0:
        return DMSLandscape.from_singles(meta["sequence"], singles, seed=DRAW_SEED)
    fitted = QuantileCalibratedLandscape.fit(meta["sequence"], singles, frame.iloc[pilot])
    return fitted.draw(DRAW_SEED, reference_size=None)


def measured_ranking(name, budgets):
    path = OUT / name / "measured.csv"
    if path.exists():
        return pd.read_csv(path, index_col=[0, 1])
    frame, design, _, _, searchable = design_for(name)
    small, kept = restrict(frame, design, searchable)
    table = ranking(campaign_rows(small, kept, small.fitness.to_numpy(), budgets))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    print(f"Measured {name}", flush=True)
    return table


def evaluate(name, size, replicate):
    path = OUT / name / f"pilot_{size}_{replicate}.json"
    if path.exists():
        return
    frame, design, meta, pool, searchable = design_for(name)
    budgets = budgets_for(name, frame)
    if not budgets:
        return
    pilot = pilot_rows(pool, size, replicate)
    values = np.asarray(
        model_for(name, meta, frame, pilot).evaluate_batch(frame.mutations.tolist()), dtype=float
    )
    singles = frame.depth <= 1
    np.testing.assert_allclose(values[singles.to_numpy()], frame.loc[singles, "fitness"], atol=1e-9)
    record = dict(
        landscape=name,
        pilot_size=size,
        replicate=replicate,
        **diagnostics(name, values, frame, design),
    )
    test = frame.iloc[design["test"]]
    real = completion_fidelity_features(frame.loc[singles], test).to_numpy()
    generated = completion_fidelity_features(
        frame.loc[singles], test, values[design["test"]]
    ).to_numpy()
    record["c2st"] = float(classifier_two_sample_test(real, generated, seed=DRAW_SEED).accuracy)
    small, kept = restrict(frame, design, searchable)
    measured = measured_ranking(name, budgets)
    generated_rows = ranking(campaign_rows(small, kept, values[searchable], budgets))
    for budget in budgets:
        left = measured.loc[budget]
        right = generated_rows.loc[budget].reindex(left.index)
        record[f"kendall_tau_{budget}"] = float(kendalltau(left.regret, right.regret).statistic)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Ramped {name} {size} {replicate}", flush=True)


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(
        purpose="Measure the return on each extra measured combination given to the generator",
        landscapes=landscapes(),
        sizes=SIZES,
        replicates=len(REPLICATES),
        budgets=BUDGETS,
        draw_seed=DRAW_SEED,
        pilot_seed=PILOT_SEED,
        sampling=(
            "Uniform without replacement from measured combinations, "
            "excluding the evaluation genotypes and every subcube corner"
        ),
        domain=(
            "Every campaign searches the same domain, with the whole pilot pool removed, "
            "so only the pilot the generator saw changes along the ramp"
        ),
        source_sha256={
            p: digest(p)
            for p in (__file__, "src/mutascape/calibrated.py", "scripts/heldout_test.py")
        },
    )
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(record)), "protocol changed"
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")


def report():
    records = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/pilot_*.json"))]
    scores = pd.DataFrame(records)
    columns = ["w1", "local_walsh_l1", "c2st", *[f"kendall_tau_{b}" for b in BUDGETS]]
    table = scores.groupby("pilot_size")[columns].agg(["mean", "count"]).round(3)
    scores.round(4).to_csv(OUT / "ramp.csv", index=False)
    table.to_csv(OUT / "summary.csv")
    print(table.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["protocol", "evaluate", "report"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    if args.phase == "protocol":
        protocol()
        return
    if args.phase == "report":
        report()
        return
    protocol()
    names = [args.only] if args.only else landscapes()
    Parallel(n_jobs=args.jobs)(
        delayed(evaluate)(name, size, replicate)
        for name in names
        for size in SIZES
        for replicate in (REPLICATES if size else REPLICATES[:1])
    )


if __name__ == "__main__":
    main()
