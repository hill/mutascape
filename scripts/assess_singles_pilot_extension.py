"""Assess Combination-calibrated Mutascape on the fixed singles-only evaluation genotypes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.experimental.quantile import QuantileCalibratedLandscape
from mutascape.mutant import parse_variant
from scripts.investigate_singles_only import (
    CONFIRMATION,
    DEVELOPMENT,
    SEEDS,
    data,
    diagnostics,
    digest,
)

OUT = Path("results/singles_pilot_extension_20260923")
FROZEN = Path("src/mutascape/data/validation/singles-pilot-extension-v1.json")


def prepare():
    OUT.mkdir(exist_ok=True)
    protocol = dict(
        model="Existing QuantileCalibratedLandscape without parameter changes",
        pilot_size=192,
        proteins=(*DEVELOPMENT, *CONFIRMATION),
        seeds=SEEDS,
        selection_seed=2026092327,
        evaluation="The same 1000 genotypes and complete subcubes as the singles-only study",
        pilot_selection="Uniform without replacement from eligible measured combinations",
        exclusions="WT, singles, evaluation genotypes and every spectral subcube corner",
        source_sha256={
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("src/mutascape/experimental/quantile.py"),
                Path("src/mutascape/experimental/pilot.py"),
            )
        },
    )
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        path.write_text(json.dumps(protocol, indent=2) + "\n")
    for index, name in enumerate((*DEVELOPMENT, *CONFIRMATION)):
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        frame, design, meta = data(name)
        excluded = np.unique(np.r_[design["test"], design["cubes"].ravel()])
        eligible = np.setdiff1d(np.flatnonzero(frame.depth > 1), excluded)
        pilot = np.random.default_rng(2026092327 + index).choice(eligible, 192, replace=False)
        assert not np.intersect1d(pilot, excluded).size
        labels = [
            parse_variant(m, wild_type_sequence=meta["sequence"]).label
            for m in frame.iloc[pilot].mutations
        ]
        pd.DataFrame({"row": pilot, "variant": labels}).to_csv(folder / "pilot.csv", index=False)
        model = QuantileCalibratedLandscape.fit(
            meta["sequence"], frame.loc[frame.depth <= 1], frame.iloc[pilot]
        )
        (folder / "model.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
        print(f"Prepared {name} pilot with {len(eligible)} eligible genotypes", flush=True)


def evaluate(name, seed):
    folder = OUT / name
    path = folder / f"metrics_{seed}.json"
    if path.exists():
        return
    frame, design, _ = data(name)
    singles = frame.loc[frame.depth <= 1]
    requested = np.unique(np.r_[design["test"], design["cubes"].ravel(), singles.index])
    model = QuantileCalibratedLandscape.from_dict(json.loads((folder / "model.json").read_text()))
    landscape = model.draw(seed)
    values = np.full(len(frame), np.nan)
    values[requested] = landscape.evaluate_batch(frame.iloc[requested].mutations.tolist())
    np.testing.assert_array_equal(values[singles.index], singles.fitness)
    np.save(folder / f"evaluation_{seed}.npy", values)
    result = dict(
        protein=name, seed=seed, pilot_size=192, **diagnostics(name, values, frame, design)
    )
    test = frame.iloc[design["test"]]
    real = completion_fidelity_features(singles, test).to_numpy()
    generated = completion_fidelity_features(singles, test, values[design["test"]]).to_numpy()
    codes = design["codes"][design["test"]]
    indicators = np.column_stack(
        [codes[:, j] == a for j, n in enumerate(design["shape"]) for a in range(n)]
    )
    for suffix, x, y in [
        ("base", real, generated),
        ("sequence", np.column_stack([real, indicators]), np.column_stack([generated, indicators])),
    ]:
        score = classifier_two_sample_test(
            x,
            y,
            real_groups=design["test"],
            synthetic_groups=design["test"],
            classifier="histogram",
            n_permutations=0,
            seed=2026092300,
        )
        result[f"auc_{suffix}"] = score.auc
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Evaluated {name} Combination-calibrated Mutascape {seed}", flush=True)


def freeze():
    rows = [
        json.loads((OUT / name / f"metrics_{seed}.json").read_text())
        for name in (*DEVELOPMENT, *CONFIRMATION)
        for seed in SEEDS
    ]
    table = pd.DataFrame(rows)
    medians = table.groupby("protein").median(numeric_only=True).reset_index()
    payload = dict(
        id="singles-pilot-extension-v1",
        protocol=json.loads((OUT / "protocol.json").read_text()),
        metrics=rows,
        medians=json.loads(medians.to_json(orient="records")),
        pilots={
            name: json.loads(pd.read_csv(OUT / name / "pilot.csv").to_json(orient="records"))
            for name in (*DEVELOPMENT, *CONFIRMATION)
        },
    )
    FROZEN.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(medians.round(3).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "freeze"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "run":
        Parallel(n_jobs=args.jobs)(
            delayed(evaluate)(name, seed)
            for name in (*DEVELOPMENT, *CONFIRMATION)
            for seed in SEEDS
        )
    else:
        freeze()


if __name__ == "__main__":
    main()
