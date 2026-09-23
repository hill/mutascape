"""Check fixed range and pair-mean changes without changing the manuscript model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance

from mutascape import SimpleLandscape
from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from scripts.investigate_singles_only import CONFIRMATION, SEEDS, data, digest, local_geometry

OUT = Path("results/singles_distribution_fixes_20260923")
PROTEINS = ("CreiLOV", *CONFIRMATION)
SETTINGS = {
    "wider_sigmoid": dict(mode="sigmoid", margin=0.5, negative_pair_mean=0.0),
    "linear": dict(mode="linear", margin=0.05, negative_pair_mean=0.0),
    "negative_pairs": dict(mode="linear", margin=0.05, negative_pair_mean=0.3),
}


def evaluate(name, label, seed):
    path = OUT / f"{name}_{label}_{seed}.json"
    if path.exists():
        return
    frame, design, metadata = data(name)
    singles = frame.loc[frame.depth <= 1]
    setting = SETTINGS[label]
    model = SimpleLandscape.from_singles(
        metadata["sequence"],
        singles,
        seed=seed,
        strength=0.3,
        mode=setting["mode"],
        margin=setting["margin"],
    )
    requested = np.unique(np.r_[design["test"], design["cubes"].ravel(), singles.index])
    selected = frame.iloc[requested]
    scores = model.evaluate_batch(selected.mutations.tolist())
    depth = selected.depth.to_numpy()
    scores -= setting["negative_pair_mean"] * model.scale * depth * (depth - 1) / 2
    values = np.full(len(frame), np.nan)
    values[requested] = scores
    np.testing.assert_array_equal(values[singles.index], singles.fitness)
    test = frame.iloc[design["test"]]
    real = completion_fidelity_features(singles, test).to_numpy()
    generated = completion_fidelity_features(singles, test, values[design["test"]]).to_numpy()
    codes = design["codes"][design["test"]]
    indicators = np.column_stack(
        [codes[:, j] == a for j, n in enumerate(design["shape"]) for a in range(n)]
    )
    result = classifier_two_sample_test(
        np.column_stack([real, indicators]),
        np.column_stack([generated, indicators]),
        real_groups=design["test"],
        synthetic_groups=design["test"],
        classifier="histogram",
        n_permutations=0,
        seed=2026092300,
    )
    measured_spectrum, _ = local_geometry(frame.fitness.to_numpy(), design["cubes"])
    spectrum, _ = local_geometry(values, design["cubes"])
    record = dict(
        protein=name,
        model=label,
        seed=seed,
        auc_sequence=result.auc,
        w1=wasserstein_distance(test.fitness, values[design["test"]]) / np.std(test.fitness),
        local_walsh_l1=float(np.abs(spectrum - measured_spectrum).sum()),
        measured_median=float(test.fitness.median()),
        generated_median=float(np.median(values[design["test"]])),
    )
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Checked {name} {label} {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        purpose="Exploratory checks after inspecting CreiLOV, with no new model selection",
        proteins=PROTEINS,
        seeds=SEEDS,
        strength=0.3,
        settings=SETTINGS,
        assessment="Existing test genotypes and complete local subcubes",
        fitting="WT and singles only. No target combination values select parameters.",
        penalty="Subtract 0.3 times single-effect SD per mutated pair in linear mode",
        limits="No search experiments and no independent confirmation after these checks",
        source_sha256={
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("src/mutascape/model.py"),
                Path("src/mutascape/benchmarks/evaluation.py"),
            )
        },
    )
    record = OUT / "protocol.json"
    if record.exists():
        assert json.loads(record.read_text()) == json.loads(json.dumps(protocol))
    else:
        record.write_text(json.dumps(protocol, indent=2) + "\n")
    Parallel(n_jobs=args.jobs)(
        delayed(evaluate)(name, label, seed)
        for name in PROTEINS
        for label in SETTINGS
        for seed in SEEDS
    )
    rows = [
        json.loads((OUT / f"{n}_{m}_{s}.json").read_text())
        for n in PROTEINS
        for m in SETTINGS
        for s in SEEDS
    ]
    frozen = json.loads(
        Path("src/mutascape/data/validation/singles-only-research-v1.json").read_text()
    )
    rows += [
        dict(r, model="Mutascape")
        for r in frozen["fidelity"]
        if r["protein"] in PROTEINS and r["model"] == "sigmoid"
    ]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics.csv", index=False)
    summary = table.groupby(["protein", "model"])[["auc_sequence", "w1", "local_walsh_l1"]].median()
    summary.to_csv(OUT / "summary.csv")
    print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
