"""Compare single-assay anchors without fitting any combinatorial measurements."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_creilov_combinatorial_validation import parse_mutant
from scipy.stats import wasserstein_distance

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.completion import LandscapePrior, PosteriorLandscape


def main():
    reference = Path("results/creilov_c2st")
    metadata = json.loads((reference / "metadata.json").read_text())
    prior = LandscapePrior.from_dict(metadata["prior"])
    sequence = "".join(Path("data/creilov/creilov_wt.fasta").read_text().splitlines()[1:])
    raw = pd.read_excel(metadata["raw_path"], nrows=100)
    labels = raw["Unnamed: 0"]
    mutations = [{p: aa for p, _, aa in parse_mutant(label)} for label in labels]
    wt = float(raw.loc[[not m for m in mutations], "mean_log"].iloc[0])
    library = (
        pd.DataFrame(
            {
                "mutations": mutations,
                "depth": list(map(len, mutations)),
                "fitness": raw.mean_log - wt,
            }
        )
        .query("depth <= 1")
        .reset_index(drop=True)
    )
    dms = pd.read_csv("data/creilov/creilov.csv")
    dms_effects = {(r.position, r.mutant): r.fitness for r in dms.itertuples()}
    agreement = pd.DataFrame(
        [
            {"position": p, "mutant": aa, "dms": dms_effects.get((p, aa), np.nan), "library": f}
            for m, d, f in library.itertuples(index=False, name=None)
            if d == 1
            for p, aa in m.items()
        ]
    )
    output = Path(".mutascape/experiments/creilov_anchors")
    output.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(output / "single_assay_agreement.csv", index=False)
    original = pd.DataFrame(
        [{"mutations": {}, "depth": 0, "fitness": 0.0}]
        + [
            {"mutations": {r.position: r.mutant}, "depth": 1, "fitness": r.fitness}
            for r in dms.itertuples()
        ]
    )
    keys = {tuple(m.items()) for m in original.mutations}
    missing = library.loc[[tuple(m.items()) not in keys for m in library.mutations]]
    original = pd.concat([original, missing], ignore_index=True)
    library_keys = {tuple(m.items()) for m in library.mutations}
    override = pd.concat(
        [original.loc[[tuple(m.items()) not in library_keys for m in original.mutations]], library],
        ignore_index=True,
    )
    rows = []
    for sample_seed in (42, 43, 44):
        sample = pd.read_csv(reference / f"samples_{sample_seed}.csv")
        variants = pd.DataFrame(
            {
                "mutations": [{p: aa for p, _, aa in parse_mutant(s)} for s in sample.variant],
                "depth": sample.depth,
                "fitness": sample.measured,
            }
        )
        # Hold the feature definition fixed across all arms, including the baseline.
        real = completion_fidelity_features(original, variants).to_numpy()
        groups = np.arange(len(sample))
        for name, observations in (
            ("original_dms", original),
            ("same_assay_override", override),
            ("same_assay_only", library),
        ):
            generator = PosteriorLandscape.condition(sequence, observations, prior)
            for draw_seed in (20260826, 20260827, 20260828):
                values = generator.draw(draw_seed).evaluate_batch(variants.mutations.tolist())
                features = completion_fidelity_features(original, variants, values).to_numpy()
                for classifier in ("logistic", "histogram"):
                    result = classifier_two_sample_test(
                        real,
                        features,
                        real_groups=groups,
                        synthetic_groups=groups,
                        classifier=classifier,
                        seed=sample_seed,
                        n_permutations=0,
                    )
                    row = dict(
                        arm=name,
                        sample_seed=sample_seed,
                        draw_seed=draw_seed,
                        classifier=classifier,
                        auc=result.auc,
                        mean=float(values.mean()),
                        std=float(values.std()),
                        real_mean=float(sample.measured.mean()),
                        raw_w1=wasserstein_distance(sample.measured, values),
                    )
                    rows.append(row)
                    print(row, flush=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
