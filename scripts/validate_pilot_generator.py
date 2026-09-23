"""Replay the archived pilot model and add a matched same-assay singles control."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.benchmarks.validity import pairwise_winner_transfer
from mutascape.completion import LandscapePrior, PosteriorLandscape
from mutascape.experimental import PilotCalibratedLandscape
from scripts.run_creilov_combinatorial_validation import parse_mutant


def main():
    archive = Path("results/creilov_pilot_readout_burden_confirmation")
    output = Path("results/review_response_20260923/pilot")
    output.mkdir(exist_ok=True)
    meta = json.loads(Path("results/creilov_c2st/metadata.json").read_text())
    raw = pd.read_excel(meta["raw_path"]).rename(columns={"Unnamed: 0": "variant"})
    raw["mutations"] = [{p: aa for p, _, aa in parse_mutant(v)} for v in raw.variant]
    raw["depth"] = raw.mutations.map(len)
    raw["fitness"] = raw.mean_log - float(raw.loc[raw.depth == 0, "mean_log"].iloc[0])
    singles = raw.loc[raw.depth <= 1, ["mutations", "depth", "fitness"]]
    sequence = "".join(Path("data/creilov/creilov_wt.fasta").read_text().splitlines()[1:])
    same_assay = PosteriorLandscape.condition(
        sequence, singles, LandscapePrior.from_dict(meta["prior"])
    )
    rows = []
    for split in (20260920, 20260921, 20260922):
        pilot = pd.read_csv(archive / f"pilot_{split}.csv", float_precision="round_trip")
        pilot["mutations"] = pilot.mutations.map(ast.literal_eval)
        model = PilotCalibratedLandscape.fit(sequence, singles, pilot)
        (output / f"model_{split}.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
        samples = pd.read_csv(archive / f"samples_{split}.csv", float_precision="round_trip")
        test = raw.set_index("variant").loc[samples.variant].reset_index()
        assert set(samples.variant).isdisjoint(pilot.variant)
        features = np.load(archive / f"features_{split}.npz")
        for draw in (20260923, 20260924, 20260925):
            replay = model.draw(draw).evaluate_batch(test.mutations.tolist())
            np.testing.assert_allclose(replay, samples[f"pilot_{draw}"], atol=1e-10, rtol=1e-10)
            controls = {
                f"pilot_{draw}": (replay, features[f"pilot_{draw}"]),
                f"same_assay_singles_{draw}": (
                    same_assay.draw(draw).evaluate_batch(test.mutations.tolist()),
                    None,
                ),
            }
            for name, (values, predictors) in controls.items():
                if predictors is None:
                    # Use the original audit's feature transform for every model.
                    # These columns depend affinely on fitness, so update its saved features.
                    original = pd.read_csv("data/creilov/creilov.csv")
                    observations = pd.DataFrame(
                        [{"mutations": {}, "depth": 0, "fitness": 0.0}]
                        + [
                            {"mutations": {r.position: r.mutant}, "depth": 1, "fitness": r.fitness}
                            for r in original.itertuples()
                        ]
                    )
                    known = {tuple(m.items()) for m in observations.mutations}
                    observations = pd.concat(
                        [
                            observations,
                            singles.loc[[tuple(m.items()) not in known for m in singles.mutations]],
                        ],
                        ignore_index=True,
                    )
                    predictors = completion_fidelity_features(observations, test, values).to_numpy()
                for classifier in ("logistic", "histogram"):
                    result = classifier_two_sample_test(
                        features["measured"],
                        predictors,
                        real_groups=features["groups"],
                        synthetic_groups=features["groups"],
                        classifier=classifier,
                        seed=split,
                        n_permutations=0,
                    )
                    rows.append(
                        dict(
                            split_seed=split,
                            model=name,
                            classifier=classifier,
                            auc=result.auc,
                            balanced_accuracy=result.accuracy,
                            raw_w1=wasserstein_distance(test.fitness, values),
                            raw_w1_real_sd=wasserstein_distance(test.fitness, values)
                            / np.std(test.fitness),
                        )
                    )
            print(f"Replayed pilot split {split}, draw {draw}", flush=True)
    pd.DataFrame(rows).to_csv(output / "matched_metrics.csv", index=False)
    scores = pd.read_csv(archive / "search.csv", float_precision="round_trip")
    scores = scores.rename(columns={"split_seed": "dataset", "search_seed": "trial"})
    scores["regret"] = -scores.best
    actual = pairwise_winner_transfer(scores, seed=42)
    saved = pd.read_csv(archive / "method_pairs.csv")
    pd.testing.assert_frame_equal(actual, saved, check_dtype=False, atol=1e-12)
    actual.to_csv(output / "replayed_method_pairs.csv", index=False)


if __name__ == "__main__":
    main()
