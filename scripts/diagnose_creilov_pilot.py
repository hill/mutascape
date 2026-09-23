"""Test a small combinatorial pilot on separate held-out genotypes."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from run_creilov_combinatorial_validation import parse_mutant
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.completion import dms_pairwise_unit_coefficient


def main():
    reference = Path("results/creilov_c2st")
    metadata = json.loads((reference / "metadata.json").read_text())
    raw = pd.read_excel(metadata["raw_path"]).rename(columns={"Unnamed: 0": "variant"})
    raw["mutations"] = [{p: aa for p, _, aa in parse_mutant(v)} for v in raw.variant]
    raw["depth"] = raw.mutations.map(len)
    raw["fitness"] = raw.mean_log - float(raw.loc[raw.depth == 0, "mean_log"].iloc[0])
    obs = raw.loc[raw.depth <= 1, ["mutations", "depth", "fitness"]]
    effects = {
        tuple(m.items())[0]: f for m, d, f in obs.itertuples(index=False, name=None) if d == 1
    }
    sequence = "".join(Path("data/creilov/creilov_wt.fasta").read_text().splitlines()[1:])
    reserved = set()
    for seed in (42, 43, 44):
        reserved.update(pd.read_csv(reference / f"samples_{seed}.csv").variant)
    pool = raw.loc[(raw.depth > 1) & ~raw.variant.isin(reserved) & np.isfinite(raw.fitness)]

    # Fixed candidate family. All correction terms vanish at WT and singles.
    def design(frame):
        depth = frame.depth.to_numpy()
        additive = np.array([sum(effects[k] for k in m.items()) for m in frame.mutations])
        pairs = depth * (depth - 1) / 2
        return additive, np.column_stack([pairs, additive * (depth - 1)])

    output = Path(".mutascape/experiments/creilov_pilot")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget in (48, 192):
        for seed in (42, 43, 44):
            pilot = pool.sample(n=budget, random_state=seed)
            additive, design_train = design(pilot)
            # No intercept preserves the single-mutant boundary.
            model = make_pipeline(
                StandardScaler(with_mean=False), Ridge(alpha=1, fit_intercept=False)
            )
            model.fit(design_train, pilot.fitness - additive)
            residual = pilot.fitness.to_numpy() - additive - model.predict(design_train)
            scale = np.sqrt(np.sum(residual**2) / np.sum(pilot.depth * (pilot.depth - 1) / 2))
            sample = pd.read_csv(reference / f"samples_{seed}.csv")
            sample["mutations"] = [{p: aa for p, _, aa in parse_mutant(v)} for v in sample.variant]
            sample["fitness"] = sample.measured
            base, design_test = design(sample)
            real = completion_fidelity_features(obs, sample).to_numpy()
            groups = np.arange(len(sample))
            for draw in (20260826, 20260827, 20260828):
                noise = np.array(
                    [
                        sum(
                            dms_pairwise_unit_coefficient(a, b, sequence, draw)
                            for a, b in combinations(sorted(m.items()), 2)
                        )
                        for m in sample.mutations
                    ]
                )
                # Each coefficient sums four independent unit-variance cells.
                values = base + model.predict(design_test) + scale * noise / 2
                synth = completion_fidelity_features(obs, sample, values).to_numpy()
                for classifier in ("logistic", "histogram"):
                    result = classifier_two_sample_test(
                        real,
                        synth,
                        real_groups=groups,
                        synthetic_groups=groups,
                        classifier=classifier,
                        seed=seed,
                        n_permutations=0,
                    )
                    row = dict(
                        budget=budget,
                        seed=seed,
                        draw=draw,
                        classifier=classifier,
                        auc=result.auc,
                        raw_w1=wasserstein_distance(sample.measured, values),
                        mean=values.mean(),
                        std=values.std(),
                        pair_scale=scale,
                        coefficients=json.dumps(model[-1].coef_.tolist()),
                    )
                    rows.append(row)
                    print(row, flush=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
