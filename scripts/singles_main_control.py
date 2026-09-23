"""Compare the main singles generator with target-excluded donor priors."""

import argparse
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed

from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.completion import LandscapePrior, PosteriorLandscape
from scripts.investigate_singles_only import (
    CONFIRMATION,
    DEVELOPMENT,
    OUT,
    SEEDS,
    build,
    data,
    diagnostics,
    search,
)


def fit_prior(name):
    path = OUT / f"prior_{name}.json"
    if path.exists():
        return
    donors = []
    for donor in DEVELOPMENT:
        if donor == name:
            continue
        frame, _, meta = data(donor)
        donors.append(
            SimpleNamespace(variants=frame, wild_type_sequence=meta["sequence"], protein=donor)
        )
    prior = LandscapePrior.fit(donors, max_order=2)
    path.write_text(
        json.dumps(dict(donors=[d.protein for d in donors], prior=prior.to_dict()), indent=2) + "\n"
    )
    print(f"Fitted donor prior for {name}", flush=True)


def evaluate(name, seed):
    folder = OUT / name
    path = folder / f"metrics_main_singles_{seed}.json"
    if path.exists():
        return
    frame, design, meta = data(name)
    singles = frame.loc[frame.depth <= 1]
    prior_name = name if name in DEVELOPMENT else "confirmation"
    record = json.loads((OUT / f"prior_{prior_name}.json").read_text())
    assert name not in record["donors"]
    prior = LandscapePrior.from_dict(record["prior"])
    posterior = PosteriorLandscape.condition(meta["sequence"], singles, prior)
    prototype = build(name, dict(mode="linear", strength=0.2, margin=0.05), seed)
    # The main coefficient has variance four. The prototype divides it by two.
    vectorised = replace(
        prototype,
        effects=posterior.preimage_effects or posterior.single_effects,
        scale=posterior.preimage_scale if posterior.nonlinearity else posterior.target_scale,
        readout=posterior.nonlinearity,
    )
    variants = frame.mutations.tolist()
    values = vectorised.evaluate_batch(variants)
    check = design["test"][:100]
    np.testing.assert_allclose(
        values[check], posterior.draw(seed).evaluate_batch([variants[i] for i in check]), atol=1e-12
    )
    np.save(folder / f"fitness_main_singles_{seed}.npy", values)
    result = dict(
        protein=name, model="main_singles", seed=seed, **diagnostics(name, values, frame, design)
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
    print(f"Evaluated {name} main_singles {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["priors", "final", "search"])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "priors":
        Parallel(n_jobs=args.jobs)(
            delayed(fit_prior)(name) for name in (*DEVELOPMENT, "confirmation")
        )
    elif args.phase == "final":
        Parallel(n_jobs=args.jobs)(
            delayed(evaluate)(name, seed)
            for name in (*DEVELOPMENT, *CONFIRMATION)
            for seed in SEEDS
        )
    else:
        Parallel(n_jobs=args.jobs)(
            delayed(search)(name, "main_singles", seed)
            for name in (*DEVELOPMENT, *CONFIRMATION)
            for seed in SEEDS
        )


if __name__ == "__main__":
    main()
