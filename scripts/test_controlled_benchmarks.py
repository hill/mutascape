"""Test whether singles-based complete libraries expose controlled search failures."""

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import hadamard

from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import (
    METHODS,
    _model_method,
    _PairwiseLinear,
    _top_rows,
    make_features,
)
from mutascape.model import DMSLandscape
from scripts.build_quantile_manuscript import BLUE, GREEN, STYLE
from scripts.search_selection_validation import digest, save_json
from scripts.singles_warm_start import DMSOracle

OUT = Path("results/controlled_benchmarks_20260930")
SOURCE = Path("results/search_selection_20260924")
PROTEINS = ("CreiLOV", "eqFP611")
STRENGTHS = (0.0, 0.3, 1.0, 3.0)
SEEDS = tuple(2026093051 + i for i in range(5))


def linear_acquire(pool, rows, values, features, rng, take):
    model = _PairwiseLinear(features.space.alphabet_sizes, max_order=1)
    model.fit(features.space.codes[rows], values)
    return _top_rows(pool, model.predict(features.space.codes[pool]), take)


SEARCH = {
    "random": METHODS["random"],
    "additive_ridge": _model_method(linear_acquire),
    "pairwise_ridge": METHODS["ridge_greedy"],
    "gp_ei": METHODS["gp_ei"],
}


def inputs(protein):
    folder = SOURCE / protein
    frame = pd.read_pickle(folder / "frame.pkl")
    sequence = json.loads((folder / "metadata.json").read_text())["sequence"]
    keys = sorted({k for m in frame.loc[frame.depth == 1].mutations for k in m.items()})
    chosen = {}
    for position, residue in keys:
        chosen.setdefault(position, residue)
    chosen = dict(list(chosen.items())[:10])
    assert len(chosen) == 10
    keys = set(chosen.items())
    singles = frame.loc[frame.mutations.map(lambda m: len(m) <= 1 and set(m.items()) <= keys)]
    assert len(singles) == 11
    codes = np.array(list(product((0, 1), repeat=10)), dtype=np.int16)
    variants = [{p: aa for bit, (p, aa) in zip(row, chosen.items()) if bit} for row in codes]
    return sequence, singles, codes, variants


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists():
        return verify()
    files = [
        Path(__file__),
        Path("docs/controlled_benchmarks_protocol_2026-09-30.md"),
        *Path("src/mutascape").rglob("*.py"),
    ]
    files += [SOURCE / p / f for p in PROTEINS for f in ("frame.pkl", "metadata.json")]
    save_json(
        path,
        dict(
            proteins=PROTEINS,
            strengths=STRENGTHS,
            seeds=SEEDS,
            budgets=[32, 64, 96],
            repeats=10,
            methods=list(SEARCH),
            source_sha256={str(p): digest(p) for p in files},
        ),
    )


def verify():
    for path, expected in json.loads((OUT / "manifest.json").read_text())["source_sha256"].items():
        assert digest(path) == expected, path


def run(protein, strength, sample):
    path = OUT / f"{protein}_{strength}_{sample}.csv"
    if path.exists():
        return
    sequence, singles, codes, variants = inputs(protein)
    model = DMSLandscape.from_singles(
        sequence, singles, seed=SEEDS[sample], mode="linear", strength=strength
    )
    values = model.evaluate_batch(variants)
    known = np.flatnonzero(codes.sum(axis=1) <= 1)
    for index in known:
        key = tuple(sorted(variants[index].items()))
        assert values[index] == model.observations[key]
    unknown = np.setdiff1d(np.arange(len(values)), known)
    space = SearchSpace(
        codes,
        (2,) * 10,
        np.arange(len(codes)),
        int(known[np.argmax(values[known])]),
        {tuple(c): i for i, c in enumerate(codes)},
    )
    features = make_features(space)
    coefficients = values @ hadamard(1024) / 1024
    order = np.array([i.bit_count() for i in range(1024)])
    energy = coefficients[1:] ** 2
    spectrum = [float(np.sum(coefficients[order == k] ** 2) / energy.sum()) for k in range(1, 11)]
    neighbours = np.arange(1024)[:, None] ^ (1 << np.arange(10))
    geometry = dict(
        protein=protein,
        strength=strength,
        sample=sample,
        pair_variance=spectrum[1],
        higher_variance=sum(spectrum[2:]),
        local_optima=int(np.all(values[:, None] > values[neighbours], axis=1).sum()),
        headroom=float(values.max() - values[known].max()),
        fraction_above_known=float((values > values[known].max()).mean()),
    )
    save_json(path.with_suffix(".json"), geometry)
    rows = []
    for budget in (32, 64, 96):
        for trial in range(10):
            seed = 930500000 + 137 * (sample * 10 + trial)
            for label, function in SEARCH.items():
                oracle = DMSOracle(values, budget, known)
                if label == "random":
                    oracle.query(np.random.default_rng(seed).choice(unknown, budget, replace=False))
                else:
                    function(space, oracle, features, seed)
                assert oracle.spent == budget
                rows.append(
                    dict(
                        protein=protein,
                        strength=strength,
                        sample=sample,
                        budget=budget,
                        trial=trial,
                        method=label,
                        regret=(values.max() - oracle.best()) / np.ptp(values),
                    )
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Completed {protein}, pair strength {strength}, sample {sample}", flush=True)


def analyse():
    files = [OUT / f"{p}_{s}_{i}.csv" for p in PROTEINS for s in STRENGTHS for i in range(5)]
    table = pd.concat([pd.read_csv(p) for p in files])
    assert len(table) == 4800
    table.to_csv(OUT / "trials.csv", index=False)
    geometry = pd.DataFrame([json.loads(p.with_suffix(".json").read_text()) for p in files])
    geometry.to_csv(OUT / "geometry.csv", index=False)
    table.groupby(["protein", "strength", "budget", "method"]).regret.mean().to_csv(
        OUT / "summary.csv"
    )
    rows = []
    rng = np.random.default_rng(2026093097)
    for protein in PROTEINS:
        for strength in STRENGTHS:
            for budget in (32, 64, 96):
                sub = table.loc[
                    (table.protein == protein)
                    & (table.strength == strength)
                    & (table.budget == budget)
                ]
                pivot = sub.pivot(index=["sample", "trial"], columns="method", values="regret")
                difference = (pivot.pairwise_ridge - pivot.additive_ridge).to_numpy().reshape(5, 10)
                samples = rng.integers(5, size=(5000, 5))
                trials = rng.integers(10, size=(5000, 5, 10))
                bootstrap = difference[samples[:, :, None], trials].mean(axis=(1, 2))
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                rows.append(
                    dict(
                        protein=protein,
                        strength=strength,
                        budget=budget,
                        difference=difference.mean(),
                        low=low,
                        high=high,
                    )
                )
    contrast = pd.DataFrame(rows)
    contrast.to_csv(OUT / "paired_contrasts.csv", index=False)
    save_json(
        OUT / "audit.json",
        dict(
            campaigns=len(table),
            complete_genotypes=1024,
            all_singles_exact=True,
            all_budgets_spent=True,
            source_sha256={str(p): digest(p) for p in files},
        ),
    )
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for protein, colour in zip(PROTEINS, (BLUE, GREEN)):
        spectrum = (
            geometry.loc[geometry.protein == protein]
            .groupby("strength")
            .pair_variance.agg(["mean", "min", "max"])
        )
        axes[0].plot(spectrum.index, spectrum["mean"], "o-", color=colour, label=protein)
        axes[0].fill_between(
            spectrum.index, spectrum["min"], spectrum["max"], color=colour, alpha=0.15
        )
        scores = contrast.loc[(contrast.protein == protein) & (contrast.budget == 96)]
        axes[1].plot(scores.strength, scores.difference, "o-", color=colour)
        axes[1].fill_between(scores.strength, scores.low, scores.high, color=colour, alpha=0.15)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Pair strength")
        ax.set_xticks(STRENGTHS)
    axes[0].set_ylabel("Variance from pair interactions")
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Pairwise minus additive search regret")
    axes[1].set_title("Negative values favour pairwise search")
    fig.tight_layout()
    fig.savefig(OUT / "controlled_benchmarks.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "controlled_benchmarks.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "analyse"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        verify()
        Parallel(n_jobs=args.jobs)(
            delayed(run)(p, s, i) for p in PROTEINS for s in STRENGTHS for i in range(5)
        )
    else:
        verify()
        analyse()
