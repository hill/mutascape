"""Task-relative checks for synthetic search benchmarks."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def _mean_regret(scores: pd.DataFrame, dataset: str | int, arm: str) -> pd.Series:
    selected = scores.loc[(scores["dataset"] == dataset) & (scores["arm"] == arm)]
    return selected.groupby("method")["regret"].mean()


def reliable_method_pairs(
    scores: pd.DataFrame,
    *,
    seed: int = 0,
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return measured method pairs whose paired interval excludes zero."""
    rng = np.random.default_rng(seed)
    tail = (1.0 - confidence) / 2.0
    rows = []
    for dataset, dataset_scores in scores.groupby("dataset"):
        measured = dataset_scores.loc[dataset_scores["arm"] == "measured"].pivot(
            index="trial", columns="method", values="regret"
        )
        for first, second in combinations(sorted(measured.columns), 2):
            differences = (measured[first] - measured[second]).dropna().to_numpy(dtype=float)
            if differences.size < 4:
                continue
            indices = rng.integers(0, differences.size, size=(n_bootstrap, differences.size))
            low, high = np.quantile(differences[indices].mean(axis=1), [tail, 1.0 - tail])
            if high < 0 or low > 0:
                rows.append(
                    {
                        "dataset": dataset,
                        "first": first,
                        "second": second,
                        "sign": -1 if high < 0 else 1,
                        "lower": float(low),
                        "upper": float(high),
                    }
                )
    return pd.DataFrame(rows, columns=["dataset", "first", "second", "sign", "lower", "upper"])


def pairwise_winner_transfer(
    scores: pd.DataFrame,
    *,
    seed: int = 0,
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Score synthetic comparisons on method pairs with a reliable measured winner.

    Trials are paired by seed. A measured pair counts only when the bootstrap
    interval for its mean regret difference excludes zero. Lower regret wins.
    A synthetic tie counts as a miss because it does not recover that winner.
    """
    pairs = reliable_method_pairs(scores, seed=seed, n_bootstrap=n_bootstrap, confidence=confidence)
    rows = []
    for dataset, dataset_scores in scores.groupby("dataset"):
        reliable = pairs.loc[pairs.dataset == dataset]
        for arm in sorted(set(dataset_scores["arm"]) - {"measured"}):
            means = _mean_regret(scores, dataset, arm)
            correct = scored = 0
            for pair in reliable.itertuples(index=False):
                if pair.first not in means or pair.second not in means:
                    continue
                scored += 1
                difference = float(means[pair.first] - means[pair.second])
                correct += int(np.sign(difference) == pair.sign)
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "reliable_pairs": len(reliable),
                    "scored_pairs": scored,
                    "correct_pairs": correct,
                    "winner_accuracy": correct / scored if scored else float("nan"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "arm",
            "reliable_pairs",
            "scored_pairs",
            "correct_pairs",
            "winner_accuracy",
        ],
    )


def measured_rank_reliability(
    scores: pd.DataFrame,
    *,
    seed: int = 0,
    n_splits: int = 500,
) -> pd.DataFrame:
    """Estimate how much method ranks vary across measured search trials."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    measured_scores = scores.loc[scores["arm"] == "measured"]
    for dataset, group in measured_scores.groupby("dataset"):
        pivot = group.pivot(index="trial", columns="method", values="regret").dropna()
        if len(pivot) < 4 or pivot.shape[1] < 2:
            continue
        values = pivot.to_numpy(dtype=float)
        taus: list[float] = []
        half = len(values) // 2
        for _ in range(n_splits):
            order = rng.permutation(len(values))
            first = values[order[:half]].mean(axis=0)
            second = values[order[half : 2 * half]].mean(axis=0)
            tau = float(kendalltau(first, second).statistic)
            if np.isfinite(tau):
                taus.append(tau)
        if not taus:
            continue
        rows.append(
            {
                "dataset": dataset,
                "n_methods": pivot.shape[1],
                "n_trials": len(pivot),
                "split_half_median": float(np.median(taus)),
                "split_half_lower": float(np.quantile(taus, 0.05)),
                "split_half_upper": float(np.quantile(taus, 0.95)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "n_methods",
            "n_trials",
            "split_half_median",
            "split_half_lower",
            "split_half_upper",
        ],
    )


def reliability_adjusted_transfer(
    transfer: pd.DataFrame,
    reliability: pd.DataFrame,
) -> pd.DataFrame:
    """Mark arms whose rank transfer reaches measured split-half reliability."""
    merged = transfer.merge(reliability, on="dataset", how="inner")
    merged["meets_reliability_floor"] = merged["kendall_tau"] >= merged["split_half_lower"]
    return merged


def fixed_method_ranking(scores: pd.DataFrame, *, excluded_protein: str) -> pd.Series:
    """Fit a fixed method ranking with one vote per training protein.

    Exclude every assay of the test protein before averaging measured regret.
    First average trials within each assay, then assays within each protein.
    """
    training = scores.loc[(scores.protein != excluded_protein) & (scores.arm == "measured")]
    if training.empty:
        raise ValueError("no training proteins remain")
    assays = training.groupby(["protein", "dataset", "method"]).regret.mean()
    return assays.groupby(["protein", "method"]).mean().groupby("method").mean()
