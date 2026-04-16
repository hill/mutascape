"""Landscape metrics for comparing real and synthetic epistasis patterns.

All functions accept a DataFrame with columns:
    mutations, fitness, additive_prediction, depth, epistatic_deviation

This lets us compute identical metrics on both real multi-mutant data
and synthetic mutascape samples.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def additive_correlation_profile(
    df: pd.DataFrame,
    depths: list[int] | None = None,
) -> dict[int, float]:
    """Pearson correlation between additive prediction and actual fitness at each depth."""
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 3:
            continue
        r, _ = sp_stats.pearsonr(sub["additive_prediction"], sub["fitness"])
        result[d] = r
    return result


def sign_epistasis_profile(
    df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
    depths: list[int] | None = None,
) -> dict[int, float]:
    """Fraction of variants with sign epistasis at each depth.

    A variant has sign epistasis if any constituent mutation has its
    contextual effect (actual - additive_without_this_mutation) with
    opposite sign to its single-site effect.
    """
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 10:
            continue

        n_sign_epi = 0
        for _, row in sub.iterrows():
            mutations = row["mutations"]
            total_fitness = row["fitness"]

            for pos, mut_aa in mutations.items():
                single_effect = single_site_effects.get(pos, {}).get(mut_aa, 0.0)
                if single_effect == 0.0:
                    continue

                # Additive prediction without this mutation
                other_additive = sum(
                    single_site_effects.get(p, {}).get(a, 0.0)
                    for p, a in mutations.items()
                    if p != pos
                )
                contextual_effect = total_fitness - other_additive

                if np.sign(contextual_effect) != np.sign(single_effect):
                    n_sign_epi += 1
                    break

        result[d] = n_sign_epi / len(sub)
    return result


def epistatic_deviation_stats(
    df: pd.DataFrame,
    depths: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Summary statistics of epistatic deviations at each depth.

    Returns dict[depth, {mean, std, mae, skewness, kurtosis, q5, q25, q50, q75, q95}]
    """
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 10:
            continue

        dev = sub["epistatic_deviation"].values
        result[d] = {
            "mean": float(np.mean(dev)),
            "std": float(np.std(dev)),
            "mae": float(np.mean(np.abs(dev))),
            "skewness": float(sp_stats.skew(dev)),
            "kurtosis": float(sp_stats.kurtosis(dev)),
            "q5": float(np.percentile(dev, 5)),
            "q25": float(np.percentile(dev, 25)),
            "q50": float(np.percentile(dev, 50)),
            "q75": float(np.percentile(dev, 75)),
            "q95": float(np.percentile(dev, 95)),
        }
    return result


def fitness_moments(
    df: pd.DataFrame,
    depths: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Mean and std of fitness distribution at each depth."""
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 3:
            continue
        result[d] = {
            "mean": float(sub["fitness"].mean()),
            "std": float(sub["fitness"].std()),
            "n": len(sub),
        }
    return result


def epistatic_deviation_distributions(
    df: pd.DataFrame,
    depths: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Raw epistatic deviation arrays at each depth (for distributional comparisons)."""
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) > 0:
            result[d] = sub["epistatic_deviation"].values
    return result


def pairwise_decomposition(
    df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
    pairwise_deviations: dict[tuple[int, int], dict[tuple[str, str], float]] | None = None,
    depths: list[int] | None = None,
) -> dict[int, float]:
    """Fraction of total epistasis variance explained by pairwise terms.

    If pairwise_deviations is None, estimates pairwise effects from the
    doubles in the dataset (depth=2 variants).

    Returns dict[depth, fraction_explained].
    """
    if depths is None:
        depths = [d for d in sorted(df["depth"].unique()) if d >= 3]

    # Estimate pairwise epistasis from doubles if not provided
    if pairwise_deviations is None:
        pairwise_deviations = _estimate_pairwise_from_doubles(df, single_site_effects)

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 30:
            continue

        total_deviations = sub["epistatic_deviation"].values
        pairwise_predictions = np.zeros(len(sub))

        for i, (_, row) in enumerate(sub.iterrows()):
            mutations = row["mutations"]
            positions = sorted(mutations.keys())
            pw_sum = 0.0
            for j, p1 in enumerate(positions):
                for p2 in positions[j + 1 :]:
                    key = (min(p1, p2), max(p1, p2))
                    aa_key = (mutations[p1], mutations[p2])
                    pw_sum += pairwise_deviations.get(key, {}).get(aa_key, 0.0)
            pairwise_predictions[i] = pw_sum

        total_var = np.var(total_deviations)
        if total_var > 0:
            residual_var = np.var(total_deviations - pairwise_predictions)
            result[d] = 1.0 - residual_var / total_var
        else:
            result[d] = 1.0

    return result


def _estimate_pairwise_from_doubles(
    df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
) -> dict[tuple[int, int], dict[tuple[str, str], float]]:
    """Estimate pairwise epistatic deviations from double mutants."""
    doubles = df[df["depth"] == 2]
    pairwise: dict[tuple[int, int], dict[tuple[str, str], float]] = {}

    for _, row in doubles.iterrows():
        mutations = row["mutations"]
        positions = sorted(mutations.keys())
        if len(positions) != 2:
            continue

        p1, p2 = positions
        a1, a2 = mutations[p1], mutations[p2]
        key = (p1, p2)
        aa_key = (a1, a2)

        if key not in pairwise:
            pairwise[key] = {}
        pairwise[key][aa_key] = row["epistatic_deviation"]

    return pairwise


def compute_all_metrics(
    df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
    depths: list[int] | None = None,
) -> dict:
    """Compute the full suite of validation metrics."""
    if depths is None:
        depths = list(range(1, 8))

    return {
        "additive_correlation": additive_correlation_profile(df, depths),
        "sign_epistasis": sign_epistasis_profile(df, single_site_effects, depths),
        "deviation_stats": epistatic_deviation_stats(df, depths),
        "fitness_moments": fitness_moments(df, depths),
        "deviation_distributions": epistatic_deviation_distributions(df, depths),
    }
