"""Statistical comparison between real and synthetic landscape metrics."""

import numpy as np
from scipy import stats as sp_stats


def profile_rmse(
    real_profile: dict[int, float],
    synth_profile: dict[int, float],
) -> float:
    """RMSE between two depth-resolved metric profiles.

    Only compares depths present in both profiles.
    """
    common = sorted(set(real_profile) & set(synth_profile))
    if not common:
        return float("inf")
    diffs = [real_profile[d] - synth_profile[d] for d in common]
    return float(np.sqrt(np.mean(np.array(diffs) ** 2)))


def distributional_comparison(
    real_values: np.ndarray,
    synth_values: np.ndarray,
) -> dict[str, float]:
    """Compare two distributions using multiple statistics.

    Returns dict with ks_statistic, ks_pvalue, wasserstein, mean_diff, std_diff.
    """
    ks_stat, ks_p = sp_stats.ks_2samp(real_values, synth_values)
    wasserstein = float(sp_stats.wasserstein_distance(real_values, synth_values))

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "wasserstein": wasserstein,
        "mean_diff": float(np.mean(synth_values) - np.mean(real_values)),
        "std_diff": float(np.std(synth_values) - np.std(real_values)),
    }


def bootstrap_ci(
    data: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a scalar metric.

    Parameters
    data : array of values to resample
    metric_fn : callable that takes an array and returns a scalar
    n_boot : number of bootstrap replicates
    alpha : significance level (0.05 = 95% CI)
    seed : random seed

    Returns (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    point = metric_fn(data)
    boot_values = np.array(
        [metric_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    )
    lower = float(np.percentile(boot_values, 100 * alpha / 2))
    upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
    return float(point), lower, upper


def composite_validation_score(
    real_metrics: dict,
    synth_metrics: dict,
    weights: dict[str, float] | None = None,
) -> float:
    """Composite score measuring how well synthetic data matches real data.

    Lower is better. Combines RMSE of depth-resolved profiles with
    mean Wasserstein distance of epistatic deviation distributions.

    Parameters
    real_metrics : output of compute_all_metrics on real data
    synth_metrics : output of compute_all_metrics on synthetic data
    weights : optional weight overrides

    Returns composite score (lower = better match).
    """
    if weights is None:
        weights = {
            "additive_correlation": 0.4,
            "sign_epistasis": 0.2,
            "wasserstein": 0.2,
            "fitness_mean": 0.2,
        }

    score = 0.0

    # Additive correlation profile RMSE
    if "additive_correlation" in weights:
        rmse = profile_rmse(
            real_metrics["additive_correlation"],
            synth_metrics["additive_correlation"],
        )
        score += weights["additive_correlation"] * rmse

    # Sign epistasis profile RMSE
    if "sign_epistasis" in weights:
        rmse = profile_rmse(
            real_metrics["sign_epistasis"],
            synth_metrics["sign_epistasis"],
        )
        score += weights["sign_epistasis"] * rmse

    # Mean Wasserstein distance across depths
    if "wasserstein" in weights:
        real_dists = real_metrics["deviation_distributions"]
        synth_dists = synth_metrics["deviation_distributions"]
        common_depths = sorted(set(real_dists) & set(synth_dists))
        if common_depths:
            w_dists = [
                sp_stats.wasserstein_distance(real_dists[d], synth_dists[d]) for d in common_depths
            ]
            score += weights["wasserstein"] * float(np.mean(w_dists))

    # Fitness mean profile RMSE
    if "fitness_mean" in weights:
        real_means = {d: v["mean"] for d, v in real_metrics["fitness_moments"].items()}
        synth_means = {d: v["mean"] for d, v in synth_metrics["fitness_moments"].items()}
        rmse = profile_rmse(real_means, synth_means)
        score += weights["fitness_mean"] * rmse

    return score
