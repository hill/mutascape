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


def tensor_pearson_r(
    real_tensor: np.ndarray,
    synth_tensor: np.ndarray,
) -> float:
    """Pearson correlation between real and synthetic fitness tensors.

    Tensors must share shape. Only matched positions where both entries are
    finite contribute; NaNs in either are skipped. Returns NaN if fewer than
    two overlapping finite entries or if either side has zero variance.

    This is the single-number answer to "does the synthetic landscape predict
    the right fitness for each variant," independent of any distributional
    or structural objective.
    """
    if real_tensor.shape != synth_tensor.shape:
        raise ValueError(
            f"tensor shape mismatch: real={real_tensor.shape}, synth={synth_tensor.shape}"
        )
    mask = np.isfinite(real_tensor) & np.isfinite(synth_tensor)
    if int(mask.sum()) < 2:
        return float("nan")
    r = real_tensor[mask].ravel()
    s = synth_tensor[mask].ravel()
    if np.std(r) == 0 or np.std(s) == 0:
        return float("nan")
    return float(np.corrcoef(r, s)[0, 1])


def fitness_wasserstein(
    real_tensor: np.ndarray,
    synth_tensor: np.ndarray,
) -> float:
    """Wasserstein-1 distance between the marginal fitness distributions of
    the real and synthetic tensors.

    Ignores NaNs on either side. Measures "shape match" independently of any
    per-variant correspondence; complements `tensor_pearson_r` which measures
    per-variant match.
    """
    r = real_tensor[np.isfinite(real_tensor)].ravel()
    s = synth_tensor[np.isfinite(synth_tensor)].ravel()
    if r.size == 0 or s.size == 0:
        return float("nan")
    return float(sp_stats.wasserstein_distance(r, s))


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
