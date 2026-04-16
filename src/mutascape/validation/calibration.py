"""Alpha calibration: find the epistatic strength that best matches real data."""

from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from mutascape import ContactMap, Landscape, RandomEpistasis, SignConsistentEpistasis

from .comparisons import composite_validation_score
from .metrics import compute_all_metrics
from .real_data import sample_synthetic_data


def build_landscape(
    dms_csv: str | Path,
    distances_npz: str | Path,
    wt_fasta: str | Path,
    alpha: float = 1.0,
    sigma: float = 0.3,
    seed: int = 42,
    epistasis_model: str = "random",
    threshold: float = 8.0,
) -> Landscape:
    """Build a landscape from DMS data and a precomputed distance matrix."""
    distances = np.load(distances_npz)
    contact_map = ContactMap.from_distance_matrix(
        distances["distances"],
        distances["positions"].tolist(),
        threshold=threshold,
    )

    if epistasis_model == "sign_consistent":
        epistasis = SignConsistentEpistasis(sigma=sigma, seed=seed)
    else:
        epistasis = RandomEpistasis(sigma=sigma, seed=seed)

    wt_seq = str(next(SeqIO.parse(str(wt_fasta), "fasta")).seq)

    return Landscape.from_dms(
        dms_path=dms_csv,
        contact_map=contact_map,
        epistasis=epistasis,
        alpha=alpha,
        seed=seed,
        wt_sequence=wt_seq,
    )


def alpha_sweep(
    real_metrics: dict,
    single_site_effects: dict[int, dict[str, float]],
    build_fn,
    alphas: list[float] | None = None,
    depths: list[int] | None = None,
    n_per_depth: int = 3000,
) -> pd.DataFrame:
    """Sweep alpha values and compute validation scores.

    Parameters
    build_fn : callable(alpha) -> Landscape
        Function that builds a landscape at the given alpha.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    if depths is None:
        depths = list(range(2, 8))

    results = []

    for alpha in alphas:
        print(f"  alpha={alpha:.2f}", end="", flush=True)

        landscape = build_fn(alpha)
        synth_df = sample_synthetic_data(landscape, depths, n_per_depth=n_per_depth)
        synth_metrics = compute_all_metrics(synth_df, single_site_effects, depths)

        score = composite_validation_score(real_metrics, synth_metrics)

        row = {
            "alpha": alpha,
            "composite_score": score,
        }
        for d in depths:
            if d in synth_metrics["additive_correlation"]:
                row[f"additive_corr_d{d}"] = synth_metrics["additive_correlation"][d]
            if d in synth_metrics["sign_epistasis"]:
                row[f"sign_epi_d{d}"] = synth_metrics["sign_epistasis"][d]

        results.append(row)
        print(f" -> score={score:.4f}")

    return pd.DataFrame(results)


def find_optimal_alpha(sweep_df: pd.DataFrame) -> float:
    """Return the alpha with the lowest composite score."""
    best_idx = sweep_df["composite_score"].idxmin()
    return float(sweep_df.loc[best_idx, "alpha"])


def refine_alpha(
    real_metrics: dict,
    single_site_effects: dict[int, dict[str, float]],
    build_fn,
    coarse_best: float,
    step: float = 0.05,
    half_range: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    """Fine-grained alpha search around a coarse optimum."""
    fine_alphas = np.arange(
        max(0.0, coarse_best - half_range),
        coarse_best + half_range + step / 2,
        step,
    ).tolist()
    return alpha_sweep(
        real_metrics,
        single_site_effects,
        build_fn,
        alphas=fine_alphas,
        **kwargs,
    )


def cross_validate_alpha(
    real_df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
    build_fn,
    alphas: list[float] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    depths: list[int] | None = None,
    n_per_depth: int = 3000,
) -> dict:
    """Cross-validate alpha selection.

    Splits real data into train/test, calibrates on train, evaluates on test.

    Parameters
    build_fn : callable(alpha) -> Landscape
    """
    if depths is None:
        depths = list(range(2, 8))

    rng = np.random.default_rng(seed)

    best_alphas = []
    test_scores = []

    for fold in range(n_folds):
        print(f"\nFold {fold + 1}/{n_folds}")

        train_indices = []
        test_indices = []
        for d in depths:
            depth_idx = real_df[real_df["depth"] == d].index.values
            rng.shuffle(depth_idx)
            split = int(0.7 * len(depth_idx))
            train_indices.extend(depth_idx[:split])
            test_indices.extend(depth_idx[split:])

        train_df = real_df.loc[train_indices]
        test_df = real_df.loc[test_indices]

        train_metrics = compute_all_metrics(train_df, single_site_effects, depths)
        test_metrics = compute_all_metrics(test_df, single_site_effects, depths)

        sweep = alpha_sweep(
            train_metrics,
            single_site_effects,
            build_fn,
            alphas=alphas,
            depths=depths,
            n_per_depth=n_per_depth,
        )
        best_alpha = find_optimal_alpha(sweep)
        best_alphas.append(best_alpha)

        landscape = build_fn(best_alpha)
        synth_df = sample_synthetic_data(landscape, depths, n_per_depth=n_per_depth)
        synth_metrics = compute_all_metrics(synth_df, single_site_effects, depths)
        test_score = composite_validation_score(test_metrics, synth_metrics)
        test_scores.append(test_score)

        print(f"  best_alpha={best_alpha:.2f}, test_score={test_score:.4f}")

    return {
        "best_alphas": best_alphas,
        "test_scores": test_scores,
        "mean_alpha": float(np.mean(best_alphas)),
        "std_alpha": float(np.std(best_alphas)),
        "mean_test_score": float(np.mean(test_scores)),
        "std_test_score": float(np.std(test_scores)),
    }
