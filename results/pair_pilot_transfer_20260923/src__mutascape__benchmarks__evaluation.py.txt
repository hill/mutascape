"""Evaluation metrics for sparse-to-complete landscape generation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist, pdist
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, ndcg_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mutascape.benchmarks.datasets import LandscapeDataset
from mutascape.completion import PosteriorLandscape


@dataclass(frozen=True)
class CompletionEvaluation:
    """Per-variant predictions and summary metrics for one held-out landscape."""

    predictions: pd.DataFrame
    metrics: dict[str, float]
    calibration: pd.DataFrame
    ensemble: NDArray[np.float64]


@dataclass(frozen=True)
class C2STResult:
    """Held-out classifier two-sample test result."""

    auc: float
    auc_std: float
    fold_aucs: tuple[float, ...]
    permutation_pvalue: float
    accuracy: float = float("nan")
    fold_accuracies: tuple[float, ...] = ()


def top_fraction_recall(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
    *,
    fraction: float = 0.01,
) -> float:
    """Fraction of the true top variants recovered by predicted ranking."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must share shape")
    n_top = max(1, int(np.ceil(y_true.size * fraction)))
    true_top = set(np.argpartition(y_true, -n_top)[-n_top:])
    predicted_top = set(np.argpartition(y_score, -n_top)[-n_top:])
    return len(true_top & predicted_top) / n_top


def predictive_metrics(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
    *,
    top_fraction: float = 0.01,
) -> dict[str, float]:
    """Prediction and design metrics used by CombinGym-style evaluation."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must be one-dimensional arrays of equal length")
    if y_true.size < 2:
        raise ValueError("at least two variants are required")

    if np.std(y_true) <= 1e-12 or np.std(y_score) <= 1e-12:
        spearman = float("nan")
    else:
        spearman = float(sp_stats.spearmanr(y_true, y_score).statistic)
    relevance = y_true - np.min(y_true)
    ndcg = 1.0 if np.max(relevance) == 0 else float(ndcg_score([relevance], [y_score]))
    return {
        "spearman": spearman,
        "mae": float(np.mean(np.abs(y_true - y_score))),
        "rmse": float(np.sqrt(np.mean((y_true - y_score) ** 2))),
        "wasserstein": float(sp_stats.wasserstein_distance(y_true, y_score)),
        "ndcg": ndcg,
        "top_recall": top_fraction_recall(y_true, y_score, fraction=top_fraction),
    }


def ensemble_crps(
    y_true: NDArray[np.float64],
    ensemble: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Continuous ranked probability score for finite ensemble predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    ensemble = np.asarray(ensemble, dtype=np.float64)
    if ensemble.ndim != 2 or ensemble.shape[1] != y_true.size:
        raise ValueError("ensemble must have shape (n_samples, n_variants)")
    if ensemble.shape[0] < 2:
        raise ValueError("CRPS requires at least two ensemble samples")

    first = np.mean(np.abs(ensemble - y_true[None, :]), axis=0)
    ordered = np.sort(ensemble, axis=0)
    n = ensemble.shape[0]
    weights = 2.0 * np.arange(n) - n + 1.0
    half_pairwise = np.sum(weights[:, None] * ordered, axis=0) / n**2
    return first - half_pairwise


def interval_calibration(
    y_true: NDArray[np.float64],
    ensemble: NDArray[np.float64],
    *,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> pd.DataFrame:
    """Coverage and mean interval width at each requested nominal level."""
    y_true = np.asarray(y_true, dtype=np.float64)
    ensemble = np.asarray(ensemble, dtype=np.float64)
    if ensemble.ndim != 2 or ensemble.shape[1] != y_true.size:
        raise ValueError("ensemble must have shape (n_samples, n_variants)")

    rows = []
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError("calibration levels must lie between zero and one")
        tail = (1.0 - level) / 2.0
        lower = np.quantile(ensemble, tail, axis=0)
        upper = np.quantile(ensemble, 1.0 - tail, axis=0)
        covered = (lower <= y_true) & (y_true <= upper)
        rows.append(
            {
                "nominal": level,
                "coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(upper - lower)),
            }
        )
    return pd.DataFrame(rows)


def posterior_ranks(
    y_true: NDArray[np.float64],
    ensemble: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Randomised-free fractional rank of truth within each ensemble."""
    y_true = np.asarray(y_true, dtype=np.float64)
    ensemble = np.asarray(ensemble, dtype=np.float64)
    below = np.sum(ensemble < y_true[None, :], axis=0)
    equal = np.sum(ensemble == y_true[None, :], axis=0)
    return (below + 0.5 * equal) / ensemble.shape[0]


def completion_fidelity_features(
    observations: pd.DataFrame,
    variants: pd.DataFrame,
    fitness: NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Describe observed or generated variants relative to measured singles."""
    wt_rows = observations.loc[observations["depth"] == 0, "fitness"]
    if wt_rows.empty:
        raise ValueError("observations must include the wild type")
    wt = float(wt_rows.iloc[0])
    single_effects = {
        tuple(mutations.items())[0]: float(value) - wt
        for mutations, value in observations.loc[
            observations["depth"] == 1, ["mutations", "fitness"]
        ].itertuples(index=False, name=None)
    }
    if not single_effects:
        raise ValueError("observations must include single mutants")

    values = (
        variants["fitness"].to_numpy(dtype=np.float64)
        if fitness is None
        else np.asarray(fitness, dtype=np.float64)
    )
    if values.ndim != 1 or values.size != len(variants):
        raise ValueError("fitness must match the number of variants")
    additive = np.asarray(
        [
            wt + sum(single_effects[item] for item in mutations.items())
            for mutations in variants["mutations"]
        ],
        dtype=np.float64,
    )
    effects = np.asarray(list(single_effects.values()), dtype=np.float64)
    scale = float(np.std(effects))
    if scale <= 1e-12:
        scale = 1.0
    fitness_z = (values - wt) / scale
    additive_z = (additive - wt) / scale
    residual_z = (values - additive) / scale
    maximum_depth = max(1.0, float(variants["depth"].max()))
    return pd.DataFrame(
        {
            "fitness_z": fitness_z,
            "additive_z": additive_z,
            "residual_z": residual_z,
            "fitness_squared": fitness_z**2,
            "absolute_residual": np.abs(residual_z),
            "residual_squared": residual_z**2,
            "depth_fraction": variants["depth"].to_numpy(dtype=float) / maximum_depth,
        }
    )


def evaluate_posterior(
    posterior: PosteriorLandscape,
    test: pd.DataFrame,
    *,
    n_samples: int = 100,
    seed: int = 0,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> CompletionEvaluation:
    """Evaluate posterior samples against hidden higher-order measurements."""
    variants = test["mutations"].tolist()
    truth = test["fitness"].to_numpy(dtype=float)
    ensemble = np.vstack(
        [posterior.sample(seed + index).evaluate_batch(variants) for index in range(n_samples)]
    )
    mean = np.mean(ensemble, axis=0)
    standard_deviation = np.std(ensemble, axis=0, ddof=1)
    lower = np.quantile(ensemble, 0.1, axis=0)
    upper = np.quantile(ensemble, 0.9, axis=0)
    crps = ensemble_crps(truth, ensemble)
    ranks = posterior_ranks(truth, ensemble)

    predictions = test.reset_index(drop=True).copy()
    predictions["prediction"] = mean
    predictions["posterior_std"] = standard_deviation
    predictions["lower_80"] = lower
    predictions["upper_80"] = upper
    predictions["crps"] = crps
    predictions["posterior_rank"] = ranks

    metrics = predictive_metrics(truth, mean)
    metrics["crps"] = float(np.mean(crps))
    calibration = interval_calibration(truth, ensemble, levels=levels)
    return CompletionEvaluation(predictions, metrics, calibration, ensemble)


def landscape_patch_features(dataset: LandscapeDataset) -> pd.DataFrame:
    """Build local genotype-graph features for distribution comparison."""
    variants = dataset.variants.reset_index(drop=True)
    wt_rows = variants[variants["depth"] == 0]
    if wt_rows.empty:
        raise ValueError("landscape has no wild-type row")
    wt_fitness = float(wt_rows.iloc[0]["fitness"])
    single_effects = {
        tuple(mutations.items())[0]: float(fitness) - wt_fitness
        for mutations, fitness in variants.loc[
            variants["depth"] == 1, ["mutations", "fitness"]
        ].itertuples(index=False, name=None)
    }
    sequence_fitness = dict(zip(variants["sequence"], variants["fitness"]))
    scale = float(np.std(variants["fitness"].to_numpy(dtype=float)))
    if scale <= 1e-12:
        scale = 1.0

    rows: list[dict[str, float]] = []
    full_sequence_coordinates = max(dataset.mutable_positions) <= len(dataset.wild_type_sequence)
    for sequence, mutations, fitness, depth in variants[
        ["sequence", "mutations", "fitness", "depth"]
    ].itertuples(index=False, name=None):
        additive = wt_fitness + sum(single_effects.get(item, 0.0) for item in mutations.items())
        neighbour_fitness: list[float] = []
        sequence_list = list(sequence)
        for position in dataset.mutable_positions:
            if full_sequence_coordinates:
                current = sequence_list[position - 1]
                sequence_index = position - 1
            else:
                site_index = dataset.mutable_positions.index(position)
                current = sequence_list[site_index]
                sequence_index = site_index
            for amino_acid in dataset.alphabets[position]:
                if amino_acid == current:
                    continue
                neighbour = sequence_list.copy()
                neighbour[sequence_index] = amino_acid
                neighbour_value = sequence_fitness.get("".join(neighbour))
                if neighbour_value is not None:
                    neighbour_fitness.append(float(neighbour_value))

        deltas = np.asarray(neighbour_fitness, dtype=float) - float(fitness)
        if deltas.size:
            delta_mean = float(np.mean(deltas))
            delta_std = float(np.std(deltas))
            delta_min = float(np.min(deltas))
            delta_max = float(np.max(deltas))
            uphill_fraction = float(np.mean(deltas > 0))
        else:
            delta_mean = delta_std = delta_min = delta_max = uphill_fraction = 0.0
        rows.append(
            {
                "fitness_z": (float(fitness) - wt_fitness) / scale,
                "depth_fraction": float(depth) / max(1, len(dataset.mutable_positions)),
                "additive_z": (additive - wt_fitness) / scale,
                "residual_z": (float(fitness) - additive) / scale,
                "neighbour_count": float(deltas.size),
                "delta_mean_z": delta_mean / scale,
                "delta_std_z": delta_std / scale,
                "delta_min_z": delta_min / scale,
                "delta_max_z": delta_max / scale,
                "uphill_fraction": uphill_fraction,
            }
        )
    return pd.DataFrame(rows)


def _c2st_scores(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray | None,
    *,
    n_splits: int,
    seed: int,
    classifier: str = "logistic",
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if classifier == "logistic":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elif classifier == "histogram":
        estimator = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError("classifier must be logistic or histogram")
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(features, labels)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(features, labels, groups)

    scores = []
    accuracies = []
    for train_index, test_index in splits:
        if len(np.unique(labels[test_index])) < 2 or len(np.unique(labels[train_index])) < 2:
            raise ValueError("each C2ST train and test fold needs both classes")
        model = clone(estimator).fit(features[train_index], labels[train_index])
        probability = model.predict_proba(features[test_index])[:, 1]
        scores.append(float(roc_auc_score(labels[test_index], probability)))
        accuracies.append(float(balanced_accuracy_score(labels[test_index], probability >= 0.5)))
    if not scores:
        raise ValueError("C2ST folds must contain both real and synthetic samples")
    return tuple(scores), tuple(accuracies)


def classifier_two_sample_test(
    real_features: NDArray[np.float64],
    synthetic_features: NDArray[np.float64],
    *,
    real_groups: NDArray | None = None,
    synthetic_groups: NDArray | None = None,
    n_splits: int = 5,
    n_permutations: int = 100,
    seed: int = 0,
    classifier: str = "logistic",
) -> C2STResult:
    """Measure held-out discrimination with AUC and balanced accuracy.

    Matching unique group arrays define paired samples. Permutations then swap
    labels within each pair. The grouped folds keep each pair out of training.
    With zero permutations the p-value is unavailable, rather than one.
    """
    real = np.asarray(real_features, dtype=np.float64)
    synthetic = np.asarray(synthetic_features, dtype=np.float64)
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("feature matrices must be two-dimensional with matching columns")
    if not np.isfinite(real).all() or not np.isfinite(synthetic).all():
        raise ValueError("C2ST features must be finite")
    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")
    features = np.vstack([real, synthetic])
    labels = np.concatenate(
        [np.zeros(len(real), dtype=np.int64), np.ones(len(synthetic), dtype=np.int64)]
    )
    groups = None
    paired = False
    if real_groups is not None or synthetic_groups is not None:
        if real_groups is None or synthetic_groups is None:
            raise ValueError("both real_groups and synthetic_groups are required")
        if np.shape(real_groups) != (len(real),) or np.shape(synthetic_groups) != (len(synthetic),):
            raise ValueError("group arrays must match their sample rows")
        paired = np.array_equal(real_groups, synthetic_groups) and len(
            np.unique(real_groups)
        ) == len(real)
        if not paired and len(np.intersect1d(real_groups, synthetic_groups)):
            raise ValueError("shared groups must be unique matched pairs in the same row order")
        groups = np.concatenate([real_groups, synthetic_groups])
        n_splits = min(n_splits, len(np.unique(groups)))
    else:
        n_splits = min(n_splits, len(real), len(synthetic))
    if n_splits < 2:
        raise ValueError("C2ST requires at least two cross-validation folds")

    fold_aucs, accuracies = _c2st_scores(
        features, labels, groups, n_splits=n_splits, seed=seed, classifier=classifier
    )
    auc = float(np.mean(fold_aucs))
    rng = np.random.default_rng(seed)
    null_aucs = []
    for permutation in range(n_permutations):
        if paired:
            swaps = rng.integers(0, 2, len(real))
            permuted = np.concatenate([swaps, 1 - swaps])
        elif groups is not None:
            raise ValueError("permutation tests with groups require matched pairs")
        else:
            permuted = rng.permutation(labels)
        scores, _ = _c2st_scores(
            features,
            permuted,
            groups,
            n_splits=n_splits,
            seed=seed,
            classifier=classifier,
        )
        null_aucs.append(float(np.mean(scores)))
    pvalue = (
        (1.0 + sum(value >= auc for value in null_aucs)) / (n_permutations + 1.0)
        if n_permutations
        else float("nan")
    )
    return C2STResult(
        auc,
        float(np.std(fold_aucs, ddof=1)),
        fold_aucs,
        pvalue,
        float(np.mean(accuracies)),
        accuracies,
    )


def maximum_mean_discrepancy(
    real_features: NDArray[np.float64],
    synthetic_features: NDArray[np.float64],
    *,
    bandwidth: float | None = None,
) -> dict[str, float]:
    """Biased RBF-kernel MMD squared with a median-distance bandwidth."""
    real = np.asarray(real_features, dtype=np.float64)
    synthetic = np.asarray(synthetic_features, dtype=np.float64)
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("feature matrices must be two-dimensional with matching columns")
    if bandwidth is None:
        distances = pdist(np.vstack([real, synthetic]), metric="euclidean")
        positive = distances[distances > 0]
        bandwidth = float(np.median(positive)) if positive.size else 1.0
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    gamma = 1.0 / (2.0 * bandwidth**2)
    kernel_xx = np.exp(-gamma * cdist(real, real, metric="sqeuclidean"))
    kernel_yy = np.exp(-gamma * cdist(synthetic, synthetic, metric="sqeuclidean"))
    kernel_xy = np.exp(-gamma * cdist(real, synthetic, metric="sqeuclidean"))
    mmd_squared = float(np.mean(kernel_xx) + np.mean(kernel_yy) - 2.0 * np.mean(kernel_xy))
    return {"mmd_squared": max(0.0, mmd_squared), "bandwidth": bandwidth}


def per_feature_c2st(
    real_features: pd.DataFrame,
    synthetic_features: pd.DataFrame,
    *,
    groups: NDArray | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Score each feature on its own, to locate what separates the samples.

    A single AUC says two samples differ. It does not say which property
    differs, and the fix depends entirely on that. Scoring one column at a time
    against the same folds attributes the separation to a feature the model can
    be changed to address.
    """
    if list(real_features.columns) != list(synthetic_features.columns):
        raise ValueError("both feature frames must carry the same columns")
    scores: dict[str, float] = {}
    for column in real_features.columns:
        left = real_features[[column]].to_numpy(dtype=np.float64)
        right = synthetic_features[[column]].to_numpy(dtype=np.float64)
        # A constant column carries no signal and would divide by zero in the
        # scaler, so it scores as chance rather than raising.
        if np.allclose(left, right):
            scores[column] = 0.5
            continue
        result = classifier_two_sample_test(
            left,
            right,
            real_groups=groups,
            synthetic_groups=groups,
            n_splits=n_splits,
            n_permutations=0,
            seed=seed,
        )
        scores[column] = result.auc
    return scores
