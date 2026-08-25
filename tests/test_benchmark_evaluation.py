import numpy as np
import pandas as pd
import pytest

from mutascape.benchmarks.datasets import LandscapeDataset, LandscapeSpec
from mutascape.benchmarks.evaluation import (
    classifier_two_sample_test,
    completion_fidelity_features,
    ensemble_crps,
    interval_calibration,
    landscape_patch_features,
    maximum_mean_discrepancy,
    posterior_ranks,
    predictive_metrics,
    top_fraction_recall,
)


def _cube_dataset() -> LandscapeDataset:
    sequences = ["AA", "BA", "AB", "BB"]
    fitness = [0.0, 1.0, -0.5, 0.8]
    rows = []
    for sequence, value in zip(sequences, fitness):
        mutations = {index: aa for index, aa in enumerate(sequence, 1) if aa != "A"}
        rows.append(
            {
                "sequence": sequence,
                "fitness": value,
                "depth": len(mutations),
                "mutations": mutations,
            }
        )
    spec = LandscapeSpec(
        name="cube",
        protein="cube",
        source="ssmula",
        relative_path="unused.csv",
        fitness_column="fitness",
        sequence_column="AAs",
        depth_column="depth",
        positions=(39, 40),
        wild_type="AA",
    )
    return LandscapeDataset(
        spec, pd.DataFrame(rows), "AA", (39, 40), {39: ("A", "B"), 40: ("A", "B")}
    )


def test_predictive_metrics_are_exact_for_perfect_prediction():
    truth = np.array([-1.0, 0.0, 1.0, 2.0])

    metrics = predictive_metrics(truth, truth)

    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["mae"] == 0.0
    assert metrics["ndcg"] == pytest.approx(1.0)
    assert metrics["top_recall"] == 1.0


def test_top_fraction_recall_detects_wrong_top_variant():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    predicted = np.array([4.0, 3.0, 2.0, 1.0])

    assert top_fraction_recall(truth, predicted, fraction=0.25) == 0.0


def test_crps_is_zero_for_exact_constant_ensemble():
    truth = np.array([1.0, -2.0])
    ensemble = np.tile(truth, (10, 1))

    assert ensemble_crps(truth, ensemble) == pytest.approx([0.0, 0.0])
    assert posterior_ranks(truth, ensemble) == pytest.approx([0.5, 0.5])


def test_interval_calibration_reports_coverage_and_width():
    truth = np.array([0.0, 1.0])
    ensemble = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])

    calibration = interval_calibration(truth, ensemble, levels=(0.8,))

    assert calibration.loc[0, "coverage"] == 1.0
    assert calibration.loc[0, "mean_width"] > 0


def test_patch_features_capture_all_cube_neighbours():
    features = landscape_patch_features(_cube_dataset())

    assert features.shape == (4, 10)
    assert set(features["neighbour_count"]) == {2.0}
    assert np.isfinite(features.to_numpy()).all()


def test_completion_features_compare_truth_with_generated_fitness():
    dataset = _cube_dataset()
    observations = dataset.variants[dataset.variants["depth"] <= 1]
    hidden = dataset.variants[dataset.variants["depth"] == 2]

    real = completion_fidelity_features(observations, hidden)
    generated = completion_fidelity_features(observations, hidden, np.array([1.2]))

    assert real.shape == (1, 7)
    assert real.loc[real.index[0], "residual_z"] != generated.loc[0, "residual_z"]
    assert np.isfinite(generated.to_numpy()).all()


def test_mmd_is_zero_for_identical_samples_and_positive_for_shift():
    real = np.arange(20, dtype=float).reshape(10, 2)
    same = maximum_mean_discrepancy(real, real)
    shifted = maximum_mean_discrepancy(real, real + 10.0)

    assert same["mmd_squared"] == pytest.approx(0.0)
    assert shifted["mmd_squared"] > 0.0


def test_c2st_finds_large_distribution_shift():
    rng = np.random.default_rng(4)
    real = rng.normal(-3.0, 0.2, size=(80, 3))
    synthetic = rng.normal(3.0, 0.2, size=(80, 3))

    result = classifier_two_sample_test(
        real,
        synthetic,
        n_splits=4,
        n_permutations=10,
        seed=3,
    )

    assert result.auc > 0.99
    assert result.permutation_pvalue <= 2 / 11
