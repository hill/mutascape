import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from mutascape.benchmarks.figures import (
    plot_calibration_curve,
    plot_completion_predictions,
    plot_metric_fingerprint,
    plot_optimizer_rank_transfer,
    plot_posterior_rank_histogram,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_metric_fingerprint():
    scores = pd.DataFrame(
        {
            "model": ["additive", "posterior"] * 2,
            "metric": ["mae", "mae", "crps", "crps"],
            "value": [0.6, 0.3, 0.7, 0.4],
        }
    )

    assert isinstance(plot_metric_fingerprint(scores), Figure)


def test_plot_calibration_curve():
    calibration = pd.DataFrame({"nominal": [0.5, 0.8], "coverage": [0.45, 0.75]})

    assert isinstance(plot_calibration_curve(calibration), Figure)


def test_plot_completion_predictions_and_ranks():
    predictions = pd.DataFrame(
        {
            "fitness": [0.0, 0.5, 1.0],
            "prediction": [0.1, 0.4, 0.9],
            "depth": [2, 2, 3],
            "posterior_rank": [0.2, 0.5, 0.8],
        }
    )

    assert isinstance(plot_completion_predictions(predictions), Figure)
    assert isinstance(plot_posterior_rank_histogram(predictions), Figure)


def test_plot_optimizer_rank_transfer():
    results = pd.DataFrame(
        {
            "landscape": ["a"] * 6,
            "oracle": ["real"] * 3 + ["synthetic"] * 3,
            "optimiser": ["random", "greedy", "ridge"] * 2,
            "score": [0.2, 0.8, 0.7, 0.3, 0.75, 0.6],
        }
    )

    assert isinstance(plot_optimizer_rank_transfer(results), Figure)
