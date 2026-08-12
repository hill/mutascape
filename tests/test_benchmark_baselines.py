import pandas as pd
import pytest

from mutascape.benchmarks.baselines import RidgeLandscapeBaseline


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mutations": [{}, {1: "B"}, {2: "B"}, {1: "B", 2: "B"}],
            "fitness": [0.0, 1.0, -0.5, 1.2],
        }
    )


def test_ridge_baseline_predicts_batches():
    model = RidgeLandscapeBaseline(max_order=2, alpha=1e-8).fit(_observations())

    predicted = model.predict([{}, {1: "B", 2: "B"}])

    assert predicted.shape == (2,)
    assert predicted == pytest.approx([0.0, 1.2], abs=1e-6)


def test_ridge_baseline_requires_fit():
    with pytest.raises(RuntimeError, match="before prediction"):
        RidgeLandscapeBaseline().predict([{1: "B"}])
