import pandas as pd
import pytest

from mutascape.benchmarks.validity import (
    measured_rank_reliability,
    pairwise_winner_transfer,
    reliability_adjusted_transfer,
)
from scripts.run_search_transfer import _transfer, _validate_scores, _with_ensemble_arms


def _scores() -> pd.DataFrame:
    rows = []
    values = {
        "measured": {"a": 0.0, "b": 1.0, "c": 2.0},
        "same": {"a": 0.1, "b": 1.1, "c": 2.1},
        "reverse": {"a": 2.0, "b": 1.0, "c": 0.0},
        "tie": {"a": 1.0, "b": 1.0, "c": 1.0},
    }
    for arm, methods in values.items():
        for method, centre in methods.items():
            for trial in range(8):
                rows.append(
                    {
                        "dataset": "example",
                        "arm": arm,
                        "method": method,
                        "trial": trial,
                        "regret": centre + 0.01 * trial,
                    }
                )
    return pd.DataFrame(rows)


def test_pairwise_winner_transfer_uses_only_reliable_real_pairs():
    result = pairwise_winner_transfer(_scores(), seed=1, n_bootstrap=200)
    by_arm = result.set_index("arm")

    assert by_arm.loc["same", "reliable_pairs"] == 3
    assert by_arm.loc["same", "winner_accuracy"] == 1.0
    assert by_arm.loc["reverse", "winner_accuracy"] == 0.0
    assert by_arm.loc["tie", "scored_pairs"] == 3
    assert by_arm.loc["tie", "winner_accuracy"] == 0.0


def test_rank_reliability_and_adjusted_transfer():
    reliability = measured_rank_reliability(_scores(), seed=1, n_splits=50)
    transfer = pd.DataFrame(
        {
            "dataset": ["example", "example"],
            "arm": ["same", "reverse"],
            "kendall_tau": [1.0, -1.0],
        }
    )

    adjusted = reliability_adjusted_transfer(transfer, reliability).set_index("arm")

    assert reliability.loc[0, "split_half_median"] == 1.0
    assert bool(adjusted.loc["same", "meets_reliability_floor"])
    assert not bool(adjusted.loc["reverse", "meets_reliability_floor"])


def test_transfer_keeps_its_schema_when_too_few_methods_can_be_ranked():
    transfer = _transfer(_scores())

    assert transfer.empty
    assert transfer.columns.tolist() == [
        "dataset",
        "arm",
        "n_methods",
        "kendall_tau",
        "spearman",
        "pearson",
    ]


def test_ensemble_rows_cover_each_primary_stochastic_comparator():
    scores = pd.DataFrame(
        {
            "dataset": ["example"] * 6,
            "protein": ["protein"] * 6,
            "method": ["search"] * 6,
            "trial": [0] * 6,
            "arm": [
                "dms_pairwise_0",
                "dms_pairwise_1",
                "potts_low_0",
                "potts_low_1",
                "rmf_high_0",
                "rmf_high_1",
            ],
            "regret": [0.1, 0.3, 0.2, 0.4, 0.5, 0.7],
        }
    )

    with_ensembles = _with_ensemble_arms(scores)
    means = with_ensembles.loc[
        with_ensembles["arm"].str.endswith("_mean"), ["arm", "regret"]
    ].set_index("arm")

    assert means.loc["dms_pairwise_mean", "regret"] == pytest.approx(0.2)
    assert means.loc["potts_low_mean", "regret"] == pytest.approx(0.3)
    assert means.loc["rmf_high_mean", "regret"] == pytest.approx(0.6)


@pytest.mark.filterwarnings("error")
def test_rank_reliability_skips_a_single_method_without_warning():
    scores = _scores().loc[lambda frame: frame["method"] == "a"]

    reliability = measured_rank_reliability(scores)

    assert reliability.empty


def test_score_validation_rejects_duplicate_trials_and_invalid_regret():
    scores = pd.DataFrame(
        {
            "dataset": ["example", "example"],
            "arm": ["measured", "measured"],
            "method": ["search", "search"],
            "trial": [0, 0],
            "best_fitness": [1.0, 1.0],
            "regret": [0.2, 1.2],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        _validate_scores(scores, expected_trials=1)

    with pytest.raises(ValueError, match="outside"):
        _validate_scores(scores.iloc[[1]].assign(trial=0), expected_trials=1)


@pytest.mark.filterwarnings("error")
def test_transfer_marks_a_constant_method_ranking_as_undefined():
    rows = []
    for arm, regrets in {
        "measured": [0.0, 0.1, 0.2, 0.3],
        "constant": [0.2, 0.2, 0.2, 0.2],
    }.items():
        for method, regret in zip(["a", "b", "c", "d"], regrets):
            rows.append(
                {
                    "dataset": "example",
                    "arm": arm,
                    "method": method,
                    "trial": 0,
                    "regret": regret,
                }
            )

    transfer = _transfer(pd.DataFrame(rows))

    assert pd.isna(transfer.loc[0, "kendall_tau"])
    assert pd.isna(transfer.loc[0, "spearman"])
    assert pd.isna(transfer.loc[0, "pearson"])
