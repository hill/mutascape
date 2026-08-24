import pandas as pd

from mutascape.benchmarks.validity import (
    measured_rank_reliability,
    pairwise_winner_transfer,
    reliability_adjusted_transfer,
)


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
