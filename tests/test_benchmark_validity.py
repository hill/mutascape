import pandas as pd
import pytest

from mutascape.benchmarks.datasets import (
    GRAPHFLA_EXTERNAL_MANIFEST,
    LandscapeDataset,
    LandscapeSpec,
)
from mutascape.benchmarks.validity import (
    measured_rank_reliability,
    pairwise_winner_transfer,
    reliability_adjusted_transfer,
)
from scripts.freeze_benchmark_validity import _arm_metrics, _reliability_floor
from scripts.run_search_transfer import (
    _include_posterior_baseline,
    _target_eligibility,
    _target_sources,
    _transfer,
    _validate_scores,
    _with_ensemble_arms,
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


def _binary_dataset() -> LandscapeDataset:
    sequences = ["AAA", "CAA", "ACA", "AAC", "CCA"]
    fitness = [0.0, 1.0, -1.0, 0.5, 0.3]
    variants = pd.DataFrame(
        {
            "sequence": sequences,
            "fitness": fitness,
            "depth": [0, 1, 1, 1, 2],
            "mutations": [
                {},
                {1: "C"},
                {2: "C"},
                {3: "C"},
                {1: "C", 2: "C"},
            ],
        }
    )
    spec = LandscapeSpec(
        name="toy",
        protein="toy-protein",
        source="combingym",
        relative_path="toy.csv",
        fitness_column="fitness",
        sequence_column="sequence",
        depth_column="depth",
    )
    return LandscapeDataset(
        spec=spec,
        variants=variants,
        wild_type_sequence="AAA",
        mutable_positions=(1, 2, 3),
        alphabets={1: ("A", "C"), 2: ("A", "C"), 3: ("A", "C")},
    )


def _write_transfer_run(root) -> None:
    pd.DataFrame(
        {
            "dataset": ["trpb-a", "trpb-b", "gb1"],
            "protein": ["TrpB", "TrpB", "GB1"],
            "arm": ["example"] * 3,
            "method": ["search"] * 3,
            "trial": [0] * 3,
            "regret": [0.0] * 3,
        }
    ).to_csv(root / "scores.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["trpb-a", "trpb-b", "gb1"],
            "arm": ["example"] * 3,
            "kendall_tau": [1.0, 1.0, 0.0],
        }
    ).to_csv(root / "transfer.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["trpb-a", "trpb-b", "gb1"],
            "arm": ["example"] * 3,
            "correct_pairs": [9, 9, 0],
            "scored_pairs": [10, 10, 10],
        }
    ).to_csv(root / "pairwise_transfer.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["trpb-a", "trpb-b", "gb1"],
            "arm": ["example"] * 3,
            "meets_reliability_floor": [True, True, False],
        }
    ).to_csv(root / "reliability_adjusted_transfer.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["trpb-a", "trpb-b", "gb1"],
            "split_half_lower": [0.9, 0.8, 0.1],
        }
    ).to_csv(root / "measured_reliability.csv", index=False)


def test_pairwise_winner_transfer_uses_only_reliable_real_pairs():
    result = pairwise_winner_transfer(_scores(), seed=1, n_bootstrap=200)
    by_arm = result.set_index("arm")

    assert by_arm.loc["same", "reliable_pairs"] == 3
    assert by_arm.loc["same", "winner_accuracy"] == 1.0
    assert by_arm.loc["reverse", "winner_accuracy"] == 0.0
    assert by_arm.loc["tie", "scored_pairs"] == 3
    assert by_arm.loc["tie", "winner_accuracy"] == 0.0


def test_target_sources_can_select_combingym_or_both_collections():
    assert _target_sources("combingym") == {"combingym"}
    assert _target_sources("all") == {"combingym", "ssmula"}


def test_graphfla_manifest_names_the_assayed_proteins():
    proteins = {spec.name: spec.protein for spec in GRAPHFLA_EXTERNAL_MANIFEST}

    assert proteins["GraphFLA_Lunzer2005_fitness"] == "IMDH"
    assert proteins["GraphFLA_Michael2024_LamB"] == "phage-lambda-J"
    assert proteins["GraphFLA_Schulz2025"] == "COV107-23"
    assert proteins["GraphFLA_Poelwijk2019"] == "eqFP611"


def test_target_eligibility_applies_supported_domain_coverage():
    included = _target_eligibility(_binary_dataset(), minimum_coverage=0.5, minimum_candidates=4)
    excluded = _target_eligibility(_binary_dataset(), minimum_coverage=0.75, minimum_candidates=4)

    assert included["supported_domain"] == 8
    assert included["finite_candidates"] == 5
    assert included["coverage"] == pytest.approx(0.625)
    assert included["included"]
    assert not excluded["included"]
    assert excluded["reason"] == "coverage below 0.750"


def test_frozen_metrics_give_each_protein_one_weight(tmp_path):
    _write_transfer_run(tmp_path)

    metrics = _arm_metrics(tmp_path, "example")

    assert metrics["kendallTau"] == pytest.approx(0.5)
    assert metrics["pairwiseAccuracy"] == pytest.approx(0.45)
    assert metrics["reliabilityPassRate"] == pytest.approx(0.5)
    assert _reliability_floor(tmp_path) == pytest.approx(0.475)


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


def test_primary_protocol_does_not_build_the_exploratory_prior():
    assert not _include_posterior_baseline("primary")
    assert _include_posterior_baseline("full")
    assert _include_posterior_baseline(None)


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
