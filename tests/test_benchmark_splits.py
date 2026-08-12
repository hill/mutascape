import pandas as pd
import pytest

from mutascape.benchmarks.datasets import LandscapeDataset, LandscapeSpec
from mutascape.benchmarks.splits import (
    iter_protein_holdouts,
    mutation_order_split,
    sample_observation_budget,
)


def _dataset(name: str, protein: str) -> LandscapeDataset:
    spec = LandscapeSpec(
        name=name,
        protein=protein,
        source="ssmula",
        relative_path="unused.csv",
        fitness_column="fitness",
        sequence_column="AAs",
        depth_column="depth",
        positions=(1,),
        wild_type="A",
    )
    variants = pd.DataFrame({"sequence": ["A"], "fitness": [0.0], "depth": [0], "mutations": [{}]})
    return LandscapeDataset(spec, variants, "A", (1,), {1: ("A",)})


def test_mutation_order_split_has_no_depth_overlap():
    variants = pd.DataFrame({"depth": [0, 1, 1, 2, 3], "fitness": range(5)})

    split = mutation_order_split(variants, max_train_depth=1)

    assert split.train["depth"].max() == 1
    assert split.test["depth"].min() == 2
    assert len(split.train) + len(split.test) == len(variants)


def test_sample_budget_keeps_anchors_and_is_deterministic():
    variants = pd.DataFrame(
        {
            "sequence": list("abcdefgh"),
            "depth": [0, 1, 1, 2, 2, 3, 3, 4],
        }
    )

    first = sample_observation_budget(variants, 5, seed=7)
    second = sample_observation_budget(variants, 5, seed=7)

    assert first.equals(second)
    assert set("abc").issubset(first["sequence"])
    assert len(first) == 5


def test_sample_budget_rejects_budget_below_anchor_count():
    variants = pd.DataFrame({"depth": [0, 1, 1, 2]})

    with pytest.raises(ValueError, match="anchor variants"):
        sample_observation_budget(variants, 2)


def test_protein_holdout_keeps_related_assays_together():
    datasets = [_dataset("a-1", "a"), _dataset("a-2", "a"), _dataset("b-1", "b")]

    folds = list(iter_protein_holdouts(datasets))
    a_fold = next(fold for fold in folds if fold.held_out_protein == "a")

    assert {dataset.name for dataset in a_fold.test} == {"a-1", "a-2"}
    assert {dataset.protein for dataset in a_fold.train} == {"b"}
