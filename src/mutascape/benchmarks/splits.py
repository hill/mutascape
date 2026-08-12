"""Leakage-resistant splits for sparse-to-complete landscape evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mutascape.benchmarks.datasets import LandscapeDataset


@dataclass(frozen=True)
class MutationOrderSplit:
    """Observed and hidden variants for one mutation-order threshold."""

    train: pd.DataFrame
    test: pd.DataFrame
    max_train_depth: int


@dataclass(frozen=True)
class ProteinHoldout:
    """Complete datasets grouped into training and held-out proteins."""

    train: tuple[LandscapeDataset, ...]
    test: tuple[LandscapeDataset, ...]
    held_out_protein: str


def mutation_order_split(
    variants: pd.DataFrame,
    max_train_depth: int = 1,
) -> MutationOrderSplit:
    """Train at or below one mutation depth and test on every higher order."""
    if max_train_depth < 0:
        raise ValueError("max_train_depth must be non-negative")
    if "depth" not in variants:
        raise ValueError("variants must contain a depth column")

    depths = pd.to_numeric(variants["depth"], errors="raise")
    train = variants.loc[depths <= max_train_depth].copy().reset_index(drop=True)
    test = variants.loc[depths > max_train_depth].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError("mutation-order split produced an empty training set")
    if test.empty:
        raise ValueError("mutation-order split produced an empty test set")
    return MutationOrderSplit(train, test, max_train_depth)


def sample_observation_budget(
    variants: pd.DataFrame,
    budget: int,
    *,
    seed: int = 0,
    anchor_depth: int = 1,
) -> pd.DataFrame:
    """Keep WT and singles, then sample a fixed budget from higher orders.

    ``budget`` counts all observations, including the anchored rows. A budget
    smaller than the anchor set is invalid because it would discard measured
    WT or single-mutant effects.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    depths = pd.to_numeric(variants["depth"], errors="raise")
    anchors = variants.loc[depths <= anchor_depth]
    if len(anchors) > budget:
        raise ValueError(f"budget {budget} is smaller than the {len(anchors)} anchor variants")

    remainder = variants.loc[depths > anchor_depth]
    n_extra = min(budget - len(anchors), len(remainder))
    if n_extra:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(remainder.index.to_numpy(), size=n_extra, replace=False)
        sampled = pd.concat([anchors, remainder.loc[chosen]])
    else:
        sampled = anchors
    return sampled.sort_index().reset_index(drop=True)


def iter_protein_holdouts(datasets: Sequence[LandscapeDataset]) -> Iterator[ProteinHoldout]:
    """Yield leave-one-protein-out folds without splitting related assays."""
    proteins = sorted({dataset.protein for dataset in datasets})
    if len(proteins) < 2:
        raise ValueError("protein holdout requires at least two distinct proteins")

    for protein in proteins:
        train = tuple(dataset for dataset in datasets if dataset.protein != protein)
        test = tuple(dataset for dataset in datasets if dataset.protein == protein)
        yield ProteinHoldout(train=train, test=test, held_out_protein=protein)
