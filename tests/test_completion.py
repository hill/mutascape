import numpy as np
import pandas as pd
import pytest

from mutascape.benchmarks.datasets import LandscapeDataset, LandscapeSpec
from mutascape.completion import (
    LandscapePrior,
    PosteriorLandscape,
    interaction_coefficients,
)


def _binary_dataset(name: str, protein: str, pair_effect: float) -> LandscapeDataset:
    wt = "AA"
    rows = []
    for sequence in ["AA", "BA", "AB", "BB"]:
        mutations = {index: aa for index, aa in enumerate(sequence, 1) if aa != "A"}
        fitness = sum(1.0 if position == 1 else -0.5 for position in mutations)
        if sequence == "BB":
            fitness += pair_effect
        rows.append(
            {
                "sequence": sequence,
                "fitness": fitness,
                "depth": len(mutations),
                "mutations": mutations,
            }
        )
    spec = LandscapeSpec(
        name=name,
        protein=protein,
        source="ssmula",
        relative_path="unused.csv",
        fitness_column="fitness",
        sequence_column="AAs",
        depth_column="depth",
        positions=(1, 2),
        wild_type=wt,
    )
    return LandscapeDataset(spec, pd.DataFrame(rows), wt, (1, 2), {1: ("A", "B"), 2: ("A", "B")})


def test_interaction_coefficients_recover_pair_effect():
    dataset = _binary_dataset("one", "p1", pair_effect=0.75)

    coefficients = interaction_coefficients(dataset.variants, dataset.wild_type_sequence)

    assert sorted(coefficients[1]) == [-0.5, 1.0]
    assert coefficients[2] == pytest.approx([0.75])


def test_interaction_coefficients_require_all_subvariants():
    dataset = _binary_dataset("one", "p1", pair_effect=0.75)
    incomplete = dataset.variants[dataset.variants["sequence"] != "AB"]

    coefficients = interaction_coefficients(incomplete, dataset.wild_type_sequence)

    assert coefficients[2] == []


def test_prior_records_training_scope_and_round_trips():
    datasets = [_binary_dataset("one", "p1", 0.5), _binary_dataset("two", "p2", -0.25)]

    prior = LandscapePrior.fit(datasets, max_order=2)
    restored = LandscapePrior.from_dict(prior.to_dict())

    assert prior.n_landscapes == 2
    assert prior.n_proteins == 2
    assert restored == prior
    assert prior.order_scale[2] > 0


def test_posterior_preserves_observed_singles_and_doubles():
    training = [_binary_dataset("one", "p1", 0.5), _binary_dataset("two", "p2", -0.25)]
    target = _binary_dataset("target", "target", 0.8)
    prior = LandscapePrior.fit(training, max_order=2)

    posterior = PosteriorLandscape.condition(
        target.wild_type_sequence,
        target.variants,
        prior,
    )
    oracle = posterior.sample(3)

    for mutations, fitness in target.variants[["mutations", "fitness"]].itertuples(
        index=False, name=None
    ):
        assert oracle.evaluate(mutations) == pytest.approx(fitness)


def test_sampled_oracle_is_stable_and_varies_between_samples():
    training = [_binary_dataset("one", "p1", 0.5), _binary_dataset("two", "p2", -0.25)]
    target = _binary_dataset("target", "target", 0.8)
    prior = LandscapePrior.fit(training, max_order=2)
    singles = target.variants[target.variants["depth"] <= 1]
    posterior = PosteriorLandscape.condition(target.wild_type_sequence, singles, prior)
    variant = {1: "B", 2: "B"}

    first = posterior.sample(11)
    repeated = posterior.sample(11)
    second = posterior.sample(12)

    assert first.evaluate(variant) == repeated.evaluate(variant)
    assert first.evaluate(variant) != second.evaluate(variant)


def test_predict_returns_ordered_intervals():
    training = [_binary_dataset("one", "p1", 0.5), _binary_dataset("two", "p2", -0.25)]
    target = _binary_dataset("target", "target", 0.8)
    prior = LandscapePrior.fit(training, max_order=2)
    singles = target.variants[target.variants["depth"] <= 1]
    posterior = PosteriorLandscape.condition(target.wild_type_sequence, singles, prior)

    prediction = posterior.predict([{1: "B", 2: "B"}], n_samples=20, seed=4)

    assert list(prediction.columns) == ["mean", "std", "lower", "upper"]
    assert prediction.loc[0, "lower"] <= prediction.loc[0, "mean"]
    assert prediction.loc[0, "mean"] <= prediction.loc[0, "upper"]
    assert prediction.loc[0, "std"] > 0


def test_whole_protein_query_requires_measured_single_effects():
    training = [_binary_dataset("one", "p1", 0.5), _binary_dataset("two", "p2", -0.25)]
    target = _binary_dataset("target", "target", 0.8)
    prior = LandscapePrior.fit(training, max_order=2)
    singles = target.variants[target.variants["depth"] <= 1]
    posterior = PosteriorLandscape.condition(target.wild_type_sequence, singles, prior)

    with pytest.raises(ValueError, match="missing single-mutant"):
        posterior.sample(0).evaluate({1: "C"})


def test_batch_evaluation_shape():
    training = [_binary_dataset("one", "p1", 0.5), _binary_dataset("two", "p2", -0.25)]
    target = _binary_dataset("target", "target", 0.8)
    prior = LandscapePrior.fit(training, max_order=2)
    singles = target.variants[target.variants["depth"] <= 1]
    posterior = PosteriorLandscape.condition(target.wild_type_sequence, singles, prior)

    values = posterior.sample(1).evaluate_batch([{}, {1: "B"}, {1: "B", 2: "B"}])

    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)


def test_assay_range_clipping_is_explicit():
    training = [_binary_dataset("one", "p1", 0.5), _binary_dataset("two", "p2", -0.25)]
    target = _binary_dataset("target", "target", 0.8)
    prior = LandscapePrior.fit(training, max_order=2)
    singles = target.variants[target.variants["depth"] <= 1]
    posterior = PosteriorLandscape.condition(target.wild_type_sequence, singles, prior)
    variant = {1: "B", 2: "B"}

    latent = posterior.sample(100).evaluate(variant)
    clipped = posterior.sample(100, clip_to_assay_range=True).evaluate(variant)

    lower = min(singles["fitness"].min(), prior.lower_bound * posterior.target_scale)
    upper = max(singles["fitness"].max(), prior.upper_bound * posterior.target_scale)
    assert lower <= clipped <= upper
    if latent < lower or latent > upper:
        assert clipped != latent
