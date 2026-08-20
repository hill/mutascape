"""Invariants for the transferable prior and the saturating readout."""

import numpy as np
import pandas as pd
import pytest

from mutascape.benchmarks.datasets import LandscapeDataset, LandscapeSpec
from mutascape.benchmarks.splits import has_informative_singles, single_effect_summary
from mutascape.benchmarks.topology import measured_tensor, summarise, supported_cube
from mutascape.completion import LandscapePrior, PosteriorLandscape, _single_effect_scale

WILD_TYPE = "ACDE"
POSITIONS = (1, 2, 3, 4)


def _landscape(name, effects, interaction=0.0, seed=0):
    """Two-site landscape with a controllable pairwise interaction."""
    rng = np.random.default_rng(seed)
    rows = [{"mutations": {}, "fitness": 0.0, "depth": 0}]
    for (position, amino_acid), effect in effects.items():
        rows.append({"mutations": {position: amino_acid}, "fitness": effect, "depth": 1})
    keys = list(effects)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            if left[0] == right[0]:
                continue
            rows.append(
                {
                    "mutations": {left[0]: left[1], right[0]: right[1]},
                    "fitness": effects[left] + effects[right] + interaction * rng.normal(),
                    "depth": 2,
                }
            )
    spec = LandscapeSpec(
        name=name,
        protein=name,
        source="ssmula",
        relative_path=f"data/{name}/scale2max/{name}.csv",
        fitness_column="fitness",
        sequence_column="AAs",
        depth_column="n_mut",
        positions=POSITIONS,
    )
    alphabets = {position: tuple(sorted({WILD_TYPE[position - 1]})) for position in POSITIONS}
    return LandscapeDataset(spec, pd.DataFrame(rows), WILD_TYPE, POSITIONS, alphabets)


def _effects(scale):
    return {
        (1, "G"): -0.4 * scale,
        (2, "V"): 0.3 * scale,
        (3, "W"): -0.2 * scale,
        (4, "K"): 0.5 * scale,
    }


class TestSingleEffectScale:
    def test_uses_standard_deviation_not_the_interquartile_range(self):
        # Most single mutations are neutral, so a distribution whose middle half
        # sits at zero must still produce a usable scale.
        mostly_neutral = np.array([0.0] * 8 + [-2.0, 2.0])
        assert _single_effect_scale(mostly_neutral) > 0.5

    def test_empty_input_is_zero(self):
        assert _single_effect_scale(np.array([])) == 0.0


class TestDepthShrinkage:
    def test_disagreeing_proteins_lose_the_depth_shift(self):
        # Same interaction magnitude, opposite depth behaviour: the pooled shift
        # should not survive transfer to an unseen protein.
        datasets = []
        for index in range(4):
            sign = 1.0 if index % 2 == 0 else -1.0
            frame = _landscape(f"P{index}", _effects(1.0), seed=index)
            frame.variants.loc[frame.variants["depth"] == 2, "fitness"] += sign * 3.0
            datasets.append(frame)
        prior = LandscapePrior.fit(datasets, max_order=2, global_epistasis=False)
        assert prior.depth_shrinkage[2] < 0.2
        assert abs(prior.depth_location[2]) < abs(prior.depth_between[2])

    def test_a_depth_reached_by_too_few_landscapes_is_zeroed(self):
        datasets = [_landscape(f"P{i}", _effects(1.0), seed=i) for i in range(3)]
        deep = _landscape("Deep", _effects(1.0), seed=9)
        deep.variants.loc[len(deep.variants)] = {
            "mutations": {1: "G", 2: "V", 3: "W"},
            "fitness": -9.0,
            "depth": 3,
        }
        prior = LandscapePrior.fit([*datasets, deep], max_order=2, global_epistasis=False)
        assert prior.depth_shrinkage.get(3, 0.0) == 0.0
        assert prior.depth_location.get(3, 0.0) == 0.0


class TestReadout:
    def test_singles_survive_the_preimage_round_trip(self):
        datasets = [_landscape(f"P{i}", _effects(1.0), interaction=0.05, seed=i) for i in range(3)]
        prior = LandscapePrior.fit(datasets, max_order=2)
        target = _landscape("T", _effects(1.0), interaction=0.05, seed=7)
        train = target.variants[target.variants["depth"] <= 1]
        posterior = PosteriorLandscape.condition(WILD_TYPE, train, prior)
        assert posterior.nonlinearity is not None
        for mutation, effect in posterior.single_effects.items():
            recovered = float(posterior.nonlinearity.forward(posterior.preimage_effects[mutation]))
            assert recovered == pytest.approx(effect, abs=1e-6)

    def test_observed_genotypes_are_returned_exactly(self):
        datasets = [_landscape(f"P{i}", _effects(1.0), interaction=0.05, seed=i) for i in range(3)]
        prior = LandscapePrior.fit(datasets, max_order=2)
        target = _landscape("T", _effects(1.0), interaction=0.05, seed=7)
        train = target.variants[target.variants["depth"] <= 1]
        landscape = PosteriorLandscape.condition(WILD_TYPE, train, prior).sample(3)
        for mutations, fitness in train[["mutations", "fitness"]].itertuples(
            index=False, name=None
        ):
            assert landscape.evaluate(mutations) == pytest.approx(fitness)

    def test_the_readout_can_be_switched_off(self):
        datasets = [_landscape(f"P{i}", _effects(1.0), seed=i) for i in range(3)]
        prior = LandscapePrior.fit(datasets, max_order=2, global_epistasis=False)
        target = _landscape("T", _effects(1.0), seed=7)
        train = target.variants[target.variants["depth"] <= 1]
        posterior = PosteriorLandscape.condition(WILD_TYPE, train, prior, global_epistasis=False)
        assert posterior.nonlinearity is None
        assert posterior.preimage_effects == {}


class TestEligibility:
    def test_censored_singles_are_rejected(self):
        rows = [{"mutations": {}, "fitness": 0.0, "depth": 0}]
        rows += [
            {"mutations": {position: "G"}, "fitness": 0.0, "depth": 1} for position in POSITIONS
        ]
        frame = pd.DataFrame(rows)
        assert not has_informative_singles(frame)
        assert single_effect_summary(frame)["distinct"] == 1.0

    def test_varying_singles_are_accepted(self):
        rows = [{"mutations": {}, "fitness": 0.0, "depth": 0}]
        rows += [
            {"mutations": {position: "G"}, "fitness": 0.1 * position, "depth": 1}
            for position in POSITIONS
        ]
        assert has_informative_singles(pd.DataFrame(rows))


class TestTopologyAdapter:
    def test_an_additive_landscape_has_one_peak_and_no_high_order_variance(self):
        # The arithmetic the topology comparison rests on. Any other answer means
        # the tensor adapter is wrong rather than the landscape being unusual.
        grid = np.add.outer(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 1.5]))
        summary = summarise(grid)
        assert summary["n_optima"] == 1.0
        assert summary["walsh_nonadditive"] == pytest.approx(0.0, abs=1e-12)
        assert summary["walsh_order_1"] == pytest.approx(1.0, abs=1e-12)

    def test_the_cube_keeps_only_substitutions_backed_by_a_single(self):
        dataset = _landscape("T", _effects(1.0), seed=1)
        train = dataset.variants[dataset.variants["depth"] <= 1]
        cube = supported_cube(dataset, train)
        assert cube.alphabets[1] == ["A", "G"]
        variants, indices = cube.variants()
        assert len(variants) == cube.size == 2**4
        tensor, coverage = measured_tensor(dataset, cube)
        assert tensor.shape == (2, 2, 2, 2)
        assert 0.0 < coverage <= 1.0


class TestReadoutScaleIsUnidentifiable:
    def test_the_sigmoid_scale_cannot_change_any_output(self):
        # g(z) = floor + span * sigmoid(z / s + tau), so the preimage of an
        # effect carries a factor of s and the composition divides it back out.
        # Anything that tries to tune saturation through the scale is a no-op,
        # and only the floor and ceiling move the landscape.
        from dataclasses import replace as replace_field

        datasets = [_landscape(f"P{i}", _effects(1.0), interaction=0.05, seed=i) for i in range(3)]
        base = LandscapePrior.fit(datasets, max_order=2)
        target = _landscape("T", _effects(1.0), interaction=0.05, seed=7)
        train = target.variants[target.variants["depth"] <= 1]
        probes = [{1: "G", 2: "V"}, {1: "G", 3: "W"}, {2: "V", 3: "W", 4: "K"}]

        reference = None
        for ratio in (1.0, 2.5, 0.3):
            prior = replace_field(base, readout_scale_ratio=ratio)
            landscape = PosteriorLandscape.condition(WILD_TYPE, train, prior).sample(3)
            values = landscape.evaluate_batch(probes)
            if reference is None:
                reference = values
            else:
                assert np.allclose(values, reference, atol=1e-12)
