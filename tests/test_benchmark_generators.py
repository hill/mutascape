from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mutascape.benchmarks.generators import (
    build_primary_generators,
    dms_pairwise_readout_tensor,
    potts_tensor,
)
from mutascape.benchmarks.topology import CubeSpec
from mutascape.completion import DMSPairwiseLandscape
from mutascape.nonlinearity import GlobalEpistasisSigmoid


def test_dms_pairwise_readout_preserves_wild_type_and_singles():
    cube = CubeSpec(
        positions=(1, 2),
        alphabets={1: ["A", "B"], 2: ["A", "B"]},
        wild_type={1: "A", 2: "A"},
    )
    readout = GlobalEpistasisSigmoid(floor=-1.0, ceiling=1.5, scale=0.8)
    single_effects = {(1, "B"): 0.4, (2, "B"): -0.3}
    preimage_effects = {
        mutation: float(readout.inverse(effect)) for mutation, effect in single_effects.items()
    }
    posterior = SimpleNamespace(
        wild_type_sequence="AA",
        nonlinearity=readout,
        single_effects=single_effects,
        preimage_effects=preimage_effects,
        preimage_scale=float(np.std(list(preimage_effects.values()))),
        target_scale=float(np.std(list(single_effects.values()))),
    )

    tensor = dms_pairwise_readout_tensor(cube, posterior, coupling=0.25, seed=9)

    assert tensor[0, 0] == pytest.approx(0.0)
    assert tensor[1, 0] == pytest.approx(0.4)
    assert tensor[0, 1] == pytest.approx(-0.3)


def test_dms_pairwise_readout_is_stable_for_seed():
    cube = CubeSpec(
        positions=(1, 2),
        alphabets={1: ["A", "B"], 2: ["A", "B"]},
        wild_type={1: "A", 2: "A"},
    )
    posterior = SimpleNamespace(
        wild_type_sequence="AA",
        nonlinearity=None,
        single_effects={(1, "B"): 0.4, (2, "B"): -0.3},
        preimage_effects={},
        preimage_scale=1.0,
        target_scale=0.35,
    )

    first = dms_pairwise_readout_tensor(cube, posterior, coupling=0.25, seed=9)
    repeated = dms_pairwise_readout_tensor(cube, posterior, coupling=0.25, seed=9)
    second = dms_pairwise_readout_tensor(cube, posterior, coupling=0.25, seed=10)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, second)


def test_dms_pairwise_tensor_matches_lazy_oracle():
    cube = CubeSpec(
        positions=(1, 2),
        alphabets={1: ["A", "C"], 2: ["A", "D"]},
        wild_type={1: "A", 2: "A"},
    )
    posterior = SimpleNamespace(
        wild_type_sequence="AA",
        observations={(): 1.2, ((1, "C"),): 1.6, ((2, "D"),): 0.9},
        nonlinearity=None,
        single_effects={(1, "C"): 0.4, (2, "D"): -0.3},
        preimage_effects={},
        preimage_scale=1.0,
        target_scale=0.35,
    )
    tensor = dms_pairwise_readout_tensor(cube, posterior, coupling=0.1, seed=7)
    variants, indices = cube.variants()
    oracle = DMSPairwiseLandscape(posterior, seed=7, coupling=0.1)
    wt = posterior.observations[()]

    for variant, index in zip(variants, indices):
        assert tensor[index] == pytest.approx(oracle.evaluate(variant) - wt)


def test_primary_generators_have_matched_draws():
    cube = CubeSpec(
        positions=(1, 2),
        alphabets={1: ["A", "C"], 2: ["A", "D"]},
        wild_type={1: "A", 2: "A"},
    )
    train = pd.DataFrame(
        {
            "mutations": [{}, {1: "C"}, {2: "D"}],
            "fitness": [1.2, 1.6, 0.9],
            "depth": [0, 1, 1],
        }
    )
    posterior = SimpleNamespace(
        wild_type_sequence="AA",
        nonlinearity=None,
        single_effects={(1, "C"): 0.4, (2, "D"): -0.3},
        preimage_effects={},
        preimage_scale=1.0,
        target_scale=0.35,
    )

    generators = build_primary_generators(
        cube, train, seed=7, posterior=posterior, coupling=0.1, draws=3
    )

    assert set(generators) == {
        "dms_pairwise_0",
        "dms_pairwise_1",
        "dms_pairwise_2",
        "potts_low_0",
        "potts_low_1",
        "potts_low_2",
        "rmf_high_0",
        "rmf_high_1",
        "rmf_high_2",
    }
    assert all(tensor.shape == (2, 2) for tensor in generators.values())
    assert not np.array_equal(generators["potts_low_0"], generators["potts_low_1"])


def test_potts_tensor_preserves_wild_type_and_singles():
    cube = CubeSpec(
        positions=(1, 2),
        alphabets={1: ["A", "C"], 2: ["A", "D"]},
        wild_type={1: "A", 2: "A"},
    )
    train = pd.DataFrame(
        {
            "mutations": [{}, {1: "C"}, {2: "D"}],
            "fitness": [1.2, 1.6, 0.9],
            "depth": [0, 1, 1],
        }
    )

    tensor = potts_tensor(cube, train, coupling=1.0, seed=5)

    assert tensor[0, 0] == pytest.approx(0.0)
    assert tensor[1, 0] == pytest.approx(0.4)
    assert tensor[0, 1] == pytest.approx(-0.3)
