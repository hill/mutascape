"""Tests for the thermodynamic non-linearity layer and its Landscape integration."""

import numpy as np
import pandas as pd
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import RandomEpistasis
from mutascape.landscape import Landscape
from mutascape.nonlinearity import Identity, ThermodynamicSigmoid, resolve


class TestIdentity:
    def test_forward_and_inverse_passthrough(self):
        g = Identity()
        for x in (-3.14, 0.0, 5.0):
            assert g.forward(x) == x
            assert g.inverse(x) == x

    def test_array_passthrough(self):
        g = Identity()
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(g.forward(x), x)
        np.testing.assert_array_equal(g.inverse(x), x)


class TestThermodynamicSigmoid:
    def test_wt_pinned_at_zero(self):
        g = ThermodynamicSigmoid(floor=-1.0, ceiling=5.0, scale=1.2)
        assert g.forward(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_asymptotes(self):
        g = ThermodynamicSigmoid(floor=-1.2, ceiling=4.5, scale=0.8)
        assert g.forward(-1e6) == pytest.approx(g.floor, abs=1e-6)
        assert g.forward(1e6) == pytest.approx(g.ceiling, abs=1e-6)

    def test_monotone(self):
        g = ThermodynamicSigmoid(floor=-1.0, ceiling=3.0, scale=1.0)
        xs = np.linspace(-10, 10, 201)
        ys = g.forward(xs)
        assert np.all(np.diff(ys) > 0)

    def test_round_trip_invertibility(self):
        g = ThermodynamicSigmoid(floor=-1.1, ceiling=7.8, scale=1.5)
        xs = np.linspace(-6, 6, 50)
        ys = g.forward(xs)
        xs_back = g.inverse(ys)
        np.testing.assert_allclose(xs, xs_back, atol=1e-10)

    def test_inverse_zero_is_zero(self):
        g = ThermodynamicSigmoid(floor=-1.5, ceiling=4.0, scale=2.0)
        assert g.inverse(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError, match="ceiling"):
            ThermodynamicSigmoid(floor=1.0, ceiling=0.5, scale=1.0)
        with pytest.raises(ValueError, match="scale must be positive"):
            ThermodynamicSigmoid(floor=-1.0, ceiling=2.0, scale=-0.1)
        with pytest.raises(ValueError, match="WT fitness 0"):
            # WT=0 not in (floor, ceiling): both positive
            ThermodynamicSigmoid(floor=0.5, ceiling=2.0, scale=1.0)

    def test_floor_hard_none_preserves_smooth_behaviour(self):
        g = ThermodynamicSigmoid(floor=-1.2, ceiling=3.0, scale=1.0, floor_hard=None)
        ref = ThermodynamicSigmoid(floor=-1.2, ceiling=3.0, scale=1.0)
        xs = np.linspace(-6, 6, 50)
        np.testing.assert_array_equal(g.forward(xs), ref.forward(xs))

    def test_floor_hard_clips_below_threshold(self):
        g = ThermodynamicSigmoid(floor=-1.2, ceiling=3.0, scale=1.0, floor_hard=-0.5)
        # Deep negative z would smooth to near the floor (-1.2), gets clipped to -0.5.
        assert g.forward(-1e6) == pytest.approx(-0.5, abs=1e-9)
        # WT (z=0) is above the hard floor, unaffected.
        assert g.forward(0.0) == pytest.approx(0.0, abs=1e-12)
        # Large positive z reaches the ceiling, also unaffected.
        assert g.forward(1e6) == pytest.approx(3.0, abs=1e-6)

    def test_floor_hard_clip_is_pointwise_on_arrays(self):
        g = ThermodynamicSigmoid(floor=-1.2, ceiling=3.0, scale=1.0, floor_hard=-0.3)
        xs = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        out = g.forward(xs)
        # Every entry above or equal to the hard floor.
        assert (out >= -0.3 - 1e-12).all()
        # WT entry still exactly zero.
        assert out[2] == pytest.approx(0.0, abs=1e-12)

    def test_floor_hard_rejects_non_negative_values(self):
        with pytest.raises(ValueError, match="floor_hard"):
            ThermodynamicSigmoid(floor=-1.0, ceiling=2.0, scale=1.0, floor_hard=0.0)
        with pytest.raises(ValueError, match="floor_hard"):
            ThermodynamicSigmoid(floor=-1.0, ceiling=2.0, scale=1.0, floor_hard=0.5)


class TestFitFromSingles:
    def test_recovers_known_scale(self):
        # Synthesize singles by pushing standard-normal ΔΔGs through a known g.
        true = ThermodynamicSigmoid(floor=-2.0, ceiling=6.0, scale=1.3)
        rng = np.random.default_rng(0)
        deltag = rng.normal(0.0, 1.0, size=2000)
        observed = true.forward(deltag)
        effects = {1: {chr(65 + i): float(observed[i]) for i in range(len(observed))}}
        fitted = ThermodynamicSigmoid.fit_from_singles(effects, margin=0.02)
        # Recovering an exact scale in the presence of finite data + margin
        # inflation is noisy; aim for order-of-magnitude agreement.
        assert 0.5 < fitted.scale < 2.5
        assert fitted.floor < observed.min()
        assert fitted.ceiling > observed.max()

    def test_rejects_range_not_straddling_wt(self):
        effects = {1: {"A": 0.5, "G": 1.2, "V": 0.8}}
        with pytest.raises(ValueError, match="straddle"):
            ThermodynamicSigmoid.fit_from_singles(effects)

    def test_rejects_empty_effects(self):
        with pytest.raises(ValueError, match="no values"):
            ThermodynamicSigmoid.fit_from_singles({})


class TestResolve:
    def test_none_gives_identity(self):
        out = resolve(None, {1: {"A": 0.1, "G": -0.2}})
        assert isinstance(out, Identity)

    def test_passes_through_instance(self):
        g = ThermodynamicSigmoid(floor=-1.0, ceiling=2.0, scale=1.0)
        out = resolve(g, {1: {"A": 0.1}})
        assert out is g

    def test_fit_string_triggers_fit(self):
        effects = {1: {"A": 0.3, "G": -0.1, "V": -0.5, "L": 0.2}}
        out = resolve("fit", effects)
        assert isinstance(out, ThermodynamicSigmoid)

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown nonlinearity"):
            resolve("thermo", {1: {"A": 0.1, "G": -0.2}})


@pytest.fixture
def tiny_landscape_args(tmp_path):
    distances = np.array(
        [
            [0.0, 6.0, 15.0],
            [6.0, 0.0, 7.0],
            [15.0, 7.0, 0.0],
        ]
    )
    cm = ContactMap.from_distance_matrix(distances, [10, 20, 30], threshold=8.0)
    dms = pd.DataFrame({
        "position": [10, 10, 20, 20, 30, 30],
        "wildtype": ["A", "A", "L", "L", "K", "K"],
        "mutant": ["G", "V", "I", "F", "R", "E"],
        "fitness": [0.5, -0.3, 0.2, -0.5, 0.1, -0.2],
    })
    csv = tmp_path / "dms.csv"
    dms.to_csv(csv, index=False)
    return dict(dms_path=csv, contact_map=cm, epistasis=RandomEpistasis(sigma=0.3, seed=42))


class TestLandscapeIntegration:
    def test_identity_preserves_legacy_evaluate(self, tiny_landscape_args):
        # Default (None -> Identity) must match explicit Identity bit-for-bit.
        default_ls = Landscape.from_dms(**tiny_landscape_args, alpha=1.0)
        identity_ls = Landscape.from_dms(**tiny_landscape_args, alpha=1.0, nonlinearity=Identity())
        for muts in ({10: "G"}, {20: "I"}, {10: "G", 20: "I"}, {10: "V", 20: "F", 30: "R"}):
            assert default_ls.evaluate(muts) == pytest.approx(identity_ls.evaluate(muts))

    def test_single_mutants_recover_dms_values(self, tiny_landscape_args):
        # With the sigmoid applied, g(g^{-1}(h_i)) = h_i must hold at every single.
        ls = Landscape.from_dms(**tiny_landscape_args, alpha=1.0, nonlinearity="fit")
        for pos, aa, expected in [(10, "G", 0.5), (10, "V", -0.3), (20, "I", 0.2),
                                  (20, "F", -0.5), (30, "R", 0.1), (30, "E", -0.2)]:
            assert ls.evaluate({pos: aa}) == pytest.approx(expected, abs=1e-9)

    def test_wt_is_zero_under_nonlinearity(self, tiny_landscape_args):
        ls = Landscape.from_dms(**tiny_landscape_args, alpha=1.0, nonlinearity="fit")
        # WT (no mutations): additive preimage = 0, epistatic = 0, g(0) = 0.
        assert ls.evaluate({}) == pytest.approx(0.0, abs=1e-12)

    def test_multi_mutants_depart_from_linear_sum(self, tiny_landscape_args):
        # The thermodynamic non-linearity must produce a multi-mutant prediction
        # that differs from naive sum-of-singles even when alpha = 0 (no pairwise).
        ls = Landscape.from_dms(
            **tiny_landscape_args,
            alpha=0.0,
            nonlinearity=ThermodynamicSigmoid(floor=-1.0, ceiling=1.2, scale=0.5),
        )
        muts = {10: "G", 20: "I", 30: "R"}
        actual = ls.evaluate(muts)
        linear_sum = ls.additive_fitness(muts)
        # Actual fitness must differ from linear sum by a meaningful amount.
        assert abs(actual - linear_sum) > 1e-3

    def test_preimage_cache_honors_dataclass_mutation(self, tiny_landscape_args):
        ls = Landscape.from_dms(**tiny_landscape_args, alpha=1.0, nonlinearity="fit")
        # Every observed single-site AA must have a cached preimage.
        for pos, effects in ls.single_site_effects.items():
            for aa in effects:
                assert aa in ls._preimage_effects[pos]
        # And the preimage must invert the outer g at that h_i.
        for pos, effects in ls.single_site_effects.items():
            for aa, h in effects.items():
                recovered = ls.nonlinearity.forward(ls._preimage_effects[pos][aa])
                assert recovered == pytest.approx(h, abs=1e-9)
