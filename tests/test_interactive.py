"""Tests for the interactive visualisation module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import RandomEpistasis
from mutascape.interactive import (
    _build_full_genotypes,
    _embed_2d,
    _encode_variants_onehot,
    _evaluate_at_alpha,
    _interpolate_surface,
    _variant_labels,
    morphing_landscape,
)
from mutascape.landscape import Landscape

WT_SEQ = "AAAAAAAAA" + "A" + "AAAAAAAAA" + "L" + "AAAAAAAAA" + "K"


@pytest.fixture
def small_contact_map():
    distances = np.array(
        [
            [0.0, 6.0, 15.0],
            [6.0, 0.0, 7.0],
            [15.0, 7.0, 0.0],
        ]
    )
    return ContactMap.from_distance_matrix(distances, [10, 20, 30], threshold=8.0)


@pytest.fixture
def dms_csv(tmp_path):
    data = {
        "position": [10, 10, 20, 20, 30, 30],
        "wildtype": ["A", "A", "L", "L", "K", "K"],
        "mutant": ["G", "V", "I", "F", "R", "E"],
        "fitness": [0.5, -0.3, 0.2, -0.5, 0.1, -0.2],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_dms.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def landscape(small_contact_map, dms_csv):
    ls = Landscape.from_dms(
        dms_path=dms_csv,
        contact_map=small_contact_map,
        epistasis=RandomEpistasis(sigma=0.3, seed=42),
        alpha=1.0,
        seed=42,
    )
    object.__setattr__(ls, "wt_sequence", WT_SEQ)
    ls._validate_wt_sequence()
    return ls


class TestOneHotEncoding:
    def test_shape(self):
        variants = [{10: "G", 20: "I"}, {10: "V"}]
        positions = [10, 20, 30]
        aa = tuple("ACDEFGHIKLMNPQRSTVWY")
        result = _encode_variants_onehot(variants, positions, aa)
        assert result.shape == (2, 3 * 20)

    def test_correct_entries(self):
        aa = tuple("ACDEFGHIKLMNPQRSTVWY")
        aa_idx = {a: i for i, a in enumerate(aa)}
        variants = [{10: "G"}]
        positions = [10]
        result = _encode_variants_onehot(variants, positions, aa)
        assert result[0, aa_idx["G"]] == 1.0
        assert result.sum() == 1.0


class TestEmbed2D:
    def test_pca_shape(self):
        features = np.random.default_rng(0).normal(size=(50, 20))
        coords, label = _embed_2d(features, method="pca")
        assert coords.shape == (50, 2)
        assert label == "PCA"

    def test_tsne_shape(self):
        features = np.random.default_rng(0).normal(size=(50, 20))
        coords, label = _embed_2d(features, method="tsne")
        assert coords.shape == (50, 2)
        assert label == "t-SNE"

    def test_unknown_method_raises(self):
        features = np.random.default_rng(0).normal(size=(10, 5))
        with pytest.raises(ValueError, match="Unknown embedding"):
            _embed_2d(features, method="nope")


class TestBuildFullGenotypes:
    def test_fills_wt(self):
        wt = {10: "A", 20: "L", 30: "K"}
        variants = [{10: "G"}]
        full = _build_full_genotypes(variants, wt)
        assert full[0] == {10: "G", 20: "L", 30: "K"}

    def test_empty_variant_is_wt(self):
        wt = {10: "A", 20: "L"}
        full = _build_full_genotypes([{}], wt)
        assert full[0] == wt


class TestVariantLabels:
    def test_single_mutant(self):
        wt = {10: "A", 20: "L"}
        labels = _variant_labels([{10: "G"}], wt)
        assert labels == ["A10G"]

    def test_double_mutant(self):
        wt = {10: "A", 20: "L"}
        labels = _variant_labels([{10: "G", 20: "I"}], wt)
        assert labels == ["A10G+L20I"]

    def test_empty_variant(self):
        labels = _variant_labels([{}], {10: "A"})
        assert labels == ["WT"]


class TestEvaluateAtAlpha:
    def test_alpha_zero_is_additive(self, landscape):
        variants = [{10: "G", 20: "I"}]
        fitness = _evaluate_at_alpha(landscape, variants, alpha=0.0)
        expected = 0.5 + 0.2  # sum of single-site effects
        assert fitness[0] == pytest.approx(expected)

    def test_different_alphas_give_different_fitness(self, landscape):
        variants = [{10: "G", 20: "I"}]
        f0 = _evaluate_at_alpha(landscape, variants, alpha=0.0)
        f5 = _evaluate_at_alpha(landscape, variants, alpha=5.0)
        assert f0[0] != pytest.approx(f5[0], abs=1e-10)


class TestInterpolateSurface:
    def test_grid_shape(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)
        fitness = rng.normal(size=100)
        xi, yi, zi = _interpolate_surface(x, y, fitness, grid_resolution=30)
        assert xi.shape == (30,)
        assert yi.shape == (30,)
        assert zi.shape == (30, 30)


class TestMorphingLandscape:
    def test_returns_figure(self, landscape):
        fig = morphing_landscape(
            landscape,
            n_samples=50,
            n_mutations=2,
            alphas=[0.0, 1.0],
            grid_resolution=15,
        )
        assert isinstance(fig, go.Figure)

    def test_frame_count_matches_alphas(self, landscape):
        alphas = [0.0, 0.5, 1.0, 2.0]
        fig = morphing_landscape(
            landscape,
            n_samples=50,
            n_mutations=2,
            alphas=alphas,
            grid_resolution=15,
        )
        assert len(fig.frames) == len(alphas)

    def test_has_slider(self, landscape):
        fig = morphing_landscape(
            landscape,
            n_samples=50,
            n_mutations=2,
            alphas=[0.0, 1.0],
            grid_resolution=15,
        )
        assert len(fig.layout.sliders) == 1
        assert len(fig.layout.sliders[0].steps) == 2

    def test_has_play_pause_buttons(self, landscape):
        fig = morphing_landscape(
            landscape,
            n_samples=50,
            n_mutations=2,
            alphas=[0.0, 1.0],
            grid_resolution=15,
        )
        buttons = fig.layout.updatemenus[0].buttons
        button_labels = [b.label for b in buttons]
        assert "Play" in button_labels
        assert "Pause" in button_labels

    def test_custom_title(self, landscape):
        fig = morphing_landscape(
            landscape,
            n_samples=50,
            n_mutations=2,
            alphas=[0.0],
            grid_resolution=15,
            title="Custom Title",
        )
        assert "Custom" in fig.layout.title.text

    def test_plot_morphing_method(self, landscape):
        """Landscape.plot_morphing() delegates correctly."""
        fig = landscape.plot_morphing(
            n_samples=50,
            n_mutations=2,
            alphas=[0.0, 1.0],
            grid_resolution=15,
        )
        assert isinstance(fig, go.Figure)
