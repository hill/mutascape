"""Tests for the landscape module."""

from itertools import islice

import numpy as np
import pandas as pd
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import RandomEpistasis
from mutascape.landscape import Landscape, Variant
from mutascape.mutant import Mutant


@pytest.fixture
def small_contact_map():
    """3-position contact map with known topology."""
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
    """Create a minimal DMS CSV file."""
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
    """A simple test landscape."""
    return Landscape.from_dms(
        dms_path=dms_csv,
        contact_map=small_contact_map,
        epistasis=RandomEpistasis(sigma=0.3, seed=42),
        alpha=1.0,
        seed=42,
    )


class TestLandscapeConstruction:
    def test_from_dms_basic(self, landscape):
        assert landscape.n_positions == 3
        assert set(landscape.positions) == {10, 20, 30}

    def test_wild_type_parsed(self, landscape):
        assert landscape.wild_type[10] == "A"
        assert landscape.wild_type[20] == "L"
        assert landscape.wild_type[30] == "K"

    def test_single_site_effects_parsed(self, landscape):
        assert landscape.single_site_effects[10]["G"] == pytest.approx(0.5)
        assert landscape.single_site_effects[20]["F"] == pytest.approx(-0.5)


class TestLandscapeEvaluation:
    def test_empty_mutations_is_zero(self, landscape):
        """No mutations = wild type = zero fitness effect."""
        assert landscape.evaluate({}) == pytest.approx(0.0)

    def test_getitem_with_mutant(self, landscape):
        m = Mutant("A10G")
        assert landscape[m] == pytest.approx(landscape.evaluate({10: "G"}))

    def test_getitem_with_string(self, landscape):
        assert landscape["A10G"] == pytest.approx(landscape.evaluate({10: "G"}))

    def test_getitem_with_combined_mutant(self, landscape):
        m = Mutant("A10G") + Mutant("L20I")
        assert landscape[m] == pytest.approx(landscape.evaluate({10: "G", 20: "I"}))

    def test_getitem_wt_string(self, landscape):
        assert landscape["wt"] == pytest.approx(0.0)

    def test_evaluate_accepts_mutant(self, landscape):
        """Mutant is a dict subclass, so evaluate() should accept it directly."""
        m = Mutant("A10G+L20I")
        assert landscape.evaluate(m) == pytest.approx(landscape.evaluate({10: "G", 20: "I"}))

    def test_single_mutation_additive(self, landscape):
        """With alpha=0, single mutation should give exactly the DMS effect."""
        landscape_additive = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=0.0,  # no epistasis
            seed=42,
        )
        fitness = landscape_additive.evaluate({10: "G"})
        assert fitness == pytest.approx(0.5)

    def test_alpha_zero_is_additive(self, landscape):
        """With alpha=0, multi-mutant fitness = sum of single effects."""
        landscape_additive = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=0.0,
            seed=42,
        )
        # G at 10 (+0.5) and I at 20 (+0.2) = 0.7 additively
        fitness = landscape_additive.evaluate({10: "G", 20: "I"})
        assert fitness == pytest.approx(0.5 + 0.2)

    def test_epistasis_changes_fitness(self, landscape):
        """With alpha > 0, multi-mutant should differ from additive sum."""
        additive_sum = landscape.additive_fitness({10: "G", 20: "I"})
        actual = landscape.evaluate({10: "G", 20: "I"})
        # They should differ (alpha=1.0 and sigma=0.3, so epistasis is non-trivial)
        # Not guaranteed but overwhelmingly likely with random couplings
        # Use a generous tolerance
        assert actual != pytest.approx(additive_sum, abs=1e-10)

    def test_additive_fitness_method(self, landscape):
        assert landscape.additive_fitness({10: "G"}) == pytest.approx(0.5)
        assert landscape.additive_fitness({10: "V", 20: "F"}) == pytest.approx(-0.3 + -0.5)

    def test_evaluate_batch(self, landscape):
        variants = [{10: "G"}, {20: "I"}, {10: "G", 20: "I"}]
        results = landscape.evaluate_batch(variants)
        assert results.shape == (3,)
        assert results[0] == pytest.approx(landscape.evaluate({10: "G"}))


class TestLandscapeSampling:
    def test_sample_returns_correct_shape(self, landscape):
        variants, fitness = landscape._sample(n=50, n_mutations=2)
        assert len(variants) == 50
        assert fitness.shape == (50,)

    def test_sample_correct_mutation_count(self, landscape):
        variants, _ = landscape._sample(n=20, n_mutations=2)
        for v in variants:
            assert len(v) == 2

    def test_sample_excludes_wild_type(self, landscape):
        variants, _ = landscape._sample(n=100, n_mutations=1, exclude_wild_type_aa=True)
        for v in variants:
            for pos, aa in v.items():
                assert aa != landscape.wild_type[pos]

    def test_sample_too_many_mutations_raises(self, landscape):
        with pytest.raises(ValueError, match="Cannot sample"):
            landscape._sample(n=10, n_mutations=10)  # only 3 positions

    def test_sample_reproducible(self, landscape):
        # Reset seed by creating new landscape
        l1 = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=1.0,
            seed=123,
        )
        l2 = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=1.0,
            seed=123,
        )
        v1, f1 = l1._sample(n=10, n_mutations=2)
        v2, f2 = l2._sample(n=10, n_mutations=2)
        np.testing.assert_array_equal(f1, f2)


class TestLandscapeStats:
    def test_stats_returns_expected_keys(self, landscape):
        stats = landscape.stats(n_samples=100, n_mutations=2)
        assert "mean_fitness" in stats
        assert "std_fitness" in stats
        assert "additive_correlation" in stats
        assert "mean_epistatic_deviation" in stats
        assert "fraction_sign_epistasis" in stats

    def test_additive_landscape_high_correlation(self, landscape):
        """An additive landscape should have correlation ~1.0."""
        additive = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=0.0,
            seed=42,
        )
        stats = additive.stats(n_samples=200, n_mutations=2)
        assert stats["additive_correlation"] == pytest.approx(1.0, abs=0.01)
        assert stats["mean_epistatic_deviation"] == pytest.approx(0.0, abs=0.01)

    def test_epistatic_landscape_lower_correlation(self, landscape):
        """A landscape with alpha > 0 should have lower additive correlation."""
        stats = landscape.stats(n_samples=500, n_mutations=2)
        assert stats["additive_correlation"] < 1.0


# --- WT sequence, generate(), to_dataframe() ---

# Position 10=A (index 9), Position 20=L (index 19), Position 30=K (index 29)
WT_SEQ = "AAAAAAAAA" + "A" + "AAAAAAAAA" + "L" + "AAAAAAAAA" + "K"
# len=30: indices 0-8='A', 9='A'(pos10), 10-18='A', 19='L'(pos20), 20-28='A', 29='K'(pos30)


@pytest.fixture
def landscape_with_seq(small_contact_map, dms_csv):
    """Landscape with wt_sequence set."""
    return Landscape.from_dms(
        dms_path=dms_csv,
        contact_map=small_contact_map,
        epistasis=RandomEpistasis(sigma=0.3, seed=42),
        alpha=1.0,
        seed=42,
        wt_fasta=None,  # set manually below
    )


@pytest.fixture
def landscape_seq(small_contact_map, dms_csv):
    """Landscape with a WT sequence that matches positions 10, 20, 30."""
    ls = Landscape.from_dms(
        dms_path=dms_csv,
        contact_map=small_contact_map,
        epistasis=RandomEpistasis(sigma=0.3, seed=42),
        alpha=1.0,
        seed=42,
    )
    # Manually set wt_sequence (positions 10=A, 20=L, 30=K in 1-indexed)
    object.__setattr__(ls, "wt_sequence", WT_SEQ)
    ls._validate_wt_sequence()
    return ls


class TestWtSequence:
    def test_mutant_sequence_single(self, landscape_seq):
        seq = landscape_seq.mutant_sequence({10: "G"})
        assert seq[9] == "G"  # pos 10 -> index 9
        assert seq[19] == "L"  # pos 20 unchanged
        assert len(seq) == len(WT_SEQ)

    def test_mutant_sequence_double(self, landscape_seq):
        seq = landscape_seq.mutant_sequence({10: "G", 20: "I"})
        assert seq[9] == "G"
        assert seq[19] == "I"
        assert seq[29] == "K"  # pos 30 unchanged

    def test_mutant_sequence_empty_is_wt(self, landscape_seq):
        seq = landscape_seq.mutant_sequence({})
        assert seq == WT_SEQ

    def test_mutant_sequence_accepts_mutant(self, landscape_seq):
        m = Mutant("A10G+L20I")
        seq = landscape_seq.mutant_sequence(m)
        assert seq[9] == "G"
        assert seq[19] == "I"

    def test_additive_fitness_accepts_mutant(self, landscape_seq):
        m = Mutant("A10G")
        assert landscape_seq.additive_fitness(m) == pytest.approx(0.5)

    def test_mutant_sequence_raises_without_wt(self, landscape):
        with pytest.raises(ValueError, match="wt_sequence is not set"):
            landscape.mutant_sequence({10: "G"})

    def test_wt_validation_catches_mismatch(self, landscape):
        bad_seq = "X" * 30  # wrong AAs everywhere
        with pytest.raises(ValueError, match="WT sequence mismatch"):
            Landscape(
                wild_type=landscape.wild_type,
                single_site_effects=landscape.single_site_effects,
                contact_map=landscape.contact_map,
                couplings=landscape.couplings,
                wt_sequence=bad_seq,
            )

    def test_wt_validation_catches_short_sequence(self, landscape):
        with pytest.raises(ValueError, match="out of range"):
            Landscape(
                wild_type=landscape.wild_type,
                single_site_effects=landscape.single_site_effects,
                contact_map=landscape.contact_map,
                couplings=landscape.couplings,
                wt_sequence="SHORT",
            )


class TestGenerate:
    def test_yields_correct_count(self, landscape_seq):
        variants = list(landscape_seq.generate(n_mutations=2, n=10, seed=99))
        assert len(variants) == 10

    def test_variant_fields(self, landscape_seq):
        variant = next(landscape_seq.generate(n_mutations=1, n=1, seed=99))
        assert isinstance(variant, Variant)
        assert isinstance(variant.mutations, Mutant)
        assert isinstance(variant.sequence, str)
        assert isinstance(variant.fitness, float)
        # label lives on mutations, not on Variant
        assert isinstance(variant.mutations.label, str)
        assert not hasattr(variant, "label")
        assert len(variant) == 3  # mutations, sequence, fitness

    def test_generated_mutant_has_wild_type_info(self, landscape_seq):
        """Generated Mutant objects should carry WT context for label rendering."""
        variant = next(landscape_seq.generate(n_mutations=2, n=1, seed=99))
        assert "?" not in variant.mutations.label

    def test_mutation_count_fixed(self, landscape_seq):
        for v in landscape_seq.generate(n_mutations=2, n=20, seed=99):
            assert len(v.mutations) == 2

    def test_mutation_count_range(self, landscape_seq):
        depths = set()
        for v in landscape_seq.generate(n_mutations=(1, 3), n=50, seed=99):
            assert 1 <= len(v.mutations) <= 3
            depths.add(len(v.mutations))
        # With 50 samples over 3 depths, we should see all of them
        assert depths == {1, 2, 3}

    def test_infinite_generator(self, landscape_seq):
        """Can take a few from an infinite generator."""
        gen = landscape_seq.generate(n_mutations=1)
        variants = list(islice(gen, 5))
        assert len(variants) == 5

    def test_sequence_matches_mutations(self, landscape_seq):
        for v in landscape_seq.generate(n_mutations=2, n=10, seed=99):
            expected = landscape_seq.mutant_sequence(v.mutations)
            assert v.sequence == expected

    def test_fitness_matches_evaluate(self, landscape_seq):
        """Generated fitness should match direct evaluate() call."""
        # Use a separate seed and no noise to get deterministic results
        ls = Landscape(
            wild_type=landscape_seq.wild_type,
            single_site_effects=landscape_seq.single_site_effects,
            contact_map=landscape_seq.contact_map,
            couplings=landscape_seq.couplings,
            alpha=1.0,
            noise_sigma=0.0,
            wt_sequence=WT_SEQ,
            seed=123,
        )
        for v in ls.generate(n_mutations=2, n=10, seed=77):
            assert v.fitness == pytest.approx(ls.evaluate(v.mutations))

    def test_raises_without_wt_sequence(self, landscape):
        with pytest.raises(ValueError, match="wt_sequence is not set"):
            next(landscape.generate(n_mutations=1, n=1))

    def test_reproducible(self, landscape_seq):
        v1 = list(landscape_seq.generate(n_mutations=2, n=10, seed=42))
        v2 = list(landscape_seq.generate(n_mutations=2, n=10, seed=42))
        for a, b in zip(v1, v2):
            assert a.mutations == b.mutations
            assert a.fitness == b.fitness


class TestToDataframe:
    def test_columns(self, landscape_seq):
        df = landscape_seq.to_dataframe(n_variants=20, n_mutations=2, seed=42)
        assert set(df.columns) == {"sequence", "variant", "fitness", "n_mutations"}

    def test_row_count(self, landscape_seq):
        # 3 positions, 2 non-WT AAs each -> 6 possible singles
        df = landscape_seq.to_dataframe(n_variants=6, n_mutations=1, seed=42)
        assert len(df) == 6

    def test_no_duplicates(self, landscape_seq):
        df = landscape_seq.to_dataframe(n_variants=20, n_mutations=1, seed=42)
        # Each variant label should be unique
        assert df["variant"].nunique() == len(df)

    def test_mutation_range(self, landscape_seq):
        df = landscape_seq.to_dataframe(n_variants=30, n_mutations=(1, 3), seed=42)
        assert df["n_mutations"].min() >= 1
        assert df["n_mutations"].max() <= 3

    def test_variant_labels_have_wt_context(self, landscape_seq):
        """Labels should use proper WT amino acids, not '?'."""
        df = landscape_seq.to_dataframe(n_variants=10, n_mutations=1, seed=42)
        for label in df["variant"]:
            assert "?" not in label
