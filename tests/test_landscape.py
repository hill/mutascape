"""Tests for the landscape module."""

import warnings
from itertools import islice

import numpy as np
import pandas as pd
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    AMINO_ACIDS,
    ContactNKEpistasis,
    MSACouplingsEpistasis,
    RandomEpistasis,
)
from mutascape.landscape import (
    Landscape,
    Variant,
    _impute_missing_single_sites,
    _pin_nk_wildtype,
)
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

    def test_synonymous_mutation_returns_zero(self, landscape):
        """A10A means 'keep wildtype at position 10', so fitness = 0.0."""
        assert landscape["A10A"] == pytest.approx(0.0)

    def test_synonymous_in_combo_is_ignored(self, landscape):
        """A10G+L20L should equal A10G (the synonymous part is a no-op)."""
        assert landscape["A10G+L20L"] == pytest.approx(landscape["A10G"])

    def test_all_synonymous_returns_zero(self, landscape):
        """A10A+L20L = wildtype = 0.0."""
        assert landscape["A10A+L20L"] == pytest.approx(0.0)

    def test_wrong_wildtype_raises(self, landscape):
        """Position 10 is A, not G. Claiming G10V should raise."""
        with pytest.raises(ValueError, match="claimed wildtype 'G'.*actual wildtype is 'A'"):
            landscape["G10V"]

    def test_wrong_wildtype_in_combo_raises(self, landscape):
        """One wrong wildtype in a combo should raise."""
        with pytest.raises(ValueError, match="claimed wildtype"):
            landscape["A10G+G20I"]  # position 20 is L, not G

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

    def test_noise_is_genotype_stable_by_default(self, landscape):
        noisy = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=1.0,
            noise_sigma=0.25,
            seed=123,
        )
        first = noisy.evaluate({10: "G", 20: "I"})
        second = noisy.evaluate({10: "G", 20: "I"})
        assert second == pytest.approx(first)

    def test_full_wildtype_context_matches_sparse_mutations_with_noise(self, landscape):
        noisy = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=1.0,
            noise_sigma=0.25,
            seed=123,
        )
        sparse = {10: "G"}
        full_context = {10: "G", 20: "L", 30: "K"}
        assert noisy.evaluate(full_context) == pytest.approx(noisy.evaluate(sparse))

    def test_stochastic_noise_mode_draws_per_query(self, landscape):
        noisy = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=1.0,
            noise_sigma=0.25,
            seed=123,
            deterministic_noise=False,
        )
        first = noisy.evaluate({10: "G", 20: "I"})
        second = noisy.evaluate({10: "G", 20: "I"})
        assert second != pytest.approx(first)

    def test_readout_texture_is_genotype_stable(self, landscape):
        textured = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=1.0,
            seed=123,
            readout_texture_sigma=0.1,
            readout_texture_order=0,
        )
        first = textured.evaluate({10: "G", 20: "I"})
        second = textured.evaluate({10: "G", 20: "I"})
        assert second == pytest.approx(first)

    def test_readout_texture_floor_gate_skips_high_fitness(self, landscape):
        base = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=0.0,
            seed=123,
        )
        textured = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=0.0,
            seed=123,
            readout_texture_sigma=0.1,
            readout_texture_order=0,
            readout_texture_floor=-1.0,
        )
        assert textured.evaluate({10: "G"}) == pytest.approx(base.evaluate({10: "G"}))


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


class TestLandscapeContactNK:
    def test_alpha_ho_zero_matches_no_higher_order(self, landscape, small_contact_map):
        nk = ContactNKEpistasis(sigma=0.3, seed=0).generate(small_contact_map)
        with_nk = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=landscape.alpha,
            seed=42,
            higher_order=nk,
            alpha_ho=0.0,  # disabled
        )
        mut = {10: "G", 20: "I"}
        assert with_nk.evaluate(mut) == pytest.approx(landscape.evaluate(mut))

    def test_wildtype_still_zero_with_nk(self, landscape, small_contact_map):
        nk = ContactNKEpistasis(sigma=1.0, seed=0).generate(small_contact_map)
        with_nk = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=landscape.alpha,
            seed=42,
            higher_order=nk,
            alpha_ho=1.0,
        )
        assert with_nk.evaluate({}) == pytest.approx(0.0, abs=1e-12)

    def test_nk_contribution_changes_double_mutant(self, landscape, small_contact_map):
        nk = ContactNKEpistasis(sigma=1.0, seed=0).generate(small_contact_map)
        with_nk = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=landscape.alpha,
            seed=42,
            higher_order=nk,
            alpha_ho=1.0,
        )
        mut = {10: "G", 20: "I"}
        # NK contribution at a double mutant is nonzero almost surely under a
        # Gaussian draw; two samples should differ by more than float epsilon.
        assert abs(with_nk.evaluate(mut) - landscape.evaluate(mut)) > 1e-6

    def test_pin_nk_wildtype_zeros_all_wt_index(self, small_contact_map):
        nk = ContactNKEpistasis(sigma=0.5, seed=0).generate(small_contact_map)
        wt = {10: "A", 20: "L", 30: "K"}
        pinned = _pin_nk_wildtype(nk, wt)
        for pos, f in pinned.per_position_functions.items():
            neighbours = pinned.neighbours_by_position[pos]
            wt_idx = (
                pinned.aa_index[wt[pos]],
                *(pinned.aa_index[wt[n]] for n in neighbours),
            )
            assert f[wt_idx] == pytest.approx(0.0, abs=1e-12)


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


class TestLocalOptima:
    def test_returns_expected_keys(self, landscape):
        result = landscape.local_optima()
        for key in (
            "n_optima",
            "peak_indices",
            "peak_fitnesses",
            "global_peak",
            "basin_sizes",
            "positions",
            "alphabets",
            "tensor",
        ):
            assert key in result

    def test_tensor_shape_matches_alphabets(self, landscape):
        result = landscape.local_optima()
        expected_shape = tuple(len(result["alphabets"][p]) for p in result["positions"])
        assert result["tensor"].shape == expected_shape
        assert result["n_variants"] == int(np.prod(expected_shape))

    def test_basin_sizes_sum_to_space_size(self, landscape):
        result = landscape.local_optima()
        assert sum(result["basin_sizes"]) == result["n_variants"]

    def test_additive_landscape_has_single_peak_matching_argmax(self, small_contact_map, dms_csv):
        # With alpha=0 and no epistatic contribution, the global optimum is
        # the combination of the best single-site mutations at each position.
        additive = Landscape.from_dms(
            dms_path=dms_csv,
            contact_map=small_contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            alpha=0.0,
        )
        result = additive.local_optima()
        assert result["n_optima"] == 1
        best_aa_per_pos = {
            pos: max(additive.single_site_effects[pos].items(), key=lambda kv: kv[1])[0]
            for pos in additive.positions
        }
        expected_global = tuple(
            result["alphabets"][pos].index(best_aa_per_pos[pos]) for pos in result["positions"]
        )
        assert result["global_peak"] == expected_global

    def test_restricted_positions(self, landscape):
        subset = landscape.positions[:2]
        result = landscape.local_optima(positions=subset)
        assert result["positions"] == subset
        assert result["tensor"].ndim == 2


class TestImputeMissingSingleSites:
    """Synthetic single-site imputation. Replaces silent 0-default at evaluate
    time with explicit per-position fill at landscape construction time."""

    @pytest.fixture
    def partial_effects(self):
        """3-position dict where each position has WT + 2 measured AAs of the 19
        non-WT alphabet. 17 missing AAs per position = 51 missing total."""
        return {
            10: {"A": 0.0, "G": 0.6, "V": -0.3},  # A is wt
            20: {"L": 0.0, "I": 0.2, "F": -0.5},  # L is wt
            30: {"K": 0.0, "R": 0.1, "E": -0.2},  # K is wt
        }

    def test_position_mean_fills_all_20_aas(self, partial_effects):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _impute_missing_single_sites(partial_effects, strategy="position_mean")
        for pos in (10, 20, 30):
            assert set(out[pos].keys()) == set(AMINO_ACIDS)

    def test_position_mean_value_is_mean_of_non_wt_measured(self, partial_effects):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _impute_missing_single_sites(partial_effects, strategy="position_mean")
        # Position 10: non-WT measured are G=+0.6 and V=-0.3; mean = +0.15
        for aa in AMINO_ACIDS:
            if aa not in {"A", "G", "V"}:
                assert out[10][aa] == pytest.approx(0.15)

    def test_wt_entry_preserved_at_zero(self, partial_effects):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _impute_missing_single_sites(partial_effects, strategy="position_mean")
        assert out[10]["A"] == 0.0
        assert out[20]["L"] == 0.0
        assert out[30]["K"] == 0.0

    def test_measured_entries_preserved(self, partial_effects):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _impute_missing_single_sites(partial_effects, strategy="position_mean")
        # Measured effects must be byte-identical
        for pos, effects in partial_effects.items():
            for aa, val in effects.items():
                assert out[pos][aa] == val

    def test_no_op_when_already_full(self):
        full = {
            10: {aa: 0.1 if aa != "A" else 0.0 for aa in AMINO_ACIDS},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # imputation should NOT warn here
            out = _impute_missing_single_sites(full, strategy="position_mean")
        assert out[10] == full[10]

    def test_position_with_only_wt_falls_back_to_zero(self):
        """The one documented fallback case: position with only WT measured has
        no useful prior, missing AAs all default to 0."""
        wt_only = {10: {"A": 0.0}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _impute_missing_single_sites(wt_only, strategy="position_mean")
        assert out[10]["A"] == 0.0
        for aa in AMINO_ACIDS:
            if aa != "A":
                assert out[10][aa] == 0.0

    def test_raise_strategy_lists_missing_aas(self, partial_effects):
        with pytest.raises(ValueError, match="missing single-site effects"):
            _impute_missing_single_sites(partial_effects, strategy="raise")

    def test_unknown_strategy_raises(self, partial_effects):
        with pytest.raises(ValueError, match="unknown missing_aa_prior strategy"):
            _impute_missing_single_sites(partial_effects, strategy="bogus")

    def test_warning_includes_imputed_count(self, partial_effects):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _impute_missing_single_sites(partial_effects, strategy="position_mean")
        msgs = [str(w.message) for w in caught if "imputed" in str(w.message)]
        assert msgs, "expected a warning about imputed entries"
        # 17 missing per position × 3 positions = 51 imputed
        assert "51" in msgs[0]

    def test_input_dict_not_mutated(self, partial_effects):
        snapshot = {pos: dict(effects) for pos, effects in partial_effects.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _impute_missing_single_sites(partial_effects, strategy="position_mean")
        assert partial_effects == snapshot


class TestFromDmsImputation:
    """Landscape.from_dms threads missing_aa_prior through the imputation."""

    def test_default_strategy_fills_all_20_aas(self, landscape):
        """The dms_csv fixture has only 2 measured AAs per position (G/V at 10,
        I/F at 20, R/E at 30). Default missing_aa_prior='position_mean' should
        give every position 20 AAs after construction."""
        for pos in landscape.positions:
            assert set(landscape.single_site_effects[pos].keys()) == set(AMINO_ACIDS)

    def test_raise_strategy_raises_on_partial_dms(self, small_contact_map, dms_csv):
        with pytest.raises(ValueError, match="missing single-site effects"):
            Landscape.from_dms(
                dms_path=dms_csv,
                contact_map=small_contact_map,
                epistasis=RandomEpistasis(sigma=0.0, seed=42),
                alpha=1.0,
                seed=42,
                missing_aa_prior="raise",
            )

    def test_imputed_single_mutant_evaluates_to_position_mean(self, landscape):
        """A single-mutant for an unmeasured AA at position 10 (e.g., A->R)
        evaluates to the same fitness as the position-mean prior under
        identity readout (the default in this fixture). Measured AAs
        keep their DMS value exactly."""
        # Measured AAs preserved
        assert landscape.evaluate({10: "G"}) == pytest.approx(0.5)
        # Position 10 measured non-WT: G=0.5, V=-0.3 → mean = 0.1
        # Unmeasured AA (e.g., R) should evaluate to that mean
        assert landscape.evaluate({10: "R"}) == pytest.approx(0.1, abs=1e-9)


class TestFromDmsWithMSACouplings:
    """Landscape.from_dms accepts MSACouplingsEpistasis directly or as a path."""

    @pytest.fixture
    def msa_coupling_npz(self, tmp_path):
        """Tiny (3, 3, 20, 20) MSA-Potts coupling tensor saved as npz.
        Use the contact map's positions ([10, 20, 30]) for the position_order."""
        rng = np.random.default_rng(0)
        J = rng.normal(0.0, 0.5, (3, 3, 20, 20)).astype(np.float64)
        # Symmetrise so J[i,j] = J[j,i].T (matches MSA construction)
        for i in range(3):
            for j in range(i + 1, 3):
                sym = 0.5 * (J[i, j] + J[j, i].T)
                J[i, j] = sym
                J[j, i] = sym.T
        npz = tmp_path / "test_msa_couplings.npz"
        np.savez(
            npz,
            coupling_matrix=J,
            positions=np.array([10, 20, 30], dtype=np.int64),
        )
        return npz

    def test_from_dms_accepts_msa_couplings_epistasis(
        self, small_contact_map, dms_csv, msa_coupling_npz
    ):
        """Landscape.from_dms accepts MSACouplingsEpistasis as the `epistasis`
        arg (the path-loading wrapper lives in build_landscape, not here)."""
        eps = MSACouplingsEpistasis.from_npz(msa_coupling_npz, scale=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc = Landscape.from_dms(
                dms_path=dms_csv,
                contact_map=small_contact_map,
                epistasis=eps,
                alpha=1.5,
                seed=42,
            )
        # Couplings should be populated for the contact pairs (10,20) and (20,30)
        assert lc.couplings.n_pairs == 2

    def test_calibration_build_landscape_threads_msa_path(
        self, dms_csv, msa_coupling_npz, tmp_path
    ):
        """The path-loading wrapper in build_landscape forwards to
        MSACouplingsEpistasis.from_npz."""
        from mutascape.validation.calibration import build_landscape

        # WT fasta with the right residues at positions 10/20/30 (1-based)
        seq = list("M" * 30)
        seq[9] = "A"  # pos 10
        seq[19] = "L"  # pos 20
        seq[29] = "K"  # pos 30
        wt_fasta = tmp_path / "wt.fasta"
        wt_fasta.write_text(">wt\n" + "".join(seq) + "\n")

        distances_npz = tmp_path / "distances.npz"
        np.savez(
            distances_npz,
            distances=np.array([[0.0, 6.0, 15.0], [6.0, 0.0, 7.0], [15.0, 7.0, 0.0]]),
            positions=np.array([10, 20, 30]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc = build_landscape(
                dms_csv=dms_csv,
                distances_npz=distances_npz,
                wt_fasta=wt_fasta,
                alpha=1.5,
                sigma=0.0,
                seed=42,
                msa_couplings=msa_coupling_npz,
            )
        # MSACouplingsEpistasis (loaded from path) was used; non-zero couplings expected
        assert lc.couplings.n_pairs == 2
