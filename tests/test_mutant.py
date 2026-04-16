"""Tests for the Mutant class."""

import pytest

from mutascape.mutant import Mutant


class TestParsing:
    def test_single_mutation(self):
        m = Mutant("A5G")
        assert m[5] == "G"
        assert m._wild_type[5] == "A"
        assert m.depth == 1

    def test_double_mutation(self):
        m = Mutant("A5G+L10V")
        assert m[5] == "G"
        assert m[10] == "V"
        assert m.depth == 2

    def test_triple_mutation(self):
        m = Mutant("A5G+L10V+K30R")
        assert m.depth == 3
        assert m.positions == [5, 10, 30]

    def test_wild_type(self):
        m = Mutant("wt")
        assert m.depth == 0
        assert m.label == "wt"

    def test_empty_string(self):
        m = Mutant("")
        assert m.depth == 0

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            Mutant("invalid")

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            Mutant("A-5G")


class TestDictConstructor:
    def test_basic(self):
        m = Mutant({5: "G", 10: "V"}, wild_type={5: "A", 10: "L"})
        assert m[5] == "G"
        assert m.label == "A5G+L10V"

    def test_without_wild_type(self):
        m = Mutant({5: "G"})
        assert m[5] == "G"
        assert m.label == "?5G"

    def test_empty_dict(self):
        m = Mutant({})
        assert m.depth == 0
        assert m.label == "wt"

    def test_from_dict_alias(self):
        m = Mutant.from_dict({5: "G"}, {5: "A"})
        assert m.label == "A5G"

    def test_dict_and_string_produce_same_result(self):
        from_str = Mutant("A5G+L10V")
        from_dict = Mutant({5: "G", 10: "V"}, wild_type={5: "A", 10: "L"})
        assert dict(from_str) == dict(from_dict)
        assert from_str.label == from_dict.label

    def test_partial_wild_type(self):
        """WT info only for some positions."""
        m = Mutant({5: "G", 10: "V"}, wild_type={5: "A"})
        assert m.label == "A5G+?10V"


class TestLabel:
    def test_sorted_by_position(self):
        m = Mutant("L10V+A5G")
        assert m.label == "A5G+L10V"

    def test_roundtrip(self):
        original = "A5G+L10V+K30R"
        m = Mutant(original)
        assert m.label == original


class TestOperators:
    def test_add(self):
        a = Mutant("A5G")
        b = Mutant("L10V")
        c = a + b
        assert c.depth == 2
        assert c[5] == "G"
        assert c[10] == "V"
        assert c.label == "A5G+L10V"

    def test_add_preserves_wild_type(self):
        c = Mutant("A5G") + Mutant("L10V")
        assert c._wild_type[5] == "A"
        assert c._wild_type[10] == "L"

    def test_add_conflict_raises(self):
        with pytest.raises(ValueError, match="Conflicting positions"):
            Mutant("A5G") + Mutant("A5V")

    def test_subtract(self):
        m = Mutant("A5G+L10V+K30R")
        result = m - Mutant("L10V")
        assert result.depth == 2
        assert result.label == "A5G+K30R"

    def test_subtract_non_overlapping(self):
        m = Mutant("A5G+L10V")
        result = m - Mutant("K30R")
        assert result.depth == 2  # nothing removed

    def test_chain_operators(self):
        m = Mutant("A5G") + Mutant("L10V") + Mutant("K30R")
        assert m.depth == 3
        result = m - Mutant("L10V")
        assert result.depth == 2


class TestDictBehaviour:
    def test_is_a_dict(self):
        m = Mutant("A5G")
        assert isinstance(m, dict)

    def test_iteration(self):
        m = Mutant("A5G+L10V")
        assert set(m.keys()) == {5, 10}
        assert set(m.values()) == {"G", "V"}

    def test_items(self):
        m = Mutant("A5G+L10V")
        assert set(m.items()) == {(5, "G"), (10, "V")}

    def test_len(self):
        assert len(Mutant("A5G+L10V")) == 2


class TestRepr:
    def test_repr(self):
        assert repr(Mutant("A5G+L10V")) == "Mutant('A5G+L10V')"

    def test_str(self):
        assert str(Mutant("A5G+L10V")) == "A5G+L10V"

    def test_wt_repr(self):
        assert repr(Mutant("wt")) == "Mutant('wt')"

    def test_repr_from_dict(self):
        m = Mutant({5: "G"}, wild_type={5: "A"})
        assert repr(m) == "Mutant('A5G')"


class TestSynonymousMutations:
    def test_single_synonymous_becomes_wildtype(self):
        m = Mutant("A5A")
        assert m.depth == 0
        assert m.label == "wt"

    def test_synonymous_in_combo_is_dropped(self):
        m = Mutant("A5G+R10R+L20V")
        assert m.depth == 2
        assert m.label == "A5G+L20V"
        assert 10 not in m

    def test_all_synonymous_becomes_wildtype(self):
        m = Mutant("A5A+R10R")
        assert m.depth == 0
        assert m.label == "wt"

    def test_synonymous_not_in_wild_type_dict(self):
        m = Mutant("A5G+R10R")
        assert 10 not in m._wild_type


class TestEdgeCases:
    def test_whitespace_in_label(self):
        m = Mutant("A5G + L10V")
        assert m.depth == 2
        assert m[5] == "G"

    def test_single_position_add_then_subtract(self):
        """Adding and removing the same mutation returns wt."""
        m = Mutant("A5G")
        result = m - Mutant("A5G")
        assert result.depth == 0
        assert result.label == "wt"

    def test_subtract_returns_mutant_type(self):
        result = Mutant("A5G+L10V") - Mutant("A5G")
        assert isinstance(result, Mutant)

    def test_add_returns_mutant_type(self):
        result = Mutant("A5G") + Mutant("L10V")
        assert isinstance(result, Mutant)
