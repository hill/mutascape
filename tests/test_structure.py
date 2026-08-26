"""Alignment and intake for the studio structure viewer."""

import pytest

from web.structure import _best_shift, describe_structure, fetch_rcsb, mean_effects

THREE_LETTER = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "V": "VAL",
    "W": "TRP",
}


def synthetic_pdb(chains: dict[str, tuple[str, int]]) -> str:
    """Render one CA-only PDB from chain sequences and their first residue number."""
    lines = []
    serial = 1
    for chain_id, (sequence, first) in chains.items():
        for offset, letter in enumerate(sequence):
            name = THREE_LETTER[letter]
            number = first + offset
            lines.append(
                f"ATOM  {serial:5d}  CA  {name} {chain_id}{number:4d}    "
                f"{offset * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_best_shift_finds_a_truncated_construct():
    assert _best_shift("GGACDEFGG", "ACDEF") == (2, 1.0)


def test_best_shift_finds_a_construct_longer_than_the_wild_type():
    shift, identity = _best_shift("ACDEF", "GGACDEFGG")
    assert shift == -2
    assert identity == 1.0


def test_best_shift_tolerates_a_point_difference():
    shift, identity = _best_shift("ACDEFGWVA", "ACDEFGWVC")
    assert shift == 0
    assert identity == pytest.approx(8 / 9)


def test_describe_structure_maps_positions_onto_the_deposited_numbering():
    text = synthetic_pdb({"A": ("ACDEF", 10)})
    described = describe_structure(
        text,
        suffix="pdb",
        wt_sequence="GGACDEFGG",
        positions=[3, 4, 9],
        wild_type={3: "A", 4: "C", 9: "G"},
        effects={3: -1.5, 4: 0.5},
    )
    assert described["chainId"] == "A"
    assert described["identity"] == 1.0
    assert [entry["residue"] for entry in described["residues"]] == [10, 11]
    assert [entry["effect"] for entry in described["residues"]] == [-1.5, 0.5]
    assert described["bound"] == 1.5
    assert described["mismatched"] == []
    # Position 9 sits past the construct, so it has no residue to draw.
    assert described["unmatched"] == [9]


def test_describe_structure_picks_the_longest_chain():
    text = synthetic_pdb({"A": ("AC", 1), "B": ("ACDEFG", 1)})
    described = describe_structure(
        text,
        suffix="pdb",
        wt_sequence="ACDEFG",
        positions=[1],
        wild_type={1: "A"},
    )
    assert described["chainId"] == "B"
    assert described["chains"] == ["A", "B"]
    assert described["chainLength"] == 6


def test_describe_structure_honours_an_explicit_chain():
    text = synthetic_pdb({"A": ("AC", 1), "B": ("ACDEFG", 1)})
    described = describe_structure(
        text,
        suffix="pdb",
        wt_sequence="ACDEFG",
        positions=[1],
        wild_type={1: "A"},
        chain_id="A",
    )
    assert described["chainId"] == "A"

    with pytest.raises(ValueError, match="chain Z"):
        describe_structure(
            text,
            suffix="pdb",
            wt_sequence="ACDEFG",
            positions=[1],
            wild_type={1: "A"},
            chain_id="Z",
        )


def test_describe_structure_flags_a_residue_the_structure_disagrees_with():
    text = synthetic_pdb({"A": ("AWDEF", 1)})
    described = describe_structure(
        text,
        suffix="pdb",
        wt_sequence="ACDEF",
        positions=[1, 2],
        wild_type={1: "A", 2: "C"},
    )
    assert described["mismatched"] == [2]
    assert described["residues"][1]["structureResidue"] == "W"


def test_describe_structure_rejects_an_unparsable_file():
    with pytest.raises(ValueError, match="no protein chain"):
        describe_structure(
            "not a structure\n",
            suffix="pdb",
            wt_sequence="ACDEF",
            positions=[1],
            wild_type={1: "A"},
        )


@pytest.mark.parametrize("identifier", ["", "1PG", "PGAX", "1pga1", "../etc"])
def test_fetch_rcsb_rejects_anything_that_is_not_a_pdb_id(identifier, tmp_path):
    with pytest.raises(ValueError, match="four characters"):
        fetch_rcsb(identifier, cache_dir=tmp_path)


def test_fetch_rcsb_serves_a_cached_entry_without_the_network(tmp_path):
    (tmp_path / "rcsb_1PGA.pdb").write_text("cached\n")
    text, suffix, identifier = fetch_rcsb("1pga", cache_dir=tmp_path)
    assert (text, suffix, identifier) == ("cached\n", "pdb", "1PGA")


def test_mean_effects_averages_every_measurement_at_a_position():
    rows = [
        {"position": 1, "fitness": 1.0},
        {"position": 1, "fitness": -3.0},
        {"position": 2, "fitness": 0.5},
    ]
    assert mean_effects(rows) == {1: -1.0, 2: 0.5}
