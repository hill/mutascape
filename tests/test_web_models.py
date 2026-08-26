"""Tests for web model storage, parsing, previews, and oracle queries."""

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from mutascape.contacts import ContactMap
from web.models import (
    BenchmarkParameters,
    ModelStore,
    build_custom_landscape,
    custom_model_id,
    evaluate_entries,
    inspect_dms_csv,
    parse_custom_source,
    preview_custom,
    score_histogram,
    substitution_map,
    wilson_interval,
)

MAVEDB = """accession,hgvs_nt,hgvs_pro,score,std
urn:x#1,NA,p.Ala1Ter,-1.1,0.4
urn:x#2,NA,p.Ala1Gly,0.5,0.4
urn:x#3,NA,p.Ala1Val,-0.2,0.3
urn:x#4,NA,p.Cys2Ala,0.3,0.2
urn:x#5,NA,p.Cys2Asp,-0.4,0.2
urn:x#6,NA,p.Cys2=,0.0,0.1
urn:x#7,NA,p.Ala1delinsGly,0.9,0.1
urn:x#8,NA,p.Ala1Gly,0.7,0.1
urn:x#9,NA,p.Cys2Glu,NA,0.1
urn:x#10,NA,p.[Ala1Gly;Cys2Ala],1.4,0.2
"""

DMS = """position,wildtype,mutant,fitness
1,A,G,0.5
1,A,V,-0.2
2,C,A,0.3
2,C,D,-0.4
"""
FASTA = ">WT\nAC\n"
VARIANT_DMS = """variant,fitness
WT,1.0
A1G,1.5
A1V,0.8
C2D,0.5
A1G+C2D,2.1
"""
PDB = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CB  ALA A   1       0.000   1.000   0.000  1.00 20.00           C
ATOM      3  CA  CYS A   2       0.000   0.000   6.000  1.00 20.00           C
ATOM      4  CB  CYS A   2       0.000   1.000   6.000  1.00 20.00           C
TER
END
"""


@pytest.fixture
def source():
    parsed, _ = parse_custom_source(DMS, FASTA)
    return parsed


def test_parse_custom_source_reports_imputation_and_contacts():
    parsed, summary = parse_custom_source(DMS, FASTA, PDB, "pdb")

    assert summary["positionCount"] == 2
    assert summary["imputedEffects"] == 34
    assert summary["canFitNonlinearity"] is True
    assert summary["contactCount"] == 1
    assert summary["structureMethod"] == "uploaded"
    assert parsed["contactPairs"] == [[1, 2]]


def test_parse_custom_source_can_infer_esmfold(monkeypatch, tmp_path):
    calls = []

    def fake_from_sequence(sequence, threshold=8.0, cache_dir=None):
        calls.append((sequence, threshold, cache_dir))
        return ContactMap(
            matrix=np.array([[False, True], [True, False]]),
            positions=(1, 2),
            threshold=threshold,
        )

    monkeypatch.setattr(ContactMap, "from_sequence", staticmethod(fake_from_sequence))
    parsed, summary = parse_custom_source(
        DMS,
        FASTA,
        structure_inference="esmfold",
        structure_cache_dir=tmp_path,
    )

    assert calls == [("AC", 8.0, tmp_path)]
    assert summary["structureMethod"] == "esmfold"
    assert summary["contactCount"] == 1
    assert parsed["hasStructure"] is True


def test_parse_custom_source_rejects_two_structure_sources():
    with pytest.raises(ValueError, match="either an uploaded structure or ESMFold"):
        parse_custom_source(DMS, FASTA, PDB, "pdb", structure_inference="esmfold")


def test_parse_custom_source_rejects_wild_type_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        parse_custom_source(DMS.replace("1,A,G", "1,C,G"), FASTA)


def test_parse_custom_source_accepts_variant_rows_and_preserves_doubles():
    parsed, summary = parse_custom_source(VARIANT_DMS, FASTA)

    assert summary["dmsSchema"] == "variants"
    assert summary["wildTypeFitness"] == 1.0
    assert summary["higherOrderRows"] == 1
    assert parsed["dmsRows"][0]["fitness"] == pytest.approx(0.5)
    double = next(row for row in parsed["observationRows"] if row["depth"] == 2)
    assert double["variant"] == "A1G+C2D"
    assert double["fitness"] == pytest.approx(1.1)
    wild_type_row = next(row for row in parsed["observationRows"] if row["depth"] == 0)
    assert wild_type_row["fitness"] == 0.0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("variant,fitness\nA1G,1.5\n", "include a WT row"),
        (
            "variant,fitness\nWT,1.0\nA1G+C2D,2.0\nA1G,1.5\n",
            "missing constituent singles",
        ),
        ("variant,fitness\nWT,1.0\nWT,1.1\nA1G,1.5\n", "duplicate measurement"),
    ],
)
def test_variant_dms_rejects_incomplete_or_duplicate_rows(text, message):
    with pytest.raises(ValueError, match=message):
        parse_custom_source(text, FASTA)


def test_custom_preview_is_deterministic_and_synthetic_only(source):
    parameters = BenchmarkParameters(seed=17)
    first = preview_custom(source, parameters, 20, {1: 1, 2: 1})
    second = preview_custom(source, parameters, 20, {1: 1, 2: 1})

    assert first == second
    assert first["comparisonAvailable"] is False
    assert len(first["sample"]["rows"]) == 20
    assert {row["depth"] for row in first["sample"]["rows"]} == {1, 2}
    graph = first["sample"]["graph"]
    assert graph["nodes"][0]["id"] == "wt"
    assert graph["edges"]
    node_ids = {node["id"] for node in graph["nodes"]}
    assert all(edge["source"] in node_ids for edge in graph["edges"])
    assert all(edge["target"] in node_ids for edge in graph["edges"])


def test_content_id_ignores_display_name(source):
    parameters = BenchmarkParameters()
    assert custom_model_id(source, parameters) == custom_model_id(source, parameters)
    assert custom_model_id(source, parameters) != custom_model_id(
        source,
        BenchmarkParameters(seed=99),
    )


def test_store_publishes_once_and_survives_reopen(tmp_path, source):
    path = tmp_path / "models.sqlite3"
    store = ModelStore(path)
    draft = store.create_draft(source, {"dmsCsv": DMS, "wtFasta": FASTA})

    first, created = store.publish(draft["id"], "Example", BenchmarkParameters())
    second, created_again = store.publish(draft["id"], "Renamed", BenchmarkParameters())

    assert created is True
    assert created_again is False
    assert first["id"] == second["id"]
    assert second["name"] == "Example"
    reopened = ModelStore(path)
    assert reopened.get_model(first["id"])["name"] == "Example"


def test_store_persists_posterior_definition(tmp_path, source):
    store = ModelStore(tmp_path / "models.sqlite3")
    draft = store.create_draft(source, {"dmsCsv": DMS, "wtFasta": FASTA})

    model, created = store.publish(
        draft["id"],
        "Posterior example",
        BenchmarkParameters(seed=17),
        model_kind="posterior",
        prior_id="combinatorial-v2",
        oracle_seed=17,
    )

    assert created is True
    assert model["modelKind"] == "posterior"
    assert model["priorId"] == "combinatorial-v2"
    assert model["oracleSeed"] == 17


def test_store_removes_expired_draft(tmp_path, source):
    store = ModelStore(tmp_path / "models.sqlite3")
    draft = store.create_draft(source, {"dmsCsv": DMS, "wtFasta": FASTA})
    expired = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE drafts SET expires_at = ? WHERE id = ?",
            (expired, draft["id"]),
        )

    assert store.get_draft(draft["id"]) is None


def test_store_ranks_leaderboard_entries_and_reports_intervals(tmp_path):
    store = ModelStore(tmp_path / "models.sqlite3")
    board = store.create_leaderboard("GB1 search", "gb1_headline", 1000)
    store.add_leaderboard_entry(board["id"], "Model A", 6, 10)
    store.add_leaderboard_entry(board["id"], "Model B", 9, 10)

    saved = store.get_leaderboard(board["id"])

    assert saved is not None
    assert [entry["modelName"] for entry in saved["entries"]] == ["Model B", "Model A"]
    assert saved["entries"][0]["successRate"] == 0.9
    assert saved["entries"][0]["confidenceLow"] < 0.9
    assert saved["entries"][0]["confidenceHigh"] > 0.9
    assert wilson_interval(0, 10)[0] == 0


def test_oracle_accepts_labels_and_compact_sequences(source):
    landscape = build_custom_landscape(source, BenchmarkParameters())
    results = evaluate_entries(
        landscape,
        [
            {"mutationLabel": "A1G", "compactSequence": None},
            {"mutationLabel": None, "compactSequence": "GD"},
        ],
    )

    assert [result["index"] for result in results] == [0, 1]
    assert results[0]["mutationLabel"] == "A1G"
    assert results[1]["mutationLabel"] == "A1G+C2D"


def test_oracle_rejects_unknown_position(source):
    landscape = build_custom_landscape(source, BenchmarkParameters())
    with pytest.raises(ValueError, match="not mutable"):
        evaluate_entries(
            landscape,
            [{"mutationLabel": "A3G", "compactSequence": None}],
        )


def test_substitution_map_fills_the_grid_and_centres_the_scale(source):
    grid = substitution_map(source)

    assert grid["positions"] == [1, 2]
    assert grid["aminoAcids"] == "ACDEFGHIKLMNPQRSTVWY"
    assert grid["wildTypeCompact"] == "AC"
    assert grid["wildTypeCells"] == [[0, 0], [1, 1]]
    assert grid["measuredCount"] == 4
    assert grid["imputedCount"] == 34
    assert grid["measuredCount"] + grid["imputedCount"] + len(grid["wildTypeCells"]) == 40
    assert grid["bound"] == pytest.approx(0.5)
    assert grid["truncated"] is False
    assert grid["totalPositions"] == 2

    measured = {(column, row): value for column, row, value in grid["measured"]}
    assert measured[(0, 5)] == pytest.approx(0.5)


def test_inspect_detects_the_mavedb_layout():
    report = inspect_dms_csv(MAVEDB)

    assert report["detected"] == {
        "schema": "hgvs",
        "columns": {"hgvs": "hgvs_pro", "score": "score"},
    }
    assert report["rowCount"] == 10
    assert report["headers"][0] == "accession"
    assert len(report["sampleRows"][0]) == len(report["headers"])


def test_inspect_reports_roles_when_nothing_matches():
    report = inspect_dms_csv("a,b\n1,2\n")

    assert report["detected"] is None
    assert {entry["id"] for entry in report["schemas"]} == {"singles", "variants", "hgvs"}
    assert all(entry["complete"] is False for entry in report["schemas"])


def test_mavedb_rows_parse_and_report_what_was_dropped():
    parsed, summary = parse_custom_source(MAVEDB, FASTA)

    assert summary["dmsSchema"] == "hgvs"
    # p.Cys2= is synonymous, so it fixes the wild-type score rather than being guessed.
    assert summary["wildTypeScoreAssumed"] is False
    assert summary["skippedRows"] == {
        "nonsense": 1,
        "indel or frameshift": 1,
        "duplicate": 1,
        "no score": 1,
    }
    assert {row["mutant"] for row in parsed["dmsRows"] if row["position"] == 1} == {"G", "V"}
    double = next(row for row in parsed["observationRows"] if row["depth"] == 2)
    assert double["variant"] == "A1G+C2A"
    assert double["fitness"] == pytest.approx(1.4)


def test_mavedb_wild_type_score_shifts_every_observation():
    parsed, _ = parse_custom_source(MAVEDB, FASTA, wild_type_score=0.5)

    single = next(row for row in parsed["dmsRows"] if row["mutant"] == "G")
    assert single["fitness"] == pytest.approx(0.0)


def test_mavedb_assumes_zero_when_no_row_pins_the_wild_type():
    parsed, summary = parse_custom_source(
        MAVEDB.replace("p.Cys2=,0.0,0.1", "p.Cys2Gly,0.6,0.1"), FASTA
    )

    assert summary["wildTypeScoreAssumed"] is True
    assert summary["wildTypeFitness"] == 0.0
    assert next(row["fitness"] for row in parsed["dmsRows"] if row["mutant"] == "G") == 0.5


def test_mavedb_drops_rows_whose_parent_contradicts_the_fasta():
    _, summary = parse_custom_source(MAVEDB.replace("Ala1", "Trp1"), FASTA)

    assert summary["skippedRows"]["wild type W1 does not match the WT FASTA"] == 4
    assert summary["positions"] == [2]


def test_column_map_reads_an_arbitrary_layout():
    text = "site,from,to,effect\n1,A,G,0.5\n2,C,D,-0.4\n"
    column_map = {
        "schema": "singles",
        "columns": {
            "position": "site",
            "wildtype": "from",
            "mutant": "to",
            "fitness": "effect",
        },
    }

    parsed, summary = parse_custom_source(text, FASTA, column_map=column_map)

    assert summary["dmsSchema"] == "singles"
    assert [row["mutant"] for row in parsed["dmsRows"]] == ["G", "D"]


def test_column_map_rejects_a_header_the_csv_lacks():
    column_map = {"schema": "hgvs", "columns": {"hgvs": "missing", "score": "score"}}

    with pytest.raises(ValueError, match="not in the CSV"):
        parse_custom_source(MAVEDB, FASTA, column_map=column_map)


def test_score_histogram_marks_wild_type_at_zero(source):
    histogram = score_histogram(source, bins=8)

    assert len(histogram["centres"]) == 8
    assert sum(histogram["singles"]) == 4
    assert histogram["higherOrderCount"] == 0
    assert histogram["wildType"] == 0.0
    assert histogram["min"] == pytest.approx(-0.4)


def test_substitution_map_sequence_scope_keeps_unassayed_columns():
    rows = "position,wildtype,mutant,fitness\n2,C,D,-0.4\n"
    parsed, _ = parse_custom_source(rows, ">WT\nACD\n")

    grid = substitution_map(parsed, scope="sequence")

    assert grid["positions"] == [1, 2, 3]
    assert grid["assayed"] == [1]
    assert grid["assayedCount"] == 1
    assert grid["sequenceLength"] == 3
    assert {column for column, _, _ in grid["measured"] + grid["imputed"]} == {1}
    assert len(grid["wildTypeCells"]) == 3


def test_substitution_map_truncates_wide_scans():
    rows = ["position,wildtype,mutant,fitness"]
    rows += [f"{position},A,G,0.1" for position in range(1, 7)]
    parsed, _ = parse_custom_source("\n".join(rows) + "\n", ">WT\n" + "A" * 6 + "\n")

    grid = substitution_map(parsed, max_positions=4)

    assert grid["positions"] == [1, 2, 3, 4]
    assert grid["totalPositions"] == 6
    assert grid["truncated"] is True
