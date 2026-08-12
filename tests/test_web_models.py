"""Tests for web model storage, parsing, previews, and oracle queries."""

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from mutascape.contacts import ContactMap
from mutascape.web_models import (
    BenchmarkParameters,
    ModelStore,
    build_custom_landscape,
    custom_model_id,
    evaluate_entries,
    parse_custom_source,
    preview_custom,
    wilson_interval,
)

DMS = """position,wildtype,mutant,fitness
1,A,G,0.5
1,A,V,-0.2
2,C,A,0.3
2,C,D,-0.4
"""
FASTA = ">WT\nAC\n"
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
