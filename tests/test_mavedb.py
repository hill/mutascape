"""Tests for the MaveDB score-set client."""

import io
import json
import urllib.error

import pytest

from web.mavedb import MaveDbError, fetch_score_set, parse_urn

SCORE_SET = {
    "title": "Example score set",
    "numVariants": 3,
    "license": {"shortName": "CC0"},
    "primaryPublicationIdentifiers": [{"url": "https://example.org/paper"}],
    "targetGenes": [
        {
            "name": "BRCA1",
            "targetSequence": {"sequence": "acdefg", "sequenceType": "protein"},
        }
    ],
}
SCORES = "accession,hgvs_pro,score\nurn:x#1,p.Ala1Gly,0.5\n"


def opener_for(payloads):
    def opener(request, timeout=None):
        for fragment, body in payloads.items():
            if request.full_url.endswith(fragment):
                return io.BytesIO(body.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    return opener


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("urn:mavedb:00001222-b-2", "urn:mavedb:00001222-b-2"),
        ("https://www.mavedb.org/score-sets/urn:mavedb:00001222-b-2", "urn:mavedb:00001222-b-2"),
        (
            "https://www.mavedb.org/score-sets/urn:mavedb:00001222-b-2"
            "?calibration=urn:mavedb:calibration-6a913e98&variant=urn:mavedb:00001222-b-2%231313",
            "urn:mavedb:00001222-b-2",
        ),
    ],
)
def test_parse_urn_accepts_urns_and_record_urls(value, expected):
    assert parse_urn(value) == expected


def test_parse_urn_rejects_anything_else():
    with pytest.raises(MaveDbError, match="cannot find a MaveDB URN"):
        parse_urn("https://example.org/not-mavedb")


def test_fetch_returns_a_fasta_and_the_scores_csv():
    opener = opener_for(
        {"/scores": SCORES, "urn:mavedb:00001222-b-2": json.dumps(SCORE_SET)},
    )

    record = fetch_score_set("urn:mavedb:00001222-b-2", opener=opener)

    assert record["dmsCsv"] == SCORES
    assert record["wtFasta"] == ">BRCA1 urn:mavedb:00001222-b-2\nACDEFG\n"
    assert record["targetName"] == "BRCA1"
    assert record["license"] == "CC0"
    assert record["publications"] == ["https://example.org/paper"]


def test_fetch_rejects_a_nucleotide_target():
    nucleotide = {
        **SCORE_SET,
        "targetGenes": [
            {"name": "X", "targetSequence": {"sequence": "acgt", "sequenceType": "dna"}}
        ],
    }
    opener = opener_for({"urn:mavedb:1": json.dumps(nucleotide)})

    with pytest.raises(MaveDbError, match="needs a protein target"):
        fetch_score_set("urn:mavedb:1", opener=opener)


def test_fetch_reports_a_missing_record():
    with pytest.raises(MaveDbError, match="no published record"):
        fetch_score_set("urn:mavedb:1", opener=opener_for({}))
