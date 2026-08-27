"""HTTP contract tests for the benchmark oracle service."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from web.server import create_app

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
A1V,1.2
C2D,0.5
A1G+C2D,2.1
"""
PARAMETERS = {
    "noiseSigma": 0.3,
    "alpha": 1.5,
    "pairwiseSigma": 0,
    "combinatorialUpside": 3,
    "seed": 42,
}


def test_custom_draft_preview_publish_and_evaluate(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))

    draft_response = client.post(
        "/api/v1/drafts",
        json={"dmsCsv": DMS, "wtFasta": FASTA},
    )
    assert draft_response.status_code == 201
    draft = draft_response.json()
    assert draft["summary"]["positionCount"] == 2

    preview_payload = {
        "source": {"kind": "draft", "id": draft["id"]},
        "parameters": PARAMETERS,
        "sample": {"size": 12, "depthWeights": {"1": 1, "2": 1}},
    }
    preview = client.post("/api/v1/previews", json=preview_payload)
    assert preview.status_code == 200
    assert preview.json()["comparisonAvailable"] is False
    assert preview.json()["parameters"] == PARAMETERS
    assert preview.json()["sample"]["graph"]["nodes"][0]["id"] == "wt"
    cached = client.post("/api/v1/previews", json=preview_payload)
    assert cached.json()["cached"] is True

    queried = client.post(
        "/api/v1/evaluations",
        json={
            "source": {"kind": "draft", "id": draft["id"]},
            "parameters": PARAMETERS,
            "entries": [{"mutationLabel": "A1G+C2D"}],
        },
    )
    assert queried.status_code == 200
    assert queried.json()["results"][0]["mutationLabel"] == "A1G+C2D"
    assert isinstance(queried.json()["results"][0]["fitness"], float)

    published = client.post(
        "/api/v1/models",
        json={"draftId": draft["id"], "name": "Example oracle", "parameters": PARAMETERS},
    )
    assert published.status_code == 201
    assert published.json()["parameters"] == PARAMETERS
    model_id = published.json()["id"]

    catalogue = client.get("/api/v1/models").json()["models"]
    assert {model["id"] for model in catalogue} >= {
        "creilov_dms_pairwise",
        "gb1_headline",
        "trpb_headline",
        model_id,
    }
    assert all(model["parameters"].keys() == PARAMETERS.keys() for model in catalogue)
    detail = client.get(f"/api/v1/models/{model_id}").json()
    assert "uploads" not in detail
    assert "dmsRows" not in detail
    assert "np.zeros" in detail["examples"]["localPython"]
    assert "A1G" in detail["examples"]["hostedPython"]
    compile(detail["examples"]["localPython"], "<local-example>", "exec")
    compile(detail["examples"]["hostedPython"], "<hosted-example>", "exec")

    reference_detail = client.get("/api/v1/models/gb1_headline").json()
    assert "V39A" in reference_detail["examples"]["hostedPython"]
    compile(reference_detail["examples"]["localPython"], "<reference-example>", "exec")

    posterior_detail = client.get("/api/v1/models/gb1_posterior").json()
    assert posterior_detail["kind"] == "posterior"
    assert posterior_detail["posteriorSettings"]["priorId"] == "combinatorial-v2"
    compile(posterior_detail["examples"]["localPython"], "<posterior-example>", "exec")
    posterior_query = client.post(
        "/api/v1/posterior-queries",
        json={
            "source": {"kind": "reference", "id": "gb1_posterior"},
            "priorId": "combinatorial-v2",
            "seed": 42,
            "ensembleSize": 8,
            "entries": [{"mutationLabel": "V39A+D40A"}],
        },
    )
    assert posterior_query.status_code == 200
    assert posterior_query.json()["results"][0]["posteriorStd"] > 0

    creilov_detail = client.get("/api/v1/models/creilov_dms_pairwise").json()
    assert creilov_detail["kind"] == "posterior"
    assert len(creilov_detail["positions"]) == 118
    assert creilov_detail["sourceSummary"]["observationCount"] == 2_122
    assert "2,121 measured single mutants" in creilov_detail["description"]

    changed_parameters = {**PARAMETERS, "seed": 43}
    changed_preview = client.post(
        "/api/v1/previews",
        json={
            "source": {"kind": "model", "id": model_id},
            "parameters": changed_parameters,
            "sample": {"size": 4, "depthWeights": {"1": 1}},
        },
    )
    assert changed_preview.status_code == 422

    evaluated = client.post(
        f"/api/v1/models/{model_id}/evaluate",
        json={
            "entries": [
                {"mutationLabel": "A1G"},
                {"compactSequence": "GD"},
            ]
        },
    )
    assert evaluated.status_code == 200
    assert [row["index"] for row in evaluated.json()["results"]] == [0, 1]


def test_upload_validation_error_is_structured(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    response = client.post(
        "/api/v1/drafts",
        json={"dmsCsv": DMS, "wtFasta": ">WT\nXX\n"},
    )

    assert response.status_code == 422
    assert "unsupported residues" in response.json()["detail"]


def test_posterior_query_preview_publish_and_fixed_evaluation(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    priors = client.get("/api/v1/priors")
    assert priors.status_code == 200
    assert priors.json()["defaultPriorId"] == "combinatorial-v2"

    draft = client.post(
        "/api/v1/drafts",
        json={"dmsCsv": VARIANT_DMS, "wtFasta": FASTA},
    ).json()
    source = {"kind": "draft", "id": draft["id"]}
    query_payload = {
        "source": source,
        "priorId": "combinatorial-v2",
        "seed": 17,
        "ensembleSize": 20,
        "entries": [{"mutationLabel": "a1v:c2d"}],
    }
    query = client.post("/api/v1/posterior-queries", json=query_payload)
    assert query.status_code == 200
    result = query.json()["results"][0]
    assert result["mutationLabel"] == "A1V+C2D"
    assert result["observed"] is False
    assert result["posteriorStd"] > 0
    assert len(result["samples"]) == 20

    preview = client.post(
        "/api/v1/posterior-previews",
        json={
            "source": source,
            "priorId": "combinatorial-v2",
            "seed": 17,
            "ensembleSize": 12,
            "sample": {"size": 5, "depthWeights": {"1": 1, "2": 1}},
        },
    )
    assert preview.status_code == 200
    assert preview.json()["sample"]["graph"]["nodes"]

    published = client.post(
        "/api/v1/models",
        json={
            "draftId": draft["id"],
            "name": "Posterior example",
            "parameters": {**PARAMETERS, "seed": 17},
            "modelType": "posterior",
            "priorId": "combinatorial-v2",
            "oracleSeed": 17,
        },
    )
    assert published.status_code == 201
    model = published.json()
    assert model["kind"] == "posterior"
    assert model["posteriorSettings"]["oracleSeed"] == 17

    fixed = client.post(
        f"/api/v1/models/{model['id']}/evaluate",
        json={"entries": [{"mutationLabel": "A1V+C2D"}]},
    )
    assert fixed.status_code == 200
    assert fixed.json()["results"][0]["fitness"] == result["fixedOracleFitness"]

    changed = client.post(
        "/api/v1/posterior-queries",
        json={**query_payload, "source": {"kind": "model", "id": model["id"]}, "seed": 18},
    )
    assert changed.status_code == 422


def test_evaluate_rejects_ambiguous_entry(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    response = client.post(
        "/api/v1/models/gb1_headline/evaluate",
        json={"entries": [{"mutationLabel": "V39A", "compactSequence": "ADGV"}]},
    )

    assert response.status_code == 422


def test_repository_logo_is_served(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    response = client.get("/assets/logo.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_frontend_routes_support_direct_loading(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))

    for path in (
        "/",
        "/landscapes",
        "/leaderboards",
        "/landscapes/gb1_headline",
        "/docs",
        "/docs/benchmarking",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert '<div id="root"></div>' in response.text
        assert 'href="/dist/main.css"' in response.text
        assert 'src="/dist/main.js"' in response.text


def test_documentation_figure_is_served(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    response = client.get("/assets/docs/oracle-profile.svg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    assert b"<svg" in response.content


def test_leaderboard_create_and_add_result(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    created = client.post(
        "/api/v1/leaderboards",
        json={
            "name": "GB1 search",
            "landscapeId": "gb1_headline",
            "queryBudget": 1000,
        },
    )

    assert created.status_code == 201
    board = created.json()
    result = client.post(
        f"/api/v1/leaderboards/{board['id']}/entries",
        json={"modelName": "My optimiser", "successes": 8, "trials": 10},
    )
    assert result.status_code == 201
    assert result.json()["successRate"] == 0.8

    boards = client.get("/api/v1/leaderboards").json()["leaderboards"]
    assert boards[0]["queryBudget"] == 1000
    assert boards[0]["entries"][0]["modelName"] == "My optimiser"


def test_leaderboard_rejects_invalid_result(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    response = client.post(
        "/api/v1/leaderboards/missing/entries",
        json={"modelName": "Broken", "successes": 11, "trials": 10},
    )

    assert response.status_code == 422


def test_cached_preview_does_not_extend_draft_lifetime(tmp_path):
    database = tmp_path / "web.sqlite3"
    client = TestClient(create_app(database))
    draft = client.post(
        "/api/v1/drafts",
        json={"dmsCsv": DMS, "wtFasta": FASTA},
    ).json()
    payload = {
        "source": {"kind": "draft", "id": draft["id"]},
        "parameters": PARAMETERS,
        "sample": {"size": 4, "depthWeights": {"1": 1}},
    }
    assert client.post("/api/v1/previews", json=payload).status_code == 200

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE drafts SET expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", draft["id"]),
        )

    response = client.post("/api/v1/previews", json=payload)
    assert response.status_code == 404


def test_substitution_map_serves_every_source_kind(tmp_path):
    database = tmp_path / "web.sqlite3"
    client = TestClient(create_app(database))
    draft = client.post(
        "/api/v1/drafts",
        json={"dmsCsv": DMS, "wtFasta": FASTA},
    ).json()

    drafted = client.post(
        "/api/v1/substitution-maps",
        json={"source": {"kind": "draft", "id": draft["id"]}},
    )
    assert drafted.status_code == 200
    grid = drafted.json()
    assert grid["positions"] == [1, 2]
    assert grid["wildTypeCompact"] == "AC"
    assert grid["measuredCount"] == 4
    assert grid["imputedCount"] == 0
    assert grid["truncated"] is False

    reference = client.post(
        "/api/v1/substitution-maps",
        json={"source": {"kind": "reference", "id": "gb1_headline"}},
    )
    assert reference.status_code == 200
    assert reference.json()["positions"] == [39, 40, 41, 54]

    model_id = client.post(
        "/api/v1/models",
        json={"draftId": draft["id"], "name": "Grid oracle", "parameters": PARAMETERS},
    ).json()["id"]
    published = client.post(
        "/api/v1/substitution-maps",
        json={"source": {"kind": "model", "id": model_id}},
    )
    assert published.status_code == 200
    assert published.json()["positions"] == [1, 2]

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE drafts SET expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", draft["id"]),
        )
    expired = client.post(
        "/api/v1/substitution-maps",
        json={"source": {"kind": "draft", "id": draft["id"]}},
    )
    assert expired.status_code == 404


def test_uploaded_structure_is_aligned_to_the_wild_type(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    draft = client.post("/api/v1/drafts", json={"dmsCsv": DMS, "wtFasta": FASTA}).json()

    pdb = (
        "ATOM      1  CA  ALA A   7       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  CYS A   8       3.800   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    response = client.post(
        "/api/v1/structures",
        json={
            "source": {"kind": "draft", "id": draft["id"]},
            "kind": "upload",
            "structureText": pdb,
            "structureFormat": "pdb",
        },
    )
    assert response.status_code == 200
    view = response.json()
    assert view["method"] == "upload"
    assert view["label"] == "Uploaded file"
    assert view["chainId"] == "A"
    assert view["identity"] == 1.0
    assert [entry["residue"] for entry in view["residues"]] == [7, 8]
    assert view["residues"][0]["effect"] == pytest.approx(0.15)
    assert view["unmatched"] == []


def test_structure_request_rejects_a_bad_pdb_id_and_a_missing_upload(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    draft = client.post("/api/v1/drafts", json={"dmsCsv": DMS, "wtFasta": FASTA}).json()
    source = {"kind": "draft", "id": draft["id"]}

    bad_id = client.post(
        "/api/v1/structures",
        json={"source": source, "kind": "rcsb", "pdbId": "nope"},
    )
    assert bad_id.status_code == 422
    assert "four characters" in bad_id.json()["detail"]

    empty = client.post("/api/v1/structures", json={"source": source, "kind": "upload"})
    assert empty.status_code == 422
    assert empty.json()["detail"] == "upload a structure file"


MAVEDB_CSV = """accession,hgvs_pro,score
urn:x#1,p.Ala1Gly,0.5
urn:x#2,p.Ala1Val,-0.2
urn:x#3,p.Cys2Ala,0.3
urn:x#4,p.Cys2Ter,-1.0
"""


def test_csv_inspection_reports_headers_and_a_guess():
    client = TestClient(create_app())

    response = client.post("/api/v1/csv-inspections", json={"dmsCsv": MAVEDB_CSV})

    assert response.status_code == 200
    report = response.json()
    assert report["headers"] == ["accession", "hgvs_pro", "score"]
    assert report["detected"]["schema"] == "hgvs"
    assert report["rowCount"] == 4

    unmapped = client.post("/api/v1/csv-inspections", json={"dmsCsv": "a,b\n1,2\n"})
    assert unmapped.json()["detected"] is None


def test_draft_accepts_a_hand_mapped_csv(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))

    response = client.post(
        "/api/v1/drafts",
        json={
            "dmsCsv": "site,from,to,effect\n1,A,G,0.5\n2,C,D,-0.4\n",
            "wtFasta": FASTA,
            "columnMap": {
                "schema": "singles",
                "columns": {
                    "position": "site",
                    "wildtype": "from",
                    "mutant": "to",
                    "fitness": "effect",
                },
            },
        },
    )

    assert response.status_code == 201
    assert response.json()["summary"]["positionCount"] == 2


def test_draft_rejects_a_column_map_the_csv_cannot_satisfy(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))

    response = client.post(
        "/api/v1/drafts",
        json={
            "dmsCsv": MAVEDB_CSV,
            "wtFasta": FASTA,
            "columnMap": {"schema": "hgvs", "columns": {"hgvs": "nope", "score": "score"}},
        },
    )

    assert response.status_code == 422
    assert "not in the CSV" in response.json()["detail"]


def test_score_histogram_and_sequence_scope(tmp_path):
    client = TestClient(create_app(tmp_path / "web.sqlite3"))
    draft = client.post("/api/v1/drafts", json={"dmsCsv": DMS, "wtFasta": FASTA}).json()
    source = {"kind": "draft", "id": draft["id"]}

    histogram = client.post("/api/v1/score-histograms", json={"source": source})
    assert histogram.status_code == 200
    assert sum(histogram.json()["singles"]) == 4
    assert histogram.json()["wildType"] == 0.0

    scoped = client.post("/api/v1/substitution-maps", json={"source": source, "scope": "sequence"})
    assert scoped.status_code == 200
    assert scoped.json()["scope"] == "sequence"
    assert scoped.json()["assayedCount"] == 2


def test_mavedb_import_uses_the_score_set_target_sequence(monkeypatch):
    import web.server as server

    def fake_fetch(reference):
        assert reference == "urn:mavedb:00001222-b-2"
        return {
            "urn": "urn:mavedb:00001222-b-2",
            "title": "Example score set",
            "targetName": "BRCA1",
            "recordUrl": "https://www.mavedb.org/score-sets/urn:mavedb:00001222-b-2",
            "wtFasta": FASTA,
            "dmsCsv": MAVEDB_CSV,
        }

    monkeypatch.setattr(server, "fetch_score_set", fake_fetch)
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/mavedb-imports",
        json={"reference": "urn:mavedb:00001222-b-2"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["wtFasta"] == FASTA
    assert payload["inspection"]["detected"]["schema"] == "hgvs"


def test_mavedb_import_reports_a_bad_reference():
    client = TestClient(create_app())

    response = client.post("/api/v1/mavedb-imports", json={"reference": "not a urn"})

    assert response.status_code == 422
    assert "cannot find a MaveDB URN" in response.json()["detail"]
