"""HTTP contract tests for the benchmark oracle service."""

import sqlite3

from fastapi.testclient import TestClient

from mutascape.server import create_app

DMS = """position,wildtype,mutant,fitness
1,A,G,0.5
1,A,V,-0.2
2,C,A,0.3
2,C,D,-0.4
"""
FASTA = ">WT\nAC\n"
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

    for path in ("/", "/landscapes", "/leaderboards", "/landscapes/gb1_headline"):
        response = client.get(path)
        assert response.status_code == 200
        assert '<div id="root"></div>' in response.text


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
