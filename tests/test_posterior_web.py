"""Tests for posterior landscape queries and browser previews."""

import pytest

from mutascape.posterior_web import (
    list_prior_metadata,
    posterior_from_source,
    preview_posterior,
    query_posterior,
)
from mutascape.web_models import parse_custom_source

FASTA = ">WT\nAC\n"
VARIANTS = """variant,fitness
WT,1.0
A1G,1.5
A1V,0.8
C2D,0.5
A1G+C2D,2.1
"""


@pytest.fixture
def source():
    parsed, _ = parse_custom_source(VARIANTS, FASTA)
    return parsed


def test_bundled_prior_has_training_provenance():
    prior = list_prior_metadata()[0]

    assert prior["id"] == "combinatorial-v1"
    assert prior["nLandscapes"] == 30
    assert prior["nProteins"] == 14
    assert {source["name"] for source in prior["sources"]} == {"SSMuLA", "CombinGym"}


def test_query_preserves_observations_and_samples_unseen_combinations(source):
    posterior, _ = posterior_from_source(source)
    kwargs = {
        "positions": [1, 2],
        "wild_type": {1: "A", 2: "C"},
        "seed": 17,
        "ensemble_size": 20,
    }
    entries = [
        {"mutationLabel": "A1G+C2D", "compactSequence": None},
        {"mutationLabel": "A1V+C2D", "compactSequence": None},
    ]

    first = query_posterior(posterior, entries, **kwargs)
    second = query_posterior(posterior, entries, **kwargs)
    changed = query_posterior(posterior, entries, **{**kwargs, "seed": 18})

    assert first == second
    assert first[0]["fixedOracleFitness"] == pytest.approx(2.1)
    assert first[0]["posteriorStd"] == pytest.approx(0.0)
    assert first[1]["posteriorStd"] > 0
    assert first[1]["fixedOracleFitness"] != changed[1]["fixedOracleFitness"]
    assert set(first[1]["intervals"]) == {"50", "80", "95"}
    assert len(first[1]["samples"]) == 20


def test_query_rejects_a_substitution_without_a_measured_single(source):
    posterior, _ = posterior_from_source(source)

    with pytest.raises(ValueError, match="missing constituent single"):
        query_posterior(
            posterior,
            [{"mutationLabel": "A1L", "compactSequence": None}],
            positions=[1, 2],
            wild_type={1: "A", 2: "C"},
            seed=1,
            ensemble_size=10,
        )


def test_posterior_preview_reports_uncertainty_in_rows_profiles_and_graph(source):
    posterior, prior = posterior_from_source(source)
    preview = preview_posterior(
        posterior,
        source_summary={
            "id": "draft_test",
            "name": "Test",
            "kind": "posterior",
            "positions": [1, 2],
            "wtCompact": "AC",
        },
        prior_metadata=prior,
        sample_size=5,
        depth_weights={1: 1, 2: 1},
        seed=9,
        ensemble_size=12,
    )

    assert preview["sample"]["requested"] == 5
    assert {point["depth"] for point in preview["profile"]} == {1, 2}
    assert all("posteriorStd" in row for row in preview["sample"]["rows"])
    assert any(node["posteriorStd"] > 0 for node in preview["sample"]["graph"]["nodes"])
    assert "does not measure realism" in preview["notices"][1]
