"""Tests for posterior landscape queries and browser previews."""

import pytest

from mutascape.priors import DEFAULT_PRIOR_ID, list_prior_metadata, load_prior_bundle
from web.models import parse_custom_source
from web.posterior import (
    posterior_from_source,
    posterior_shape_summary,
    preview_posterior,
    query_posterior,
)

FASTA = ">WT\nAC\n"
VARIANTS = """variant,fitness
WT,1.0
A1G,1.5
A1V,1.2
C2D,0.5
A1G+C2D,2.1
"""


@pytest.fixture
def source():
    parsed, _ = parse_custom_source(VARIANTS, FASTA)
    return parsed


def test_bundled_prior_has_training_provenance():
    priors = {prior["id"]: prior for prior in list_prior_metadata()}
    prior = priors[DEFAULT_PRIOR_ID]

    # CR6261_H9_binding reports every single mutant at the assay floor, so it
    # cannot anchor a normalised coefficient and never entered the prior.
    assert prior["nLandscapes"] == 29
    assert prior["nProteins"] == 14
    assert prior["excludedLandscapes"] == ["CR6261_H9_binding"]
    assert {source["name"] for source in prior["sources"]} == {"SSMuLA", "CombinGym"}


def test_default_prior_shrinks_shallow_depth_shifts():
    _, prior = load_prior_bundle()

    # Proteins disagree about the depth-two residual far more than they agree on
    # its size, so almost none of the pooled shift survives transfer.
    assert prior.depth_shrinkage[2] < 0.05
    assert abs(prior.depth_location[2]) < 0.05


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
    # The upload puts WT at 1.0 and the double at 2.1, and every observation reaches the
    # posterior relative to wild type.
    assert first[0]["fixedOracleFitness"] == pytest.approx(1.1)
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
    assert preview["settings"]["generatorKind"] == "dms-pairwise-v1"
    assert preview["benchmarkValidity"]["nLandscapes"] == 16
    assert preview["benchmarkValidity"]["external"]["nLandscapes"] == 7
    assert preview["benchmarkValidity"]["budgetCurve"]["budgets"] == [48, 96, 192, 384]
    assert len(preview["benchmarkValidity"]["budgetCurve"]["series"]) == 3
    assert "single mutants exact" in preview["notices"][1]


def test_shape_summary_separates_posterior_from_an_additive_surrogate(source):
    posterior, _ = posterior_from_source(source)
    shape = posterior_shape_summary(posterior, seed=7)

    assert shape is not None
    # An additive surrogate carries no variance above interaction order one,
    # which is the whole reason a smooth benchmark ranks search methods wrongly.
    assert shape["additive"]["walshNonadditive"] < 1e-9
    assert shape["posterior"]["walshNonadditive"] > 0.01


def test_shape_summary_declines_an_unenumerable_domain(monkeypatch):
    import web.posterior as module

    monkeypatch.setattr(module, "MAX_SHAPE_CUBE", 2)
    parsed, _ = parse_custom_source(VARIANTS, FASTA)
    posterior, _ = posterior_from_source(parsed)
    assert posterior_shape_summary(posterior, seed=7) is None
