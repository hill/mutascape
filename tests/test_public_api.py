"""Tests for the small public generator and oracle interface."""

import pandas as pd
import pytest

from mutascape import LandscapeGenerator, Oracle, from_dms, load_oracle

SINGLES = pd.DataFrame(
    {
        "position": [1, 1, 2, 2],
        "wildtype": ["A", "A", "C", "C"],
        "mutant": ["G", "V", "A", "D"],
        "fitness": [0.5, -0.2, 0.3, -0.4],
    }
)


@pytest.fixture
def generator() -> LandscapeGenerator:
    return from_dms(SINGLES, wt_sequence="AC", maximum_depth=2)


def test_draw_accepts_labels_mappings_and_mutation_separators(generator):
    oracle = generator.draw(seed=42)

    expected = oracle.evaluate({1: "G", 2: "D"})
    assert oracle["A1G+C2D"] == pytest.approx(expected)
    assert oracle.score("a1g:c2d") == pytest.approx(expected)
    assert oracle.score({"1": "g", "2": "d"}) == pytest.approx(expected)


def test_score_many_returns_canonical_labelled_series(generator):
    scores = generator.draw(seed=42).score_many(["WT", "a1g", "a1g:c2d"])

    assert isinstance(scores, pd.Series)
    assert scores.name == "fitness"
    assert scores.index.tolist() == ["wt", "A1G", "A1G+C2D"]
    assert scores.loc["A1G"] == pytest.approx(0.5)


def test_oracle_reports_observation_provenance_and_manifest(generator):
    oracle = generator.draw(seed=17)

    assert isinstance(generator, LandscapeGenerator)
    assert isinstance(oracle, Oracle)
    assert oracle.is_observed("A1G") is True
    assert oracle.is_observed("A1G+C2D") is False
    manifest = oracle.manifest()
    assert manifest["generator"] == "dms-pairwise-v1"
    assert manifest["prior"] == "combinatorial-v2"
    assert manifest["seed"] == 17
    assert manifest["coupling"] == 0.1
    assert manifest["positions"] == [1, 2]
    assert manifest["substitutions"] == 4
    assert manifest["maximum_depth"] == 2
    assert manifest["observations"] == 5
    assert manifest["wild_type_fitness"] == 0.0
    assert len(str(manifest["source_digest"])) == 32
    assert "seed=17" in repr(oracle)


def test_domain_describes_counts_and_samples_distinct_variants(generator):
    domain = generator.domain

    assert domain.positions == (1, 2)
    assert domain.substitutions_at(1) == ("G", "V")
    assert domain.count(depth=0) == 1
    assert domain.count(depth=1) == 4
    assert domain.count(depth=2) == 4
    assert domain.count() == 9
    sampled = domain.sample(depth=2, n=4, seed=3)
    assert len({mutant.label for mutant in sampled}) == 4
    assert all(domain.contains(mutant) for mutant in sampled)


def test_maximum_depth_rejects_deeper_queries():
    generator = from_dms(SINGLES, wt_sequence="AC", maximum_depth=1)

    with pytest.raises(ValueError, match="maximum depth of 1"):
        generator.draw(seed=2).score("A1G+C2D")


def test_variant_csv_and_fasta_paths_are_loaded(tmp_path):
    dms_path = tmp_path / "variants.csv"
    fasta_path = tmp_path / "wt.fasta"
    pd.DataFrame(
        {
            "variant": ["WT", "A1G", "C2D", "A1G:C2D"],
            "fitness": [1.0, 1.5, 0.5, 2.1],
        }
    ).to_csv(dms_path, index=False)
    fasta_path.write_text(">WT\nAC\n")

    oracle = from_dms(dms_path, fasta=fasta_path).draw(seed=5)

    assert oracle["A1G:C2D"] == pytest.approx(2.1)
    assert oracle.is_observed("A1G+C2D") is True


def test_saved_manifest_recreates_and_checks_an_oracle(tmp_path):
    generator = from_dms(
        SINGLES,
        wt_sequence="AC",
        wild_type_fitness=1.0,
        maximum_depth=2,
    )
    oracle = generator.draw(seed=23)
    path = tmp_path / "oracle.json"
    oracle.save_manifest(path)

    recreated = load_oracle(path, dms=SINGLES, wt_sequence="AC")

    assert recreated["WT"] == pytest.approx(1.0)
    assert recreated["A1G+C2D"] == pytest.approx(oracle["A1G+C2D"])
    changed = SINGLES.copy()
    changed.loc[0, "fitness"] = 0.6
    with pytest.raises(ValueError, match="do not match"):
        load_oracle(path, dms=changed, wt_sequence="AC")


def test_loader_requires_one_sequence_source():
    with pytest.raises(ValueError, match="exactly one"):
        from_dms(SINGLES)
    with pytest.raises(ValueError, match="exactly one"):
        from_dms(SINGLES, fasta="wt.fasta", wt_sequence="AC")


def test_variant_input_requires_constituent_singles():
    variants = pd.DataFrame(
        {
            "variant": ["WT", "A1G", "A1G+C2D"],
            "fitness": [1.0, 1.5, 2.1],
        }
    )

    with pytest.raises(ValueError, match="constituent singles for C2D"):
        from_dms(variants, wt_sequence="AC")
