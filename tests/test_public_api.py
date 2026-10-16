"""Tests for the small public generator and oracle interface."""

import subprocess
import sys

import pandas as pd
import pytest

from mutascape import (
    LandscapeGenerator,
    Oracle,
    QuantileCalibratedLandscape,
    from_dms,
    legacy,
    load_oracle,
)

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
    assert manifest["generator"] == "dms-sigmoid-v1"
    assert manifest["seed"] == 17
    assert manifest["strength"] == 0.3
    assert manifest["mode"] == "sigmoid"
    assert manifest["margin"] == 0.05
    assert manifest["lower"] < -0.4 < 0.5 < manifest["upper"]
    assert manifest["contacts"] is None
    assert manifest["positions"] == [1, 2]
    assert manifest["substitutions"] == 4
    assert manifest["maximum_depth"] == 2
    assert manifest["observations"] == 5
    assert manifest["wild_type_fitness"] == 0.0
    assert len(str(manifest["source_digest"])) == 32
    assert "seed=17" in repr(oracle)
    assert oracle.provenance("A1G") == "measured"
    assert oracle.provenance("A1G+C2D") == "synthetic"


def test_draw_strength_overrides_pair_scale(generator):
    additive = generator.draw(seed=4, strength=0.0)
    coupled = generator.draw(seed=4, strength=1.0)

    assert additive["A1G"] == coupled["A1G"] == 0.5
    assert additive["A1G+C2D"] != coupled["A1G+C2D"]
    assert coupled.manifest()["strength"] == 1.0


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
            "variant": ["WT", "A1G", "A1V", "C2D"],
            "fitness": [1.0, 1.5, 0.2, 0.5],
        }
    ).to_csv(dms_path, index=False)
    fasta_path.write_text(">WT\nAC\n")

    oracle = from_dms(dms_path, fasta=fasta_path).draw(seed=5)

    assert oracle["A1G"] == pytest.approx(1.5)
    assert oracle.is_observed("A1G+C2D") is False


def test_combinations_require_the_calibration_argument():
    variants = pd.DataFrame(
        {"variant": ["WT", "A1G", "C2D", "A1G+C2D"], "fitness": [1.0, 1.5, 0.5, 2.1]}
    )

    with pytest.raises(ValueError, match="calibration="):
        from_dms(variants, wt_sequence="AC")


@pytest.fixture
def calibration_inputs():
    singles = pd.DataFrame(
        {
            "variant": ["WT", "A1G", "A1V", "C2D", "C2E", "D3F", "D3G"],
            "fitness": [1.0, 1.4, 0.3, 0.6, 1.1, 0.9, 0.2],
        }
    )
    calibration = pd.DataFrame(
        {
            "variant": ["A1G+C2D", "A1V+C2E", "C2D+D3F", "A1G+D3G", "A1V+C2D+D3F"],
            "fitness": [0.8, 0.1, 0.4, 0.3, 0.0],
        }
    )

    return singles, calibration


def test_calibration_matches_existing_model_and_keeps_measurements(calibration_inputs):
    singles, calibration = calibration_inputs
    original_singles, original_calibration = singles.copy(deep=True), calibration.copy(deep=True)
    generator = from_dms(singles, wt_sequence="ACD", calibration=calibration)
    reference = QuantileCalibratedLandscape.fit(
        "ACD",
        singles.rename(columns={"variant": "mutations"}),
        calibration.rename(columns={"variant": "mutations"}),
    )
    assert generator.to_dict() == reference.to_dict()
    oracle = generator.draw(seed=3)
    variants = [*singles.variant, *calibration.variant, "A1G+C2E", "A1V+D3G"]
    pd.testing.assert_series_equal(
        oracle.score_many(variants), reference.draw(seed=3).score_many(variants)
    )
    pd.testing.assert_series_equal(
        oracle.score_many(variants), generator.draw(seed=3).score_many(variants)
    )
    assert oracle.score_many(singles.variant.tolist()).tolist() == pytest.approx(singles.fitness)

    assert oracle["A1G"] == pytest.approx(1.4)
    assert oracle.is_observed("C2E") is True
    assert oracle.score_many(["A1G+C2E"]).index.tolist() == ["A1G+C2E"]
    pd.testing.assert_frame_equal(singles, original_singles)
    pd.testing.assert_frame_equal(calibration, original_calibration)


def test_calibration_loads_csv_and_fasta_paths(calibration_inputs, tmp_path):
    singles, calibration = calibration_inputs
    singles.to_csv(tmp_path / "singles.csv", index=False)
    calibration.to_csv(tmp_path / "combinations.csv", index=False)
    (tmp_path / "wt.fasta").write_text(">WT\nACD\n")
    generator = from_dms(
        tmp_path / "singles.csv",
        fasta=tmp_path / "wt.fasta",
        calibration=tmp_path / "combinations.csv",
    )
    reference = from_dms(singles, wt_sequence="ACD", calibration=calibration)
    assert generator.to_dict() == reference.to_dict()


def test_calibration_accepts_mutation_mappings(calibration_inputs):
    singles, calibration = calibration_inputs
    mapped = calibration.drop(columns="variant").assign(
        mutations=[
            {1: "G", 2: "D"},
            {1: "V", 2: "E"},
            {2: "D", 3: "F"},
            {1: "G", 3: "G"},
            {1: "V", 2: "D", 3: "F"},
        ]
    )
    generator = from_dms(singles, wt_sequence="ACD", calibration=mapped)
    reference = from_dms(singles, wt_sequence="ACD", calibration=calibration)
    assert generator.to_dict() == reference.to_dict()


def test_the_removed_pilot_alias_is_rejected(calibration_inputs):
    singles, calibration = calibration_inputs
    with pytest.raises(TypeError):
        from_dms(singles, wt_sequence="ACD", pilot=calibration)


@pytest.mark.parametrize("options", [{"complete": "esm2"}, {"maximum_depth": 2}])
def test_calibration_rejects_unsupported_options(calibration_inputs, options):
    singles, calibration = calibration_inputs
    with pytest.raises(ValueError, match="does not support complete or maximum_depth"):
        from_dms(singles, wt_sequence="ACD", calibration=calibration, **options)


def test_margin_widens_the_generated_range(calibration_inputs):
    singles, calibration = calibration_inputs
    narrow = from_dms(singles, wt_sequence="ACD", calibration=calibration, margin=0.05)
    wide = from_dms(singles, wt_sequence="ACD", calibration=calibration, margin=0.40)
    assert wide.reference.readout.ceiling > narrow.reference.readout.ceiling
    assert wide.reference.readout.floor < narrow.reference.readout.floor


@pytest.mark.parametrize(
    ("variants", "fitness", "error"),
    [
        ([], [], "must contain measurements"),
        (["WT"], [1.0], "only multi-mutant genotypes"),
        (["A1G"], [1.4], "only multi-mutant genotypes"),
        (["A1G+C2D"] * 2, [0.3, 0.3], "duplicate measurement"),
        (["A1G+C2D"], [float("nan")], "must be finite"),
        (["A1G+C2D"], [0.3], "must identify both burden terms"),
    ],
)
def test_invalid_calibration_is_not_ignored(calibration_inputs, variants, fitness, error):
    singles, _ = calibration_inputs
    calibration = pd.DataFrame({"variant": variants, "fitness": fitness})
    with pytest.raises(ValueError, match=error):
        from_dms(singles, wt_sequence="ACD", calibration=calibration)


def test_omitting_calibration_keeps_single_site_generator():
    original = from_dms(SINGLES, wt_sequence="AC")
    explicit = from_dms(SINGLES, wt_sequence="AC", calibration=None)
    assert isinstance(explicit, LandscapeGenerator)
    assert explicit.draw(seed=42).manifest() == original.draw(seed=42).manifest()
    assert explicit.draw(seed=42)["A1G+C2D"] == original.draw(seed=42)["A1G+C2D"]


def test_legacy_manifest_still_loads():
    legacy_oracle = legacy.from_dms(SINGLES, wt_sequence="AC").draw(seed=8)
    restored = load_oracle(legacy_oracle.manifest(), dms=SINGLES, wt_sequence="AC")

    assert restored.manifest()["generator"] == "dms-pairwise-v1"
    assert restored["A1G+C2D"] == pytest.approx(legacy_oracle["A1G+C2D"])


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
    variants = pd.DataFrame({"variant": ["WT", "A1G", "A1G+C2D"], "fitness": [1.0, 1.5, 2.1]})

    with pytest.raises(ValueError, match="constituent singles for C2D"):
        from_dms(variants, wt_sequence="AC")


def test_saved_manifest_needs_matching_singles():
    oracle = from_dms(SINGLES, wt_sequence="AC").draw(seed=1)
    missing = SINGLES.iloc[:3]

    with pytest.raises(ValueError, match="do not match"):
        load_oracle(oracle.manifest(), dms=missing, wt_sequence="AC")


def test_core_import_does_not_load_fastapi():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mutascape; assert 'fastapi' not in sys.modules",
        ],
        check=True,
    )
