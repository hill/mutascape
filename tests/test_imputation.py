"""Regression checks for optional single completion and provenance."""

import numpy as np
import pandas as pd
import pytest

from mutascape import from_dms, load_oracle
from mutascape.epistasis import AMINO_ACIDS
from mutascape.imputation import fit_single_calibration


@pytest.fixture
def fake_esm(monkeypatch):
    def score(self, sequence, positions=None):
        return {
            p: {aa: float(i - 10) for i, aa in enumerate(AMINO_ACIDS)}
            for p in range(1, len(sequence) + 1)
        }

    monkeypatch.setattr("mutascape.imputation.ESM2Scorer.score", score)


def singles():
    return pd.DataFrame({"variant": ["WT", "A1C", "A1D", "A1E"], "fitness": [2.0, 2.2, 1.5, 1.0]})


def test_completion_keeps_anchors_readout_and_existing_combinations(fake_esm):
    measured = from_dms(singles(), wt_sequence="AC")
    completed = from_dms(singles(), wt_sequence="AC", complete="esm2")
    assert completed.nonlinearity == measured.nonlinearity
    assert completed.preimage_scale == measured.preimage_scale
    assert completed.observations == measured.observations
    assert len(completed.single_effects) == 38
    oracle = completed.draw(42)
    assert oracle["A1C"] == 2.2
    assert oracle.provenance("A1C") == "measured"
    assert oracle.provenance("C2A") == "imputed"
    assert not oracle.is_observed("C2A")
    assert oracle.provenance("A1C+C2A") == "synthetic"
    assert np.isfinite(oracle["A1C+C2A"])
    assert oracle["C2A"] == completed.draw(99)["C2A"]
    with pytest.raises(ValueError, match="not mutable"):
        measured.draw(42)["C2A"]


def test_frozen_completion_roundtrip_does_not_load_esm(fake_esm, monkeypatch):
    oracle = from_dms(singles(), wt_sequence="AC", complete="esm2").draw(42)

    def fail(*args, **kwargs):
        raise AssertionError("weights must not load during restore")

    monkeypatch.setattr("mutascape.imputation.ESM2Scorer.score", fail)
    restored = load_oracle(oracle.manifest(), dms=singles(), wt_sequence="AC")
    assert restored.manifest() == oracle.manifest()
    assert restored["A1C+C2A"] == oracle["A1C+C2A"]
    damaged = oracle.manifest()
    damaged["completion"] = {**damaged["completion"], "effects": [[1, "C", 0.1]]}
    with pytest.raises(ValueError, match="overlapping"):
        load_oracle(damaged, dms=singles(), wt_sequence="AC")


def test_calibration_rejects_uninformative_data():
    with pytest.raises(ValueError, match="variation"):
        fit_single_calibration(np.ones(4), np.arange(4))


def test_web_completion_freezes_values_and_marks_grid(fake_esm):
    from web.models import parse_custom_source, substitution_map
    from web.posterior import complete_source, posterior_from_source, query_posterior

    source, summary = parse_custom_source(dms_csv=singles().to_csv(index=False), wt_fasta=">wt\nAC")
    complete_source(source, summary)
    posterior, _ = posterior_from_source(source)
    grid = substitution_map(source, scope="sequence", impute_missing=False)
    assert grid["measuredCount"] == 3
    assert grid["imputedCount"] == 35
    result = query_posterior(
        posterior,
        [{"mutationLabel": "C2A"}],
        positions=[1, 2],
        wild_type={1: "A", 2: "C"},
        seed=42,
        ensemble_size=3,
    )[0]
    assert result["provenance"] == "imputed"
    assert result["posteriorStd"] == pytest.approx(0)
