"""Check the follow-up's independent samples and complete campaign design."""

import numpy as np

from scripts import calibration_confidence as followup
from scripts import manuscript_v2 as original


def test_followup_streams_are_distinct():
    keys = [
        (role, protein, sample, landscape)
        for role in (2, 5, 7, 8)
        for protein in original.EVALUATION
        for sample in range(followup.SAMPLES)
        for landscape in range(original.LANDSCAPES)
    ]
    seeds = [followup.fresh_seed(*key) for key in keys]
    assert len(set(seeds)) == len(seeds)
    assert all(fresh != original.seed_for(*key) for key, fresh in zip(keys, seeds))


def test_landscape_loader_keeps_sample_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "OUTPUT", tmp_path)
    folder = tmp_path / "GB1" / "landscapes"
    folder.mkdir(parents=True)
    np.save(folder / "sigmoid_1_0.npy", [1.0])
    np.save(folder / "sigmoid_2_0.npy", [2.0])
    np.testing.assert_array_equal(followup.landscape_values("GB1", "sigmoid", 1, 0), [1.0])
    np.testing.assert_array_equal(followup.landscape_values("GB1", "sigmoid", 2, 0), [2.0])
