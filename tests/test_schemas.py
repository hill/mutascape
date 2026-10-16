"""A saved oracle is the reproducibility claim, so a corrupt payload must fail loudly."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mutascape.calibrated import QuantileCalibratedLandscape
from mutascape.completion import LandscapePrior
from mutascape.schemas import OracleManifest, PilotModel, QuantileModel

FIXTURE = Path(__file__).parent / "data" / "quantile_model.json"


@pytest.fixture
def payload() -> dict:
    return json.loads(FIXTURE.read_text())


def corrupt(payload: dict, mutate) -> dict:
    copy = json.loads(json.dumps(payload))
    mutate(copy)
    return copy


def test_round_trip_preserves_every_value(payload):
    restored = QuantileCalibratedLandscape.from_dict(payload)
    assert restored.to_dict() == payload


def test_round_trip_is_stable_across_two_passes(payload):
    once = QuantileCalibratedLandscape.from_dict(payload).to_dict()
    twice = QuantileCalibratedLandscape.from_dict(once).to_dict()
    assert twice == once


def test_version_one_payloads_restore_with_the_linear_tail(payload):
    legacy = corrupt(
        payload, lambda d: (d.__setitem__("model", "quantile-pilot-v1"), d.pop("upper_tail", None))
    )
    assert QuantileModel.model_validate(legacy).resolved_upper_tail == "linear"
    assert QuantileCalibratedLandscape.from_dict(legacy).upper_tail == "linear"


@pytest.mark.parametrize(
    ("reason", "mutate"),
    [
        ("unknown version", lambda d: d.__setitem__("model", "quantile-pilot-v9")),
        ("too few calibration values", lambda d: d.__setitem__("pilot_fitness", [1.0, 2.0])),
        ("non-finite fitness", lambda d: d["pilot_fitness"].__setitem__(0, float("nan"))),
        ("negative pair scale", lambda d: d["reference"].__setitem__("pair_scale", -1.0)),
        (
            "wrong coefficient count",
            lambda d: d["reference"].__setitem__("coefficients", [1.0, 2.0, 3.0]),
        ),
        (
            "no wild-type anchor",
            lambda d: d["reference"].__setitem__(
                "anchors", [a for a in d["reference"]["anchors"] if a["mutations"]]
            ),
        ),
        ("inverted readout", lambda d: d["reference"]["readout"].__setitem__("ceiling", -99.0)),
        (
            "version one claiming the latent tail",
            lambda d: (
                d.__setitem__("model", "quantile-pilot-v1"),
                d.__setitem__("upper_tail", "latent"),
            ),
        ),
    ],
)
def test_corrupt_payloads_are_rejected(payload, reason, mutate):
    with pytest.raises(ValidationError):
        QuantileCalibratedLandscape.from_dict(corrupt(payload, mutate))


def test_files_written_before_the_tail_was_configurable_get_the_old_default(payload):
    older = corrupt(payload, lambda d: d.pop("tail_fraction", None))
    assert QuantileCalibratedLandscape.from_dict(older).tail_fraction == 0.1


def test_tail_fraction_outside_the_unit_interval_is_rejected(payload):
    with pytest.raises(ValidationError):
        QuantileCalibratedLandscape.from_dict(
            corrupt(payload, lambda d: d.__setitem__("tail_fraction", 0.0))
        )


def test_unknown_keys_survive_so_newer_files_still_load(payload):
    forward = corrupt(payload, lambda d: d.__setitem__("a_field_from_the_future", 1))
    assert QuantileCalibratedLandscape.from_dict(forward).upper_tail == payload["upper_tail"]


def test_pilot_model_rejects_a_foreign_version(payload):
    with pytest.raises(ValidationError):
        PilotModel.model_validate(
            corrupt(payload["reference"], lambda d: d.__setitem__("model", "other"))
        )


def test_oracle_manifest_requires_its_identifying_fields():
    with pytest.raises(ValidationError):
        OracleManifest.model_validate({"generator": "dms-sigmoid-v1"})


def test_oracle_manifest_rejects_a_foreign_generator():
    with pytest.raises(ValidationError):
        OracleManifest.model_validate({"generator": "dms-pairwise-v1"})


def test_bundled_prior_round_trips():
    root = Path(__file__).resolve().parents[1] / "src/mutascape/data/priors"
    for path in sorted(root.glob("*.json")):
        body = json.loads(path.read_text())["parameters"]
        assert LandscapePrior.from_dict(body).to_dict() == body
