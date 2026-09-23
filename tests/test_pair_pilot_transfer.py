"""Check that the transfer experiment preserves the original pair model."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from scripts.compare_c2st_alternatives import AlternativeFits, basis
from scripts.test_pair_pilot_transfer import PairFit
from tests.test_c2st_alternatives import calibration_data


def test_transfer_model_replays_original_alternative():
    singles, pilot = calibration_data()
    old = AlternativeFits.fit("AAAA", singles, pilot)
    new = PairFit.fit("AAAA", singles, pilot)
    for model in ("pilot_reference", "learned_pairs"):
        for seed in (42, 73):
            variants = pilot.mutations.tolist()
            assert_allclose(new.predict(variants, seed, model), old.predict(variants, seed, model))
            assert_array_equal(
                new.predict(singles.mutations.tolist(), seed, model), singles.fitness
            )
            assert_allclose(
                new.predict(variants[::-1], seed, model), new.predict(variants, seed, model)[::-1]
            )


def test_coefficient_covariance_keeps_unmeasured_directions():
    singles, pilot = calibration_data()
    new = PairFit.fit("AAAA", singles, pilot.iloc[:4])
    x = basis(pilot.mutations.iloc[:4], new.keys)[1].astype(float)
    precision = new.coefficient_precision * np.eye(x.shape[1]) + new.noise_precision * x.T @ x
    assert_allclose(new.root @ new.root.T, np.linalg.inv(precision), atol=1e-9)


def test_vectorised_peak_count_matches_strict_neighbour_rule():
    from mutascape.validation.metrics import local_optima_from_tensor
    from scripts.diagnose_pair_pilot_transfer import strict_peaks

    for tensor in (
        np.array([[1.0, 1.0], [0.0, 2.0]]),
        np.zeros((3, 3)),
        np.random.default_rng(42).normal(size=(3, 4, 2)),
    ):
        assert strict_peaks(tensor) == local_optima_from_tensor(tensor)["n_optima"]
