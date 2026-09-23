"""Checks for the three pilot alternatives compared against C2ST."""

from itertools import combinations

import numpy as np
import pandas as pd

from scripts.compare_c2st_alternatives import AlternativeFits


def calibration_data():
    effects = {1: 0.1, 2: 0.2, 3: -0.1, 4: 0.4}
    singles = pd.DataFrame(
        {
            "mutations": [{}, *[{p: "C"} for p in effects]],
            "fitness": [2.0, *[2 + value for value in effects.values()]],
        }
    )
    mutations = [
        {p: "C" for p in subset} for size in (2, 3, 4) for subset in combinations(range(1, 5), size)
    ]
    pilot = pd.DataFrame(
        {
            "mutations": mutations,
            "fitness": [
                2 + sum(effects[p] for p in m) - 0.2 * (len(m) - 1) + 0.1 * (1 in m and 2 in m)
                for m in mutations
            ],
        }
    )
    return singles, pilot


def test_alternatives_preserve_measurements_and_query_order():
    singles, pilot = calibration_data()
    fitted = AlternativeFits.fit("AAAA", singles, pilot)
    for model in ("pilot_reference", "loo_noise", "raw_quadratic", "learned_pairs"):
        np.testing.assert_array_equal(
            fitted.predict(singles.mutations.tolist(), 42, model), singles.fitness
        )
        variants = pilot.mutations.tolist()
        values = fitted.predict(variants, 42, model)
        np.testing.assert_allclose(fitted.predict(variants[::-1], 42, model), values[::-1])
        np.testing.assert_allclose(fitted.predict(variants[:2], 42, model), values[:2])
        np.testing.assert_array_equal(fitted.predict(variants, 42, model), values)
        assert np.isfinite(values).all()


def test_reference_implementation_matches_existing_generator():
    singles, pilot = calibration_data()
    fitted = AlternativeFits.fit("AAAA", singles, pilot)
    variants = pilot.mutations.tolist()
    np.testing.assert_allclose(
        fitted.predict(variants, 42, "pilot_reference"),
        fitted.reference.draw(42).evaluate_batch(variants),
        atol=1e-12,
    )
