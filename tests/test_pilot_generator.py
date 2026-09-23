from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mutascape.experimental import PilotCalibratedLandscape


def inputs():
    singles = pd.DataFrame(
        {
            "mutations": [{}, {1: "C"}, {2: "C"}, {3: "C"}, {4: "C"}],
            "fitness": [2.0, 2.1, 2.2, 1.9, 2.4],
        }
    )
    pilot = pd.DataFrame(
        {
            "mutations": [
                {i: "C" for i in subset}
                for size in (2, 3, 4)
                for subset in combinations(range(1, 5), size)
            ]
        }
    )
    effects = {1: 0.1, 2: 0.2, 3: -0.1, 4: 0.4}
    pilot["fitness"] = [
        2 + sum(effects[i] for i in m) - 0.5 * (len(m) - 1) for m in pilot.mutations
    ]
    return singles, pilot


def test_pilot_preserves_anchors_and_fixed_draw_across_batches():
    singles, pilot = inputs()
    model = PilotCalibratedLandscape.fit("AAAA", singles, pilot)
    draw = model.draw(7)
    np.testing.assert_array_equal(draw.evaluate_batch(singles.mutations.tolist()), singles.fitness)
    variants = pilot.mutations.tolist()
    np.testing.assert_allclose(draw.evaluate_batch(variants), [draw.evaluate(v) for v in variants])
    np.testing.assert_array_equal(
        draw.evaluate_batch(variants), model.draw(7).evaluate_batch(variants)
    )
    assert not np.allclose(draw.evaluate_batch(variants), model.draw(8).evaluate_batch(variants))
    with pytest.raises(ValueError):
        draw.evaluate({1: "D"})


def test_pilot_input_contract_rejects_hidden_or_invalid_observations():
    singles, pilot = inputs()
    with pytest.raises(ValueError, match="single-mutant input contains"):
        PilotCalibratedLandscape.fit("AAAA", pd.concat([singles, pilot.iloc[:1]]), pilot)
    with pytest.raises(ValueError, match="multi-mutants"):
        PilotCalibratedLandscape.fit("AAAA", singles, pd.concat([pilot, singles.iloc[:1]]))
    with pytest.raises(ValueError, match="duplicate pilot"):
        PilotCalibratedLandscape.fit("AAAA", singles, pd.concat([pilot, pilot.iloc[:1]]))
    pilot.loc[0, "fitness"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PilotCalibratedLandscape.fit("AAAA", singles, pilot)


def test_pilot_fitting_is_invariant_to_row_order_and_fitness_offset():
    singles, pilot = inputs()
    first = PilotCalibratedLandscape.fit("AAAA", singles, pilot).draw(8)
    singles["fitness"] += 10
    pilot["fitness"] += 10
    second = PilotCalibratedLandscape.fit("AAAA", singles.iloc[::-1], pilot.iloc[::-1]).draw(8)
    np.testing.assert_allclose(
        second.evaluate_batch(pilot.mutations.tolist()),
        first.evaluate_batch(pilot.mutations.tolist()) + 10,
    )


def test_pilot_serialisation_preserves_predictions():
    import json

    singles, pilot = inputs()
    model = PilotCalibratedLandscape.fit("AAAA", singles, pilot)
    restored = PilotCalibratedLandscape.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(
        model.draw(9).evaluate_batch(pilot.mutations.tolist()),
        restored.draw(9).evaluate_batch(pilot.mutations.tolist()),
    )
    assert restored.calibration_keys == model.calibration_keys
