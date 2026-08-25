import pandas as pd
import pytest

from mutascape.benchmarks.search import (
    TabularOracle,
    greedy_search,
    random_search,
    ridge_guided_search,
)


def _landscape():
    candidates = [{}, {1: "B"}, {2: "B"}, {1: "B", 2: "B"}]
    variants = pd.DataFrame({"mutations": candidates, "fitness": [0.0, 1.0, 0.5, 2.0]})
    return candidates, TabularOracle.from_dataframe(variants)


@pytest.mark.parametrize("search", [random_search, greedy_search, ridge_guided_search])
def test_search_methods_respect_budget(search):
    candidates, oracle = _landscape()

    result = search(oracle, candidates, budget=3, seed=2)

    assert result.queries == 3
    assert len(result.incumbent_trace) == 3
    assert all(
        earlier <= later
        for earlier, later in zip(result.incumbent_trace, result.incumbent_trace[1:])
    )


def test_search_is_deterministic_for_seed():
    candidates, oracle = _landscape()

    first = random_search(oracle, candidates, budget=3, seed=9)
    second = random_search(oracle, candidates, budget=3, seed=9)

    assert first == second


def test_tabular_oracle_rejects_unmeasured_variant():
    _, oracle = _landscape()

    with pytest.raises(ValueError, match="absent"):
        oracle.evaluate_batch([{3: "C"}])
