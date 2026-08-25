"""Budget-matched search methods for real and synthetic landscape oracles."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from mutascape.benchmarks.baselines import RidgeLandscapeBaseline


class FitnessOracle(Protocol):
    """Minimal batch interface shared by Mutascape and tabular oracles."""

    def evaluate_batch(self, variants: list[dict[int, str]]) -> np.ndarray: ...


@dataclass(frozen=True)
class SearchResult:
    """Best variant and incumbent trace from one budgeted search run."""

    method: str
    best_fitness: float
    best_variant: dict[int, str]
    queries: int
    incumbent_trace: tuple[float, ...]


@dataclass(frozen=True)
class TabularOracle:
    """Fitness oracle backed by measured mutation dictionaries."""

    fitness: dict[tuple[tuple[int, str], ...], float]

    @classmethod
    def from_dataframe(cls, variants: pd.DataFrame) -> TabularOracle:
        values = {
            tuple(sorted(mutations.items())): float(fitness)
            for mutations, fitness in variants[["mutations", "fitness"]].itertuples(
                index=False, name=None
            )
        }
        return cls(values)

    def evaluate_batch(self, variants: list[dict[int, str]]) -> np.ndarray:
        try:
            return np.asarray(
                [self.fitness[tuple(sorted(variant.items()))] for variant in variants],
                dtype=float,
            )
        except KeyError as error:
            raise ValueError(
                f"variant is absent from the tabular oracle: {error.args[0]}"
            ) from error


def _candidate_keys(candidates: list[dict[int, str]]) -> list[tuple[tuple[int, str], ...]]:
    keys = [tuple(sorted(value.items())) for value in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("candidate variants must be unique")
    return keys


def _result(
    method: str,
    candidates: list[dict[int, str]],
    indices: list[int],
    fitness: np.ndarray,
) -> SearchResult:
    incumbent = np.maximum.accumulate(fitness)
    best_offset = int(np.argmax(fitness))
    return SearchResult(
        method=method,
        best_fitness=float(fitness[best_offset]),
        best_variant=dict(candidates[indices[best_offset]]),
        queries=len(indices),
        incumbent_trace=tuple(float(value) for value in incumbent),
    )


def random_search(
    oracle: FitnessOracle,
    candidates: list[dict[int, str]],
    *,
    budget: int,
    seed: int,
) -> SearchResult:
    """Query a uniform sample without replacement."""
    _candidate_keys(candidates)
    if budget <= 0:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=min(budget, len(candidates)), replace=False).tolist()
    fitness = oracle.evaluate_batch([candidates[index] for index in indices])
    return _result("random", candidates, indices, fitness)


def _hamming_distance(
    first: tuple[tuple[int, str], ...],
    second: tuple[tuple[int, str], ...],
) -> int:
    first_states = dict(first)
    second_states = dict(second)
    positions = set(first_states) | set(second_states)
    return sum(first_states.get(position) != second_states.get(position) for position in positions)


def greedy_search(
    oracle: FitnessOracle,
    candidates: list[dict[int, str]],
    *,
    budget: int,
    seed: int,
) -> SearchResult:
    """Take the best unqueried Hamming-one step until the budget is spent."""
    keys = _candidate_keys(candidates)
    if budget <= 0:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(seed)
    wild_type_index = next((index for index, key in enumerate(keys) if not key), None)
    current = int(rng.integers(len(candidates))) if wild_type_index is None else wild_type_index
    queried: list[int] = []
    values: list[float] = []
    seen: set[int] = set()

    while len(queried) < min(budget, len(candidates)):
        if current not in seen:
            value = float(oracle.evaluate_batch([candidates[current]])[0])
            queried.append(current)
            values.append(value)
            seen.add(current)
        neighbours = [
            index
            for index, key in enumerate(keys)
            if index not in seen and _hamming_distance(keys[current], key) == 1
        ]
        if not neighbours:
            remaining = [index for index in range(len(candidates)) if index not in seen]
            if not remaining:
                break
            current = int(rng.choice(remaining))
            continue
        remaining_budget = budget - len(queried)
        if len(neighbours) > remaining_budget:
            neighbours = rng.choice(neighbours, size=remaining_budget, replace=False).tolist()
        neighbour_values = oracle.evaluate_batch([candidates[index] for index in neighbours])
        queried.extend(neighbours)
        values.extend(float(value) for value in neighbour_values)
        seen.update(neighbours)
        current = neighbours[int(np.argmax(neighbour_values))]

    return _result("greedy", candidates, queried, np.asarray(values, dtype=float))


def ridge_guided_search(
    oracle: FitnessOracle,
    candidates: list[dict[int, str]],
    *,
    budget: int,
    seed: int,
    initial_size: int = 24,
    batch_size: int = 8,
) -> SearchResult:
    """Refit a pairwise ridge model and query its best unmeasured batch."""
    _candidate_keys(candidates)
    if budget <= 0:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(seed)
    target_budget = min(budget, len(candidates))
    initial = min(initial_size, target_budget)
    queried = rng.choice(len(candidates), size=initial, replace=False).tolist()
    values = oracle.evaluate_batch([candidates[index] for index in queried]).tolist()

    while len(queried) < target_budget:
        train = pd.DataFrame(
            {
                "mutations": [candidates[index] for index in queried],
                "fitness": values,
            }
        )
        model = RidgeLandscapeBaseline(max_order=2).fit(train)
        unseen = [index for index in range(len(candidates)) if index not in set(queried)]
        scores = model.predict([candidates[index] for index in unseen])
        take = min(batch_size, target_budget - len(queried))
        selected_offsets = np.argpartition(scores, -take)[-take:]
        selected = [unseen[int(offset)] for offset in selected_offsets]
        selected_values = oracle.evaluate_batch([candidates[index] for index in selected])
        queried.extend(selected)
        values.extend(float(value) for value in selected_values)

    return _result("ridge_guided", candidates, queried, np.asarray(values, dtype=float))
