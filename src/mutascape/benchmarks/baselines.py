"""Small, auditable baselines for landscape-completion benchmarks."""

from dataclasses import dataclass
from itertools import combinations
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge


def _feature_dict(mutations: dict[int, str], max_order: int) -> dict[str, float]:
    states = tuple(
        sorted((int(position), amino_acid) for position, amino_acid in mutations.items())
    )
    features = {f"{position}:{amino_acid}": 1.0 for position, amino_acid in states}
    for order in range(2, min(max_order, len(states)) + 1):
        for group in combinations(states, order):
            label = "|".join(f"{position}:{amino_acid}" for position, amino_acid in group)
            features[label] = 1.0
    return features


@dataclass
class RidgeLandscapeBaseline:
    """Ridge model over mutation identities and optional interaction features."""

    max_order: int = 1
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.max_order < 1:
            raise ValueError("max_order must be positive")
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = Ridge(alpha=self.alpha)
        self._fitted = False

    def fit(self, observations: pd.DataFrame) -> Self:
        """Fit the model to normalised observations."""
        required = {"mutations", "fitness"}
        missing = sorted(required - set(observations.columns))
        if missing:
            raise ValueError(f"observations are missing columns: {', '.join(missing)}")
        features = [_feature_dict(value, self.max_order) for value in observations["mutations"]]
        matrix = self.vectorizer.fit_transform(features)
        self.model.fit(matrix, observations["fitness"].to_numpy(dtype=float))
        self._fitted = True
        return self

    def predict(self, variants: list[dict[int, str]]) -> NDArray[np.float64]:
        """Predict fitness for a batch of mutation dictionaries."""
        if not self._fitted:
            raise RuntimeError("fit the baseline before prediction")
        features = [_feature_dict(value, self.max_order) for value in variants]
        matrix = self.vectorizer.transform(features)
        return np.asarray(self.model.predict(matrix), dtype=np.float64)
