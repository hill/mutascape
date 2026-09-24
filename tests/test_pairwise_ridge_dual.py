"""Check kernel fitting against explicit indicator regression."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from mutascape.benchmarks.methods import _PairwiseLinear


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("constant", [False, True])
def test_dual_matches_explicit_ridge_with_intercept(order, constant):
    rng = np.random.default_rng(482)
    sizes = (3, 5, 2, 4)
    train = np.column_stack([rng.integers(size - 1, size=53) for size in sizes])
    test = np.column_stack([rng.integers(size, size=29) for size in sizes])
    values = np.full(53, 7.5) if constant else rng.normal(7.5, 2, 53)
    model = _PairwiseLinear(sizes, alpha=0.7, max_order=order)
    reference = Ridge(alpha=0.7).fit(model._design(train).astype(float), values)
    model._fit_dual(train, values)
    np.testing.assert_allclose(
        model.predict(test), reference.predict(model._design(test)), atol=1e-10
    )


def test_large_design_uses_equivalent_fit():
    rng = np.random.default_rng(791)
    sizes = (8,) * 16
    train = rng.integers(8, size=(45, 16))
    test = rng.integers(8, size=(18, 16))
    values = rng.normal(size=45)
    model = _PairwiseLinear(sizes)
    reference = Ridge().fit(model._design(train).astype(float), values)
    model.fit(train, values)
    np.testing.assert_allclose(
        model.predict(test), reference.predict(model._design(test)), atol=1e-10
    )
