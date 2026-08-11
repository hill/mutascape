"""Tests for structure-conditioned sparse-landscape metrics."""

import numpy as np
import pandas as pd
import pytest

from mutascape.contacts import ContactMap
from mutascape.validation.structural_metrics import (
    add_contact_features,
    contact_bin_profile,
    contact_epistasis_regression,
)


def test_add_contact_features_counts_induced_contacts():
    contact_map = ContactMap(
        matrix=np.array([
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]),
        positions=(1, 2, 3),
        threshold=8.0,
        distances=np.array([
            [0.0, 5.0, 12.0],
            [5.0, 0.0, 9.0],
            [12.0, 9.0, 0.0],
        ]),
    )
    df = pd.DataFrame({"mutations": [{1: "A", 2: "C", 3: "D"}, {1: "A"}]})

    out = add_contact_features(df, contact_map)

    assert out.loc[0, "possible_contact_pairs"] == pytest.approx(3.0)
    assert out.loc[0, "contact_count"] == pytest.approx(1.0)
    assert out.loc[0, "contact_fraction"] == pytest.approx(1.0 / 3.0)
    assert out.loc[0, "largest_contact_component_fraction"] == pytest.approx(2.0 / 3.0)
    assert out.loc[1, "possible_contact_pairs"] == pytest.approx(0.0)
    assert np.isnan(out.loc[1, "contact_proximity"])


def test_contact_epistasis_regression_detects_positive_enrichment():
    df = pd.DataFrame({
        "abs_epistatic_deviation": [0.0, 1.0, 2.0, 3.0, 4.0],
        "contact_proximity": [0.0, 0.25, 0.5, 0.75, 1.0],
    })

    result = contact_epistasis_regression(df, controls=())

    assert result.n == 5
    assert result.beta > 0.99
    assert result.partial_r2 > 0.99


def test_contact_bin_profile_returns_quantile_means():
    df = pd.DataFrame({
        "abs_epistatic_deviation": [1.0, 2.0, 3.0, 4.0],
        "contact_proximity": [0.1, 0.2, 0.8, 0.9],
    })

    profile = contact_bin_profile(df, n_bins=2)

    assert profile["n"].tolist() == [2, 2]
    assert profile["value_mean"].tolist() == [1.5, 3.5]
