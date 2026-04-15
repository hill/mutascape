"""Tests for the scoring module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mutascape.contacts import ContactMap
from mutascape.epistasis import AMINO_ACIDS, RandomEpistasis
from mutascape.landscape import Landscape
from mutascape.scoring import PLMScorer


class MockScorer(PLMScorer):
    """Deterministic scorer for testing. Returns small random-looking effects."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def score(
        self,
        sequence: str,
        positions: list[int] | None = None,
    ) -> dict[int, dict[str, float]]:
        if positions is None:
            positions = list(range(1, len(sequence) + 1))

        results: dict[int, dict[str, float]] = {}
        for pos in positions:
            wt_aa = sequence[pos - 1]
            effects: dict[str, float] = {}
            for aa in AMINO_ACIDS:
                if aa == wt_aa:
                    effects[aa] = 0.0
                else:
                    effects[aa] = float(self._rng.normal(-0.1, 0.3))
            results[pos] = effects
        return results


class TestPLMScorerInterface:
    def test_output_format(self):
        scorer = MockScorer()
        result = scorer.score("ACDEF")
        assert isinstance(result, dict)
        assert len(result) == 5
        for pos in range(1, 6):
            assert pos in result
            assert isinstance(result[pos], dict)
            assert len(result[pos]) == 20

    def test_wt_is_zero(self):
        scorer = MockScorer()
        seq = "ACDEF"
        result = scorer.score(seq)
        for pos in range(1, 6):
            wt_aa = seq[pos - 1]
            assert result[pos][wt_aa] == 0.0

    def test_positions_filter(self):
        scorer = MockScorer()
        result = scorer.score("ACDEF", positions=[1, 3])
        assert set(result.keys()) == {1, 3}

    def test_all_amino_acids_present(self):
        scorer = MockScorer()
        result = scorer.score("A")
        assert set(result[1].keys()) == set(AMINO_ACIDS)


class TestFromPLM:
    @pytest.fixture
    def contact_map(self):
        distances = np.array([
            [0.0, 5.0, 12.0, 15.0, 20.0],
            [5.0, 0.0, 7.0, 10.0, 18.0],
            [12.0, 7.0, 0.0, 4.0, 11.0],
            [15.0, 10.0, 4.0, 0.0, 9.0],
            [20.0, 18.0, 11.0, 9.0, 0.0],
        ])
        return ContactMap.from_distance_matrix(distances, [1, 2, 3, 4, 5], threshold=8.0)

    def test_builds_landscape(self, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_plm(
            sequence="ACDEF",
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            scorer=scorer,
            alpha=1.0,
        )
        assert ls.n_positions == 5
        assert ls.wt_sequence == "ACDEF"

    def test_evaluate_works(self, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_plm(
            sequence="ACDEF",
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            scorer=scorer,
        )
        fitness = ls.evaluate({1: "G"})
        assert isinstance(fitness, float)
        assert np.isfinite(fitness)

    def test_getitem_works(self, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_plm(
            sequence="ACDEF",
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            scorer=scorer,
        )
        assert ls["A1G"] == pytest.approx(ls.evaluate({1: "G"}))

    def test_generate_works(self, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_plm(
            sequence="ACDEF",
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            scorer=scorer,
        )
        variants = list(ls.generate(n_mutations=2, n=5, seed=99))
        assert len(variants) == 5
        for v in variants:
            assert len(v.sequence) == 5

    def test_subset_positions(self, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_plm(
            sequence="ACDEF",
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            scorer=scorer,
            positions=[1, 2, 3],
        )
        assert ls.n_positions == 3
        assert set(ls.positions) == {1, 2, 3}

    def test_wt_fitness_is_zero(self, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_plm(
            sequence="ACDEF",
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            scorer=scorer,
        )
        assert ls.evaluate({}) == pytest.approx(0.0)


class TestHybridDMSandPLM:
    """Test from_dms() with scorer parameter for hybrid landscapes."""

    @pytest.fixture
    def dms_csv(self, tmp_path):
        """DMS data covering only positions 1 and 2."""
        data = {
            "position": [1, 1, 2, 2],
            "wildtype": ["A", "A", "C", "C"],
            "mutant": ["G", "V", "L", "F"],
            "fitness": [0.5, -0.3, 0.2, -0.5],
        }
        path = tmp_path / "dms.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    @pytest.fixture
    def contact_map(self):
        distances = np.array([
            [0.0, 5.0, 12.0, 15.0, 20.0],
            [5.0, 0.0, 7.0, 10.0, 18.0],
            [12.0, 7.0, 0.0, 4.0, 11.0],
            [15.0, 10.0, 4.0, 0.0, 9.0],
            [20.0, 18.0, 11.0, 9.0, 0.0],
        ])
        return ContactMap.from_distance_matrix(distances, [1, 2, 3, 4, 5], threshold=8.0)

    def test_expands_to_all_positions(self, dms_csv, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_dms(
            dms_path=dms_csv,
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            wt_sequence="ACDEF",
            scorer=scorer,
        )
        # DMS covers positions 1, 2. Scorer fills 3, 4, 5.
        assert ls.n_positions == 5
        assert set(ls.positions) == {1, 2, 3, 4, 5}

    def test_dms_takes_priority(self, dms_csv, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_dms(
            dms_path=dms_csv,
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            wt_sequence="ACDEF",
            scorer=scorer,
        )
        # Position 1 should have DMS effects, not pLM predictions
        assert ls.single_site_effects[1]["G"] == pytest.approx(0.5)
        assert ls.single_site_effects[1]["V"] == pytest.approx(-0.3)

    def test_plm_positions_have_all_amino_acids(self, dms_csv, contact_map):
        scorer = MockScorer(seed=42)
        ls = Landscape.from_dms(
            dms_path=dms_csv,
            contact_map=contact_map,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            wt_sequence="ACDEF",
            scorer=scorer,
        )
        # Position 3 is pLM-predicted, should have all 20 AAs
        assert len(ls.single_site_effects[3]) == 20

    def test_scorer_requires_wt_sequence(self, dms_csv, contact_map):
        scorer = MockScorer(seed=42)
        with pytest.raises(ValueError, match="scorer requires wt_sequence"):
            Landscape.from_dms(
                dms_path=dms_csv,
                contact_map=contact_map,
                epistasis=RandomEpistasis(sigma=0.3, seed=42),
                scorer=scorer,
            )


# ESM2 integration tests (only run if torch + esm are installed)
try:
    import esm  # noqa: F401
    import torch  # noqa: F401

    from mutascape.scoring import ESM2Scorer

    _ESM_AVAILABLE = True
except ImportError:
    _ESM_AVAILABLE = False


@pytest.mark.skipif(not _ESM_AVAILABLE, reason="ESM2 not installed")
class TestESM2Scorer:
    def test_score_short_sequence(self, tmp_path):
        scorer = ESM2Scorer(
            model_name="esm2_t6_8M_UR50D",
            device="cpu",
            batch_size=8,
            cache_dir=tmp_path / "cache",
        )
        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = scorer.score(seq)

        assert len(result) == 20
        for pos in range(1, 21):
            wt_aa = seq[pos - 1]
            assert result[pos][wt_aa] == 0.0
            assert len(result[pos]) == 20
            for aa, val in result[pos].items():
                assert np.isfinite(val)

    def test_cache_hit(self, tmp_path):
        scorer = ESM2Scorer(
            model_name="esm2_t6_8M_UR50D",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
        seq = "ACDEFGHIKL"

        result1 = scorer.score(seq)
        # Second call should hit cache
        result2 = scorer.score(seq)

        for pos in result1:
            for aa in result1[pos]:
                assert result1[pos][aa] == result2[pos][aa]

    def test_positions_subset(self, tmp_path):
        scorer = ESM2Scorer(
            model_name="esm2_t6_8M_UR50D",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
        result = scorer.score("ACDEFGHIKL", positions=[1, 5, 10])
        assert set(result.keys()) == {1, 5, 10}
