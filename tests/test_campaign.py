"""Budget, rounds, scoring, and assay noise for the public campaign interface."""

import numpy as np
import pandas as pd
import pytest

from mutascape import Campaign, run_campaign
from mutascape.completion import LandscapePrior
from mutascape.legacy import PosteriorLandscape

WILD_TYPE = "VDGV"


def _training_frame(effects: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = [{"sequence": WILD_TYPE, "fitness": 0.0, "depth": 0, "mutations": {}}]
    for position, residues in effects.items():
        for amino_acid, fitness in residues.items():
            rows.append(
                {
                    "sequence": WILD_TYPE,
                    "fitness": fitness,
                    "depth": 1,
                    "mutations": {position: amino_acid},
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def oracle():
    effects = {
        1: {"A": 0.4, "C": -0.9, "E": 0.1},
        2: {"A": -0.3, "C": 0.7, "E": -1.2},
        3: {"A": 0.2, "C": -0.5, "E": 0.9},
        4: {"A": -0.7, "C": 0.3, "E": -0.1},
    }
    frame = _training_frame(effects)
    prior = LandscapePrior(
        order_location={2: 0.0},
        order_scale={2: 1.0},
        order_shape={2: tuple(np.linspace(-2.0, 2.0, 257).tolist())},
        depth_location={},
        depth_scale={},
        depth_between={},
        depth_shrinkage={},
        magnitude_edges=(0.0, 1.0),
        magnitude_scale=(1.0,),
        texture_scale=0.0,
        readout_scale_ratio=1.0,
        combinatorial_upside=1.5,
        combinatorial_downside=1.5,
        lower_bound=-3.0,
        upper_bound=2.0,
        n_landscapes=1,
        n_proteins=1,
    )
    generator = PosteriorLandscape.condition(WILD_TYPE, frame, prior)
    return generator.sample_dms_pairwise(seed=7, coupling=0.1)


@pytest.fixture
def candidates(oracle):
    return [mutant for mutant in oracle.domain.sample(depth=2, n=40, seed=3)]


def test_measure_spends_budget_once_per_variant(oracle, candidates):
    campaign = Campaign(oracle, candidates, budget=10)
    first = campaign.measure(candidates[:4])
    assert campaign.spent == 4
    repeat = campaign.measure(candidates[:4])
    assert campaign.spent == 4
    np.testing.assert_allclose(first, repeat)


def test_budget_is_a_hard_stop(oracle, candidates):
    campaign = Campaign(oracle, candidates, budget=5)
    campaign.measure(candidates[:5])
    beyond = campaign.measure(candidates[5:8])
    assert campaign.remaining == 0
    assert np.isnan(beyond).all()


def test_rounds_advance_with_each_batch(oracle, candidates):
    campaign = Campaign(oracle, candidates, budget=20)
    campaign.measure(candidates[:3])
    campaign.measure(candidates[3:6])
    assert campaign.round == 2
    assert campaign.observed["round"].tolist() == [0, 0, 0, 1, 1, 1]


def test_result_scores_against_the_whole_candidate_set(oracle, candidates):
    def method(campaign: Campaign) -> None:
        campaign.measure(campaign.candidates)

    result = run_campaign(method, oracle, candidates, budget=len(candidates))
    assert result.queries == len(candidates)
    assert result.simple_regret == pytest.approx(0.0)
    assert result.best_found == pytest.approx(result.optimum)


def test_regret_is_positive_when_the_optimum_is_missed(oracle, candidates):
    def method(campaign: Campaign) -> None:
        campaign.measure(campaign.sample(4))

    result = run_campaign(method, oracle, candidates, budget=4, seed=1)
    assert 0.0 < result.simple_regret <= 1.0
    assert result.best_found <= result.optimum


def test_encode_is_one_hot_over_observed_substitutions(oracle, candidates):
    campaign = Campaign(oracle, candidates, budget=5)
    design = campaign.encode()
    assert design.shape[0] == len(candidates)
    assert set(np.unique(design)) <= {0.0, 1.0}
    # Every candidate here carries two substitutions.
    np.testing.assert_array_equal(design.sum(axis=1), np.full(len(candidates), 2.0))


def test_encode_selects_rows_for_named_variants(oracle, candidates):
    campaign = Campaign(oracle, candidates, budget=5)
    rows = campaign.encode(candidates[:3])
    np.testing.assert_array_equal(rows, campaign.encode()[:3])


def test_noise_perturbs_reads_but_not_the_score(oracle, candidates):
    quiet = Campaign(oracle, candidates, budget=40, noise=0.0)
    noisy = Campaign(oracle, candidates, budget=40, noise=0.5, seed=11)
    clean_reads = quiet.measure(candidates)
    noisy_reads = noisy.measure(candidates)
    assert not np.allclose(clean_reads, noisy_reads)
    assert quiet.result().optimum == pytest.approx(noisy.result().optimum)


def test_noise_is_stable_for_a_repeated_read(oracle, candidates):
    campaign = Campaign(oracle, candidates, budget=10, noise=0.5, seed=11)
    first = campaign.measure(candidates[:3])
    repeat = campaign.measure(candidates[:3])
    np.testing.assert_allclose(first, repeat)


def test_noise_scores_the_truth_under_the_selected_read(oracle, candidates):
    campaign = Campaign(oracle, candidates, budget=40, noise=1.5, seed=5)
    campaign.measure(candidates)
    result = campaign.result()
    # The shipped variant is chosen by its read, so its true fitness can sit
    # below the best true fitness the campaign actually measured.
    assert result.best_found <= result.optimum
    assert result.best_observed != pytest.approx(result.best_found)


def test_duplicate_candidates_are_rejected(oracle, candidates):
    with pytest.raises(ValueError, match="duplicate"):
        Campaign(oracle, [*candidates, candidates[0]], budget=5)


def test_invalid_settings_are_rejected(oracle, candidates):
    with pytest.raises(ValueError, match="budget"):
        Campaign(oracle, candidates, budget=0)
    with pytest.raises(ValueError, match="batch_size"):
        Campaign(oracle, candidates, budget=5, batch_size=0)
    with pytest.raises(ValueError, match="noise"):
        Campaign(oracle, candidates, budget=5, noise=-1.0)
    with pytest.raises(ValueError, match="candidate"):
        Campaign(oracle, [], budget=5)


def test_readme_flow_runs_end_to_end(oracle, candidates):
    """The documented shape: draw an oracle, sample a domain, run rounds."""
    generator = oracle.generator
    plate = 8
    wanted = min(60, generator.domain.count(depth=2))
    domain_candidates = generator.domain.sample(depth=2, n=wanted, seed=0)

    def method(campaign: Campaign) -> None:
        campaign.measure(campaign.sample(plate))
        while campaign.remaining:
            design = campaign.encode()
            observed = campaign.observed
            assert design.shape[0] == len(campaign.candidates)
            assert set(observed.columns) == {"variant", "fitness", "round"}
            picks = campaign.sample(min(plate, campaign.remaining))
            if not picks:
                break
            campaign.measure(picks)

    result = run_campaign(
        method,
        generator.draw(seed=42),
        domain_candidates,
        budget=32,
        batch_size=plate,
        noise=0.1,
    )
    assert result.queries == 32
    assert result.rounds == 4
    assert 0.0 <= result.simple_regret <= 1.0


def test_sample_draws_only_unmeasured_candidates(oracle, candidates):
    campaign = Campaign(oracle, candidates, budget=40)
    campaign.measure(candidates[:35])
    drawn = campaign.sample(10)
    assert len(drawn) == len(candidates) - 35
    measured = set(campaign.observed["variant"])
    assert not measured & {mutant.label for mutant in drawn}
