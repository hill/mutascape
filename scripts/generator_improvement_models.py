"""Construct DMS-preserving experimental landscapes without combination labels."""

from itertools import combinations

import numpy as np

from mutascape.model import DMSLandscape, _pair

MODELS = (
    "wide_sigmoid",
    "effect_pairs",
    "diminishing4",
    "diminishing8",
    "power_pool15",
    "msa_slip4",
    "msa_slip8",
    "msa_regression",
)
STOCHASTIC = MODELS[:5]


def single_site_decomposition(codes, known_values):
    """Return WT, per-site measured effects and their additive sum."""
    codes = np.asarray(codes)
    depth = (codes != 0).sum(axis=1)
    known = np.flatnonzero(depth <= 1)
    supplied = np.asarray(known_values, dtype=float)
    if supplied.shape != (len(known),):
        raise ValueError("Supply values only for WT and single-site genotypes")
    wt_rows = np.flatnonzero(depth[known] == 0)
    if len(wt_rows) != 1:
        raise ValueError("The search space must contain one WT genotype")
    wt = float(supplied[wt_rows[0]])
    effects = np.zeros(codes.shape, dtype=float)
    for site in range(codes.shape[1]):
        for state in np.unique(codes[:, site]):
            if state == 0:
                continue
            rows = np.flatnonzero((depth[known] == 1) & (codes[known, site] == state))
            if len(rows) != 1:
                raise ValueError("Every substitution must have one measured single-site value")
            effects[codes[:, site] == state, site] = supplied[rows[0]] - wt
    return wt, effects, wt + effects.sum(axis=1)


def replace_single_effects(codes, measured_known, prior, regression=False):
    """Retain prior interactions after replacing its additive single-site effects."""
    known = np.flatnonzero((codes != 0).sum(axis=1) <= 1)
    single = (codes[known] != 0).sum(axis=1) == 1
    wt, _, additive = single_site_decomposition(codes, measured_known)
    prior_wt, _, prior_additive = single_site_decomposition(codes, prior[known])
    x = prior[known][single] - prior_wt
    y = np.asarray(measured_known)[single] - wt
    if np.std(x) < 1e-12:
        scale = 0.0
    elif regression:
        scale = max(0.0, float(np.mean((x - x.mean()) * (y - y.mean())) / np.var(x)))
    else:
        scale = float(np.std(y) / np.std(x))
    residual = prior - prior_additive
    np.testing.assert_allclose(residual[known], 0, atol=1e-10)
    values = additive + scale * residual
    values[known] = measured_known
    return values, dict(
        interaction_scale=scale, single_effect_correlation=float(np.corrcoef(x, y)[0, 1])
    )


def candidate_values(model, codes, singles, variants, sequence, seed, prior=None):
    """Use only WT, single-site fitness and optional MSA scores to fit a candidate."""
    known = np.flatnonzero((codes != 0).sum(axis=1) <= 1)
    measured_known = singles.fitness.to_numpy()
    wt, effects, additive = single_site_decomposition(codes, measured_known)
    if model.startswith("msa_"):
        if prior is None:
            raise ValueError("An MSA candidate requires prior scores")
        return replace_single_effects(codes, measured_known, prior, model == "msa_regression")
    if model == "wide_sigmoid":
        fitted = DMSLandscape.from_singles(sequence, singles, margin=1.0, seed=seed)
        return fitted.evaluate_batch(variants), dict(margin=1.0, strength=0.3)
    fitted = DMSLandscape.from_singles(sequence, singles, mode="linear", strength=0.3, seed=seed)
    random_pairs = fitted.evaluate_batch(variants) - additive
    metadata = {}
    if model == "effect_pairs":
        values = additive.copy()
        positions = sorted({p for v in variants for p in v})
        if len(positions) != codes.shape[1]:
            raise ValueError("The genotype columns must match the mutation positions")
        for i, j in combinations(range(len(positions)), 2):
            pi, pj = positions[i], positions[j]
            for a in np.unique(codes[:, i]):
                if a == 0:
                    continue
                for b in np.unique(codes[:, j]):
                    if b == 0:
                        continue
                    rows = np.flatnonzero((codes[:, i] == a) & (codes[:, j] == b))
                    if not len(rows):
                        continue
                    v = variants[rows[0]]
                    coefficient = _pair((pi, v[pi]), (pj, v[pj]), sequence, seed)
                    values[rows] += coefficient * np.sqrt(
                        np.abs(effects[rows, i] * effects[rows, j])
                    )
        metadata["strength"] = 1.0
    elif model.startswith("diminishing"):
        horizon = int(model.removeprefix("diminishing"))
        positive = np.maximum(effects, 0)
        singles_positive = measured_known[measured_known > wt] - wt
        mean = float(singles_positive.mean()) if len(singles_positive) else 0.0
        pair_sum = (positive.sum(axis=1) ** 2 - (positive**2).sum(axis=1)) / 2
        coefficient = 2 / ((horizon - 1) * mean) if mean > 0 else 0.0
        values = additive - coefficient * pair_sum + random_pairs
        metadata.update(horizon=horizon, coefficient=coefficient, strength=0.3)
    elif model == "power_pool15":
        positive = (np.maximum(effects, 0) ** 1.5).sum(axis=1) ** (1 / 1.5)
        negative = ((-np.minimum(effects, 0)) ** 1.5).sum(axis=1) ** (1 / 1.5)
        values = wt + positive - negative + random_pairs
        metadata.update(exponent=1.5, strength=0.3)
    else:
        raise ValueError(f"Unknown model {model}")
    values[known] = measured_known
    return values, metadata
