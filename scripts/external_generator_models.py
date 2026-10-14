"""Score native external landscape families on explicit genotype domains."""

import sys
from itertools import combinations

import numpy as np

from scripts.fit_slip_comparators import ALPHABET, CACHE


def nk_values(codes, sizes, k, seed):
    """Evaluate categorical NK with uniform independent contribution tables."""
    rng = np.random.default_rng(seed)
    n_sites = len(sizes)
    if k < 0 or k >= n_sites:
        raise ValueError("K must be at least zero and below the number of sites")
    output = np.zeros(len(codes))
    for site in range(n_sites):
        others = np.delete(np.arange(n_sites), site)
        group = np.sort(np.r_[site, rng.choice(others, k, replace=False)])
        table = rng.uniform(size=tuple(int(sizes[i]) for i in group))
        output += table[tuple(codes[:, i] for i in group)]
    return output / n_sites


def rmf_values(codes, sizes, seed, noise=1.0):
    """Evaluate a Hamming slope towards a random reference plus genotype noise."""
    rng = np.random.default_rng(seed)
    reference = np.array([rng.integers(s) for s in sizes])
    # Generate noise on the full categorical domain so shared genotypes agree.
    flat = np.ravel_multi_index(codes.T, sizes)
    rng_noise = np.random.default_rng(seed + 1)
    deviations = rng_noise.normal(size=int(np.prod(sizes)))[flat]
    return -(codes != reference).sum(axis=1).astype(float) + noise * deviations


def potts_components(weight, bias, wt):
    """Convert Mogwai potentials to exact effects on the full WT background."""
    weight, bias = np.asarray(weight, float), np.asarray(bias, float)
    n, q = bias.shape
    field = bias.copy()
    for site in range(n):
        field += weight[:, :, site, wt[site]]
    field -= field[np.arange(n), wt, None]
    return field


def pair_effect(weight, wt, first, second):
    table = weight[first, :, second, :].astype(float)
    return (
        table
        - table[:, wt[second], None]
        - table[wt[first], None, :]
        + table[wt[first], wt[second]]
    )


def slip_scales(weight, field, wt, horizon, adaptive_fraction=None, negative_pair_fraction=None):
    """Apply the published SLIP tuning controls over all aligned positions."""
    mask = np.ones(field.shape, dtype=bool)
    mask[np.arange(len(wt)), wt] = False
    single_std = float(field[mask].std())
    single_offset = 0.0
    if adaptive_fraction is not None:
        single_offset = -float(np.quantile(field[mask], 1 - adaptive_fraction))
    shifted = field + single_offset * mask
    adaptive = mask & (shifted >= 0)
    mean_single = float(shifted[adaptive].mean())
    blocks = []
    for i, j in combinations(range(len(wt)), 2):
        block = pair_effect(weight, wt, i, j)[np.ix_(adaptive[i], adaptive[j])]
        if block.size:
            blocks.append(block.ravel())
    effects = np.concatenate(blocks) if blocks else np.array([])
    count = len(effects)
    epi_offset = 0.0
    if negative_pair_fraction is not None and count:
        epi_offset = -float(np.quantile(effects, negative_pair_fraction))
    mean_epi = float((effects + epi_offset).mean()) if count else np.nan
    if not np.isfinite(mean_epi) or mean_epi == 0 or single_std == 0:
        raise ValueError("SLIP tuning is undefined for this fitted model")
    scale = 1 / single_std
    coupling = -2 * scale * mean_single / ((horizon - 1) * mean_epi)
    return (
        scale,
        coupling,
        dict(
            single_std=single_std,
            adaptive_single_fraction=float(adaptive.sum() / mask.sum()),
            mean_adaptive_single=mean_single,
            mean_adaptive_pair=mean_epi,
            adaptive_pairs=count,
            field_scale=scale,
            coupling_scale=coupling,
            horizon=horizon,
            negative_coupling_scale=bool(coupling < 0),
            single_mut_offset=single_offset,
            epi_offset=epi_offset,
            requested_adaptive_fraction=adaptive_fraction,
            requested_negative_pair_fraction=negative_pair_fraction,
        ),
    )


def slip_values(protein, codes, positions, alphabets, full_positions, horizon=None):
    """Use exact WT-centred effects and verify against official SLIP scoring."""
    archive = np.load(CACHE / protein / "mogwai_1000.npz")
    weight, bias, wt = archive["weight"], archive["bias"], archive["query_seq"]
    field = potts_components(weight, bias, wt)
    match_lookup = {int(p): i for i, p in enumerate(archive["positions"])}
    sites = [match_lookup[full_positions[p]] for p in positions]
    residues = [np.array([ALPHABET.index(a) for a in alphabets[p]]) for p in positions]
    sf, se, report = (
        (1.0, 1.0, {})
        if horizon is None
        else slip_scales(
            weight, field, wt, horizon, adaptive_fraction=0.1, negative_pair_fraction=0.8
        )
    )
    single_offset, epi_offset = report.get("single_mut_offset", 0), report.get("epi_offset", 0)
    output = np.zeros(len(codes))
    for i, site in enumerate(sites):
        states = residues[i][codes[:, i]]
        output += sf * (field[site, states] + single_offset * (states != wt[site]))
    for i, j in combinations(range(len(sites)), 2):
        table = pair_effect(weight, wt, sites[i], sites[j])
        output += se * table[residues[i][codes[:, i]], residues[j][codes[:, j]]]
        output += se * epi_offset * (codes[:, i] != 0) * (codes[:, j] != 0)
    sys.path.insert(0, str((CACHE / "slip").resolve()))
    from potts_model import PottsModel

    native = PottsModel(
        -weight.astype(float).transpose(0, 2, 1, 3),
        -bias.astype(float),
        wt,
        field_scale=sf,
        coupling_scale=se,
        single_mut_offset=single_offset,
        epi_offset=epi_offset,
    )
    rows = np.unique(np.linspace(0, len(codes) - 1, 32).astype(int))
    sequences = np.tile(wt, (len(rows), 1))
    for i, site in enumerate(sites):
        sequences[:, site] = residues[i][codes[rows, i]]
    expected = native.evaluate(sequences)
    error = float(np.max(np.abs(expected - output[rows])))
    # Official SLIP applies its offsets through float32 one-hot arrays.
    np.testing.assert_allclose(output[rows], expected, atol=3e-5, rtol=2e-6)
    report.update(native_scoring_max_error=error, validation_genotypes=len(rows))
    return output, report
