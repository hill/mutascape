"""Walsh-Hadamard variance decomposition of small complete combinatorial landscapes.

For a small number of positions L with observed alphabets A_1, ..., A_L, the
fitness tensor f[a_1, ..., a_L] contains one entry per variant. The variance of
that tensor decomposes exactly into contributions from every subset S of the
positions: subsets of size k contribute the k-th order epistatic variance.

This is Poelwijk et al. 2019 (Nat Commun 10:4213) generalised to categorical
alphabets of arbitrary (possibly different) size, via the sum-of-orthogonal-
projectors identity (equivalent to ANOVA on a complete factorial design). For
|A_i| = 2 this reduces to the classical binary Walsh-Hadamard decomposition.

Missing entries are filled with the tensor mean prior to decomposition so the
subsequent variance accounting stays orthogonal; the coverage fraction is
reported alongside so callers can judge how much of the signal is real.
"""

from collections.abc import Sequence
from itertools import combinations, product

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mutascape.landscape import Landscape

from .real_data import parse_proteingym_mutant


def observed_alphabets(
    df: pd.DataFrame,
    positions: Sequence[int],
    wt_aa_at_position: dict[int, str],
) -> dict[int, list[str]]:
    """Alphabets observed at each position in the combinatorial data.

    Includes the WT amino acid even if it never appears explicitly in the
    mutation strings.
    """
    seen: dict[int, set[str]] = {p: {wt_aa_at_position[p]} for p in positions}
    for mutations in df["mutations"]:
        for pos, aa in mutations.items():
            if pos in seen:
                seen[pos].add(aa)
    return {p: sorted(seen[p]) for p in positions}


def fitness_tensor_from_real(
    df: pd.DataFrame,
    positions: Sequence[int],
    wt_aa_at_position: dict[int, str],
    alphabets: dict[int, list[str]],
) -> tuple[NDArray[np.float64], float]:
    """Assemble a fitness tensor from the combinatorial DataFrame.

    Entries for variants absent from the data are left as NaN so the caller
    can decide how to impute. Returns (tensor, coverage_fraction).
    """
    positions = list(positions)
    shape = tuple(len(alphabets[p]) for p in positions)
    tensor = np.full(shape, np.nan, dtype=np.float64)
    index_lookup = [{aa: i for i, aa in enumerate(alphabets[p])} for p in positions]
    wt_index = tuple(index_lookup[i][wt_aa_at_position[p]] for i, p in enumerate(positions))
    tensor[wt_index] = 0.0  # WT fitness is 0 by our convention

    for _, row in df.iterrows():
        mutations = row["mutations"]
        idx = []
        ok = True
        for i, p in enumerate(positions):
            aa = mutations.get(p, wt_aa_at_position[p])
            if aa not in index_lookup[i]:
                ok = False
                break
            idx.append(index_lookup[i][aa])
        if ok:
            tensor[tuple(idx)] = float(row["fitness"])

    coverage = float(np.isfinite(tensor).sum()) / tensor.size
    return tensor, coverage


def fitness_tensor_from_landscape(
    landscape: Landscape,
    positions: Sequence[int],
    alphabets: dict[int, list[str]],
) -> NDArray[np.float64]:
    """Evaluate a mutascape landscape on every variant in the product space."""
    positions = list(positions)
    shape = tuple(len(alphabets[p]) for p in positions)
    tensor = np.empty(shape, dtype=np.float64)

    all_variants = []
    all_indices = []
    for combo in product(*(alphabets[p] for p in positions)):
        mutations = {pos: aa for pos, aa in zip(positions, combo) if aa != landscape.wild_type[pos]}
        all_variants.append(mutations)
        idx = tuple(alphabets[p].index(aa) for p, aa in zip(positions, combo))
        all_indices.append(idx)

    fitness = landscape.evaluate_batch(all_variants)
    for idx, f in zip(all_indices, fitness):
        tensor[idx] = f
    return tensor


def _anova_effects(
    tensor: NDArray[np.float64],
) -> dict[tuple[int, ...], NDArray[np.float64]]:
    """Signed ANOVA effect tensor for every subset S of the axes.

    `effects[S]` keeps the S-axes at full size and the complement axes at size 1
    (via keepdims), holding the pure interaction effect of the positions in S
    after removing all lower-order effects. Summing every effect (including the
    empty grand-mean set) reconstructs the tensor. This is the shared core of
    both the variance decomposition (`walsh_decompose`) and the signed
    coefficient vector (`walsh_signed_effects`).
    """
    L = tensor.ndim
    all_axes = set(range(L))

    # S-projected tensors: E[tensor | dims in S are fixed], keyed by sorted S.
    means: dict[tuple[int, ...], NDArray[np.float64]] = {}
    for k in range(L + 1):
        for S in combinations(range(L), k):
            complement = tuple(sorted(all_axes - set(S)))
            if complement:
                means[S] = tensor.mean(axis=complement, keepdims=True)
            else:
                means[S] = tensor.copy()

    # Effect of S is its projection minus the effects of all its strict subsets
    # (inclusion-exclusion over the ANOVA projection).
    effects: dict[tuple[int, ...], NDArray[np.float64]] = {}
    for k in range(L + 1):
        for S in combinations(range(L), k):
            e = means[S].copy()
            for j in range(k):
                for Ssub in combinations(S, j):
                    e = e - effects[Ssub]
            effects[S] = e
    return effects


def walsh_decompose(tensor: NDArray[np.float64]) -> dict[int, float]:
    """Variance fraction carried by each interaction order.

    For an L-way tensor, subsets of size k (k = 1..L) contribute the k-th
    order variance; the 0-th order "subset" is the grand mean and carries
    no variance. Returns a dict mapping order -> variance fraction summing
    to 1 over k=1..L (within floating-point).

    Implementation: classical ANOVA decomposition on a complete factorial.
    For every non-empty subset S, the contribution is the sum of squared
    deviations of the S-averaged tensor from lower-order contributions,
    multiplied by the product of the sizes of the non-S dimensions.
    """
    if tensor.ndim == 0:
        return {}

    if np.isnan(tensor).any():
        tensor = np.where(np.isnan(tensor), np.nanmean(tensor), tensor)

    L = tensor.ndim
    total_var = float(np.var(tensor)) * tensor.size
    if total_var <= 0:
        return {k: 0.0 for k in range(1, L + 1)}

    effects = _anova_effects(tensor)
    order_ssq: dict[int, float] = {k: 0.0 for k in range(1, L + 1)}
    for S, e in effects.items():
        if len(S) < 1:
            continue
        broadcast_factor = 1
        for d in range(L):
            if d not in S:
                broadcast_factor *= tensor.shape[d]
        order_ssq[len(S)] += float(np.sum(e**2)) * broadcast_factor

    return {k: order_ssq[k] / total_var for k in range(1, L + 1)}


def variance_fraction_comparison(
    real_tensor: NDArray[np.float64],
    synth_tensor: NDArray[np.float64],
) -> dict[str, float | dict[int, float]]:
    """Compute Walsh decompositions of real and synthetic tensors side-by-side."""
    real = walsh_decompose(real_tensor)
    synth = walsh_decompose(synth_tensor)
    per_order_abs_diff = {k: abs(real[k] - synth[k]) for k in real}
    return {
        "real": real,
        "synth": synth,
        "abs_diff_per_order": per_order_abs_diff,
        "max_abs_diff": (
            max(per_order_abs_diff.values()) if per_order_abs_diff else 0.0
        ),
        "mean_abs_diff": (
            float(np.mean(list(per_order_abs_diff.values())))
            if per_order_abs_diff
            else 0.0
        ),
        "l1_abs_diff": float(sum(per_order_abs_diff.values())),
    }


def walsh_l1_distance(
    real: dict[int, float],
    synth: dict[int, float],
) -> float:
    """Sum of |real - synth| variance fractions across shared interaction orders.

    A single scalar "how different are the two Walsh spectra." Ranges from 0
    (identical spectra) to 2 (one spectrum entirely in one order the other
    entirely in a different order, since fractions sum to 1 on each side).
    """
    orders = sorted(set(real) & set(synth))
    return float(sum(abs(real[k] - synth[k]) for k in orders))


def order_wasserstein(
    real: dict[int, float],
    synth: dict[int, float],
) -> float:
    """1-Wasserstein distance between two per-order variance-fraction spectra.

    Treats interaction order as an ordinal axis, so moving variance from order 2
    to order 3 costs less than moving it from order 1 to order 15, which the
    plain L1 of `walsh_l1_distance` cannot express. Computed as the sum of
    absolute differences of the cumulative distributions over the shared orders
    (the closed form of 1-Wasserstein on unit-spaced support). Each side is
    renormalised to sum 1 first so the inputs need not be exactly normalised.
    Same family as the existing `fitness_W1`. Lower is better, 0 is identical.
    """
    orders = sorted(set(real) & set(synth))
    if not orders:
        return 0.0
    r = np.array([real[k] for k in orders], dtype=np.float64)
    s = np.array([synth[k] for k in orders], dtype=np.float64)
    if r.sum() > 0:
        r = r / r.sum()
    if s.sum() > 0:
        s = s / s.sum()
    return float(np.sum(np.abs(np.cumsum(r) - np.cumsum(s))))


def spectral_cosine(
    real_coeffs: NDArray[np.float64],
    synth_coeffs: NDArray[np.float64],
) -> float:
    """Cosine similarity between two signed Walsh-coefficient vectors.

    1.0 means identical direction (the same interaction pattern up to a positive
    scale), 0.0 orthogonal, -1.0 opposite sign. Scale-invariant by construction,
    so it tests interaction STRUCTURE (which subsets carry the signal, with what
    relative sign and magnitude) rather than overall magnitude. The magnitude is
    handled separately by the per-order spectra. Returns 0.0 if either vector
    has zero norm.
    """
    real = np.asarray(real_coeffs, dtype=np.float64).ravel()
    synth = np.asarray(synth_coeffs, dtype=np.float64).ravel()
    nr = float(np.linalg.norm(real))
    ns = float(np.linalg.norm(synth))
    if nr == 0.0 or ns == 0.0:
        return 0.0
    return float(np.dot(real, synth) / (nr * ns))


def walsh_signed_effects(
    tensor: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Flattened signed Walsh/ANOVA coefficient vector and its per-entry orders.

    Returns `(coeffs, orders)` where `coeffs[i]` is a signed interaction
    coefficient and `orders[i]` is its interaction order |S|. The squared
    entries sum to the total sum of squares about the mean (Parseval), with each
    subset's unique effect values weighted by sqrt(broadcast_factor) so the
    per-order squared sums equal the ANOVA variance contributions. The grand
    mean (order 0) is included so callers can mask it out.

    Two tensors evaluated on the SAME alphabets produce coefficient vectors that
    are aligned element by element (subsets are emitted in a deterministic
    (order, sorted-subset) order), so `spectral_cosine` on a masked slice
    compares like with like. This generalises the binary-cube
    `binary_walsh_coefficients` to categorical alphabets of arbitrary size.
    """
    if np.isnan(tensor).any():
        tensor = np.where(np.isnan(tensor), np.nanmean(tensor), tensor)

    L = tensor.ndim
    effects = _anova_effects(tensor)
    coeff_parts: list[NDArray[np.float64]] = []
    order_parts: list[NDArray[np.int64]] = []
    for S in sorted(effects, key=lambda s: (len(s), s)):
        broadcast_factor = 1
        for d in range(L):
            if d not in S:
                broadcast_factor *= tensor.shape[d]
        vals = np.atleast_1d(np.squeeze(effects[S])).ravel() * np.sqrt(
            broadcast_factor
        )
        coeff_parts.append(vals)
        order_parts.append(np.full(vals.size, len(S), dtype=np.int64))

    return np.concatenate(coeff_parts), np.concatenate(order_parts)


def binary_walsh_coefficients(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Orthonormal fast Walsh-Hadamard transform of a binary (2x...x2) cube.

    Returns a same-shaped tensor of signed coefficients. Orthonormal so the sum
    of squared coefficients equals the sum of squared inputs (Parseval). Flatten
    it and pair with `index_orders(L)` to slice coefficients by interaction
    order.
    """
    a = np.asarray(tensor, dtype=np.float64).copy()
    for axis in range(a.ndim):
        a = np.moveaxis(a, axis, 0)
        s = a.shape
        a = a.reshape((2, -1))
        x0, x1 = a[0].copy(), a[1].copy()
        a[0] = (x0 + x1) / np.sqrt(2)
        a[1] = (x0 - x1) / np.sqrt(2)
        a = a.reshape(s)
        a = np.moveaxis(a, 0, axis)
    return a


def index_orders(L: int) -> NDArray[np.int64]:
    """Hamming weight (interaction order) of each flat index of a 2^L cube."""
    return np.array([bin(i).count("1") for i in range(2**L)], dtype=np.int64)


def aggregate_binary_cube(
    values: NDArray[np.float64],
    cell_idx: NDArray[np.int64],
    L: int,
) -> NDArray[np.float64]:
    """Average per-variant values into a 2^L binary cube; empty cells are NaN."""
    n_cells = 2**L
    sums = np.zeros(n_cells, dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(sums, cell_idx, values)
    np.add.at(counts, cell_idx, 1)
    out = np.full(n_cells, np.nan, dtype=np.float64)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out.reshape((2,) * L)


def fill_cube_mean(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fill NaN cells of a cube with the mean of its observed cells."""
    flat = tensor.ravel()
    mean = float(np.nanmean(flat))
    return np.where(np.isnan(flat), mean, flat).reshape(tensor.shape)


def binary_cube_spectrum(
    values: NDArray[np.float64],
    cell_idx: NDArray[np.int64],
    L: int,
) -> tuple[NDArray[np.float64], dict[int, float]]:
    """Signed Walsh coefficients and per-order variance fractions for a cube
    assembled from per-variant values aggregated to the 2^L binary projection.
    """
    cube = fill_cube_mean(aggregate_binary_cube(values, cell_idx, L))
    coeffs = binary_walsh_coefficients(cube).ravel()
    orders = index_orders(L)
    sq = coeffs**2
    nonconst = float(sq[orders >= 1].sum())
    if nonconst <= 0:
        fracs = {k: 0.0 for k in range(1, L + 1)}
    else:
        fracs = {k: float(sq[orders == k].sum() / nonconst) for k in range(1, L + 1)}
    return coeffs, fracs


def truncate_fractions(
    fracs: dict[int, float], max_order: int | None
) -> dict[int, float]:
    """Keep orders 1..max_order and renormalise to sum 1.

    The high-order tail of a Walsh spectrum estimated from noisy measurements is
    dominated by noise, not signal (e.g. real CreiLOV carries ~97% of its
    variance in orders 1-2). Truncating to the signal-bearing orders before
    computing a distance keeps the comparison from being swamped by that tail.
    `max_order=None` is a no-op.
    """
    if max_order is None:
        return dict(fracs)
    kept = {k: v for k, v in fracs.items() if k <= max_order}
    total = sum(kept.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in kept.items()}


def binary_cube_spectral_metrics(
    real_values: NDArray[np.float64],
    synth_values: NDArray[np.float64],
    cell_idx: NDArray[np.int64],
    L: int,
    max_order: int | None = None,
) -> dict[str, float]:
    """Spectral comparison metrics for one real/synth pair on a 2^L binary cube.

    Returns order-2 cosine and order>=2 cosine (interaction *pattern* agreement,
    scale-invariant), the ordinal per-order Wasserstein distance, and the legacy
    per-order L1 (interaction *budget* agreement, magnitude-sensitive). With
    `max_order` set, the two per-order distances are computed on the signal-
    bearing orders 1..max_order (renormalised) instead of the full, noise-tailed
    spectrum; the cosines are unaffected by this argument.
    """
    orders = index_orders(L)
    real_coeffs, real_fracs = binary_cube_spectrum(real_values, cell_idx, L)
    synth_coeffs, synth_fracs = binary_cube_spectrum(synth_values, cell_idx, L)
    real_t = truncate_fractions(real_fracs, max_order)
    synth_t = truncate_fractions(synth_fracs, max_order)
    return {
        "cos_order2": spectral_cosine(
            real_coeffs[orders == 2], synth_coeffs[orders == 2]
        ),
        "cos_ge2": spectral_cosine(
            real_coeffs[orders >= 2], synth_coeffs[orders >= 2]
        ),
        "order_w1": order_wasserstein(real_t, synth_t),
        "walsh_l1": walsh_l1_distance(real_t, synth_t),
    }


def proteingym_to_mutations_df(
    combinatorial_csv,
    positions: Sequence[int],
    wt_fitness: float = 1.0,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Load a prepared combinatorial CSV into the standard validation schema.

    Expects columns `mutant`, `fitness`, `depth` as written by prepare_gb1.py
    (mutation strings already in domain numbering, fitness relative to WT).
    Returns (DataFrame with mutations dict, fitness, depth; wt_aa_at_position).
    """
    raw = pd.read_csv(combinatorial_csv)
    rows = []
    wt_aa: dict[int, str] = {}
    for _, row in raw.iterrows():
        parsed = parse_proteingym_mutant(row["mutant"])
        mutations = {}
        for pos, (wt, mut) in parsed.items():
            wt_aa.setdefault(pos, wt)
            mutations[pos] = mut
        rows.append({
            "mutations": mutations,
            "fitness": float(row["fitness"]),
            "depth": int(row["depth"]),
        })
    df = pd.DataFrame(rows)
    # Filter to variants strictly within the requested positions
    pos_set = set(positions)
    df = df[df["mutations"].apply(lambda m: set(m.keys()).issubset(pos_set))].reset_index(drop=True)
    return df, wt_aa
