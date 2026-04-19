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

    # Precompute the S-projected tensors: E[tensor | dims in S are fixed]
    # keyed by tuple(sorted(S)).
    means: dict[tuple[int, ...], NDArray[np.float64]] = {}
    all_axes = set(range(L))
    for k in range(L + 1):
        for S in combinations(range(L), k):
            complement = tuple(sorted(all_axes - set(S)))
            if complement:
                means[S] = tensor.mean(axis=complement, keepdims=True)
            else:
                means[S] = tensor.copy()

    # Effect of a subset S is its projection minus the sum of effects of
    # all its strict subsets (inclusion-exclusion over the ANOVA projection).
    effects: dict[tuple[int, ...], NDArray[np.float64]] = {}
    ssq: dict[tuple[int, ...], float] = {}
    for k in range(L + 1):
        for S in combinations(range(L), k):
            e = means[S].copy()
            for j in range(k):
                for Ssub in combinations(S, j):
                    e = e - effects[Ssub]
            effects[S] = e
            broadcast_factor = 1
            for d in range(L):
                if d not in S:
                    broadcast_factor *= tensor.shape[d]
            ssq[S] = float(np.sum(e**2)) * broadcast_factor

    order_ssq: dict[int, float] = {k: 0.0 for k in range(1, L + 1)}
    for S, s in ssq.items():
        if len(S) >= 1:
            order_ssq[len(S)] += s

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
