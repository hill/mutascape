"""Landscape metrics for comparing real and synthetic epistasis patterns.

All functions accept a DataFrame with columns:
    mutations, fitness, additive_prediction, depth, epistatic_deviation

This lets us compute identical metrics on both real multi-mutant data
and synthetic mutascape samples.

The complete-space metrics (local_optima_from_tensor, accessible_paths,
fitness_distance_correlation_from_tensor) operate on a fitness tensor
of shape (|A_1|, ..., |A_L|) for L positions with observed alphabets A_i,
as returned by fourier.fitness_tensor_from_real / _from_landscape.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as sp_stats


def additive_correlation_profile(
    df: pd.DataFrame,
    depths: list[int] | None = None,
) -> dict[int, float]:
    """Pearson correlation between additive prediction and actual fitness at each depth."""
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 3:
            continue
        r, _ = sp_stats.pearsonr(sub["additive_prediction"], sub["fitness"])
        result[d] = r
    return result


def sign_epistasis_profile(
    df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
    depths: list[int] | None = None,
) -> dict[int, float]:
    """Fraction of variants with sign epistasis at each depth.

    A variant has sign epistasis if any constituent mutation has its
    contextual effect (actual - additive_without_this_mutation) with
    opposite sign to its single-site effect.
    """
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 10:
            continue

        n_sign_epi = 0
        for _, row in sub.iterrows():
            mutations = row["mutations"]
            total_fitness = row["fitness"]

            for pos, mut_aa in mutations.items():
                single_effect = single_site_effects.get(pos, {}).get(mut_aa, 0.0)
                if single_effect == 0.0:
                    continue

                # Additive prediction without this mutation
                other_additive = sum(
                    single_site_effects.get(p, {}).get(a, 0.0)
                    for p, a in mutations.items()
                    if p != pos
                )
                contextual_effect = total_fitness - other_additive

                if np.sign(contextual_effect) != np.sign(single_effect):
                    n_sign_epi += 1
                    break

        result[d] = n_sign_epi / len(sub)
    return result


def epistatic_deviation_stats(
    df: pd.DataFrame,
    depths: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Summary statistics of epistatic deviations at each depth.

    Returns dict[depth, {mean, std, mae, skewness, kurtosis, q5, q25, q50, q75, q95}]
    """
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 10:
            continue

        dev = sub["epistatic_deviation"].values
        result[d] = {
            "mean": float(np.mean(dev)),
            "std": float(np.std(dev)),
            "mae": float(np.mean(np.abs(dev))),
            "skewness": float(sp_stats.skew(dev)),
            "kurtosis": float(sp_stats.kurtosis(dev)),
            "q5": float(np.percentile(dev, 5)),
            "q25": float(np.percentile(dev, 25)),
            "q50": float(np.percentile(dev, 50)),
            "q75": float(np.percentile(dev, 75)),
            "q95": float(np.percentile(dev, 95)),
        }
    return result


def fitness_moments(
    df: pd.DataFrame,
    depths: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Mean and std of fitness distribution at each depth."""
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 3:
            continue
        result[d] = {
            "mean": float(sub["fitness"].mean()),
            "std": float(sub["fitness"].std()),
            "n": len(sub),
        }
    return result


def epistatic_deviation_distributions(
    df: pd.DataFrame,
    depths: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Raw epistatic deviation arrays at each depth (for distributional comparisons)."""
    if depths is None:
        depths = sorted(df["depth"].unique())

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) > 0:
            result[d] = sub["epistatic_deviation"].values
    return result


def pairwise_decomposition(
    df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
    pairwise_deviations: dict[tuple[int, int], dict[tuple[str, str], float]] | None = None,
    depths: list[int] | None = None,
) -> dict[int, float]:
    """Fraction of total epistasis variance explained by pairwise terms.

    If pairwise_deviations is None, estimates pairwise effects from the
    doubles in the dataset (depth=2 variants).

    Returns dict[depth, fraction_explained].
    """
    if depths is None:
        depths = [d for d in sorted(df["depth"].unique()) if d >= 3]

    # Estimate pairwise epistasis from doubles if not provided
    if pairwise_deviations is None:
        pairwise_deviations = _estimate_pairwise_from_doubles(df, single_site_effects)

    result = {}
    for d in depths:
        sub = df[df["depth"] == d]
        if len(sub) < 30:
            continue

        total_deviations = sub["epistatic_deviation"].values
        pairwise_predictions = np.zeros(len(sub))

        for i, (_, row) in enumerate(sub.iterrows()):
            mutations = row["mutations"]
            positions = sorted(mutations.keys())
            pw_sum = 0.0
            for j, p1 in enumerate(positions):
                for p2 in positions[j + 1 :]:
                    key = (min(p1, p2), max(p1, p2))
                    aa_key = (mutations[p1], mutations[p2])
                    pw_sum += pairwise_deviations.get(key, {}).get(aa_key, 0.0)
            pairwise_predictions[i] = pw_sum

        total_var = np.var(total_deviations)
        if total_var > 0:
            residual_var = np.var(total_deviations - pairwise_predictions)
            result[d] = 1.0 - residual_var / total_var
        else:
            result[d] = 1.0

    return result


def _estimate_pairwise_from_doubles(
    df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
) -> dict[tuple[int, int], dict[tuple[str, str], float]]:
    """Estimate pairwise epistatic deviations from double mutants."""
    doubles = df[df["depth"] == 2]
    pairwise: dict[tuple[int, int], dict[tuple[str, str], float]] = {}

    for _, row in doubles.iterrows():
        mutations = row["mutations"]
        positions = sorted(mutations.keys())
        if len(positions) != 2:
            continue

        p1, p2 = positions
        a1, a2 = mutations[p1], mutations[p2]
        key = (p1, p2)
        aa_key = (a1, a2)

        if key not in pairwise:
            pairwise[key] = {}
        pairwise[key][aa_key] = row["epistatic_deviation"]

    return pairwise


def roughness_slope_ratio(
    df: pd.DataFrame,
) -> float:
    """Roughness-to-slope ratio (Szendro et al. 2013, J Stat Mech).

    r/s = RMSD(residuals from the best-fit additive model) / mean(|additive predictions|).
    Single scalar; larger values mean a more rugged landscape relative to its
    additive slope. Collapses the depth-resolved profiles into one number that
    is directly comparable across datasets.
    """
    add_pred = df["additive_prediction"].to_numpy()
    fitness = df["fitness"].to_numpy()
    residuals = fitness - add_pred
    rmsd = float(np.sqrt(np.mean(residuals**2)))
    slope = float(np.mean(np.abs(add_pred)))
    if slope <= 0:
        return float("inf")
    return rmsd / slope


def fitness_distance_correlation(
    df: pd.DataFrame,
) -> float:
    """Pearson correlation between fitness and Hamming distance from WT (Jones & Forrest 1995).

    Distance from WT is taken as the variant's `depth` (number of mutations away).
    Negative values indicate a funnel-like landscape (closer to WT -> higher fitness).
    """
    if len(df) < 3:
        return float("nan")
    r, _ = sp_stats.pearsonr(df["depth"].to_numpy(), df["fitness"].to_numpy())
    return float(r)


def _neighbours_of(idx: tuple[int, ...], shape: tuple[int, ...]):
    """Single-mutation neighbour indices for a cell of a fitness tensor."""
    for d in range(len(idx)):
        for v in range(shape[d]):
            if v == idx[d]:
                continue
            new = list(idx)
            new[d] = v
            yield tuple(new)


def local_optima_from_tensor(
    tensor: NDArray[np.float64],
) -> dict[str, object]:
    """Count local optima of a complete fitness tensor.

    A local optimum is a cell whose fitness is strictly greater than all of
    its single-mutation (Hamming-1) neighbours. Returns a dict with:
        n_optima : number of local optima
        peak_indices : list of argmax indices, sorted by fitness descending
        peak_fitnesses : corresponding fitness values
        global_peak : index of the single global optimum
        basin_sizes : list of basin sizes (count of cells that reach each peak
            via greedy single-step ascent)
    """
    tensor = np.where(np.isnan(tensor), np.nanmin(tensor) - 1e6, tensor)
    shape = tensor.shape
    is_peak = np.zeros(shape, dtype=bool)
    for idx in np.ndindex(*shape):
        f = tensor[idx]
        if all(tensor[n] < f for n in _neighbours_of(idx, shape)):
            is_peak[idx] = True

    peak_coords = list(zip(*np.where(is_peak)))
    peak_fitnesses = [float(tensor[p]) for p in peak_coords]
    order = np.argsort(peak_fitnesses)[::-1]
    peak_coords = [peak_coords[i] for i in order]
    peak_fitnesses = [peak_fitnesses[i] for i in order]

    peak_id: dict[tuple[int, ...], int] = {p: i for i, p in enumerate(peak_coords)}

    def ascend(start: tuple[int, ...]) -> tuple[int, ...]:
        current = start
        while current not in peak_id:
            best = current
            best_f = tensor[current]
            for n in _neighbours_of(current, shape):
                if tensor[n] > best_f:
                    best_f = tensor[n]
                    best = n
            if best == current:
                break
            current = best
        return current

    basin_counts = [0] * len(peak_coords)
    for idx in np.ndindex(*shape):
        final = ascend(idx)
        if final in peak_id:
            basin_counts[peak_id[final]] += 1

    return {
        "n_optima": len(peak_coords),
        "peak_indices": peak_coords,
        "peak_fitnesses": peak_fitnesses,
        "global_peak": peak_coords[0] if peak_coords else None,
        "basin_sizes": basin_counts,
    }


def accessible_paths(
    tensor: NDArray[np.float64],
    peak_idx: tuple[int, ...] | None = None,
    n_sources: int | None = None,
    n_rsign_samples: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    """Fraction of monotonic single-step paths from each variant to the global peak.

    For each source variant at Hamming distance d from the peak, there are d!
    direct paths fixing one position at a time. A path is "accessible" if every
    step strictly increases fitness (Weinreich et al. 2006). The reported fraction
    is the mean across sampled sources of (accessible_paths / d!). Also returns
    the fraction of 2x2 sub-landscapes that show reciprocal sign epistasis, the
    predominant cause of inaccessible paths (Poelwijk 2011).
    """
    from itertools import combinations, permutations

    tensor = np.where(np.isnan(tensor), np.nanmin(tensor) - 1e6, tensor)
    shape = tensor.shape
    L = tensor.ndim
    rng = np.random.default_rng(seed)

    if peak_idx is None:
        peak_idx = np.unravel_index(int(np.argmax(tensor)), shape)

    all_sources = list(np.ndindex(*shape))
    if n_sources is not None and n_sources < len(all_sources):
        sample_idx = rng.choice(len(all_sources), size=n_sources, replace=False)
        sources = [all_sources[i] for i in sample_idx]
    else:
        sources = all_sources

    total_fraction = 0.0
    counted = 0
    for src in sources:
        differing = [d for d in range(L) if src[d] != peak_idx[d]]
        k = len(differing)
        if k == 0:
            continue
        n_accessible = 0
        n_paths = 1
        for i in range(1, k + 1):
            n_paths *= i
        for order in permutations(differing):
            cur = list(src)
            cur_f = tensor[tuple(cur)]
            ok = True
            for d in order:
                cur[d] = peak_idx[d]
                nxt_f = tensor[tuple(cur)]
                if nxt_f <= cur_f:
                    ok = False
                    break
                cur_f = nxt_f
            if ok:
                n_accessible += 1
        total_fraction += n_accessible / n_paths
        counted += 1
    mean_fraction = total_fraction / counted if counted else float("nan")

    # Reciprocal sign epistasis: sample random 2x2 sub-landscapes.
    if L >= 2:
        n_rsign = 0
        n_sampled = 0
        pair_choices = list(combinations(range(L), 2))
        for _ in range(n_rsign_samples):
            i, j = pair_choices[rng.integers(len(pair_choices))]
            others = [d for d in range(L) if d != i and d != j]
            bg = [int(rng.integers(shape[d])) for d in others]
            a1, b1 = rng.choice(shape[i], size=2, replace=False)
            a2, b2 = rng.choice(shape[j], size=2, replace=False)
            idx = [0] * L
            for d, v in zip(others, bg):
                idx[d] = v
            def cell(x1, x2, base=idx, i=i, j=j):
                base = list(base)
                base[i] = int(x1)
                base[j] = int(x2)
                return float(tensor[tuple(base)])
            f_aa = cell(a1, a2)
            f_ab = cell(a1, b2)
            f_ba = cell(b1, a2)
            f_bb = cell(b1, b2)
            e1 = (f_ba - f_aa) * (f_bb - f_ab)
            e2 = (f_ab - f_aa) * (f_bb - f_ba)
            if e1 < 0 and e2 < 0:
                n_rsign += 1
            n_sampled += 1
        reciprocal_sign_frac = n_rsign / n_sampled if n_sampled else 0.0
    else:
        reciprocal_sign_frac = 0.0

    return {
        "mean_accessible_fraction": float(mean_fraction),
        "reciprocal_sign_frac": float(reciprocal_sign_frac),
        "n_sources": counted,
        "peak_idx": tuple(int(x) for x in peak_idx),
    }


def fitness_distance_correlation_from_tensor(
    tensor: NDArray[np.float64],
    peak_idx: tuple[int, ...] | None = None,
) -> float:
    """Pearson correlation between fitness and Hamming distance from the global peak.

    Tensor variant of fitness_distance_correlation, for complete combinatorial
    spaces where every variant is present.
    """
    tensor = np.where(np.isnan(tensor), np.nanmean(tensor), tensor)
    shape = tensor.shape
    if peak_idx is None:
        peak_idx = np.unravel_index(int(np.argmax(tensor)), shape)

    distances = np.empty(shape, dtype=int)
    for idx in np.ndindex(*shape):
        distances[idx] = sum(1 for d, p in zip(idx, peak_idx) if d != p)

    flat_f = tensor.ravel()
    flat_d = distances.ravel()
    if np.std(flat_f) <= 0 or np.std(flat_d) <= 0:
        return float("nan")
    r, _ = sp_stats.pearsonr(flat_d, flat_f)
    return float(r)


def compute_all_metrics(
    df: pd.DataFrame,
    single_site_effects: dict[int, dict[str, float]],
    depths: list[int] | None = None,
) -> dict:
    """Compute the full suite of validation metrics."""
    if depths is None:
        depths = list(range(1, 8))

    return {
        "additive_correlation": additive_correlation_profile(df, depths),
        "sign_epistasis": sign_epistasis_profile(df, single_site_effects, depths),
        "deviation_stats": epistatic_deviation_stats(df, depths),
        "fitness_moments": fitness_moments(df, depths),
        "deviation_distributions": epistatic_deviation_distributions(df, depths),
    }
