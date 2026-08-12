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
        peak_fraction : n_optima / number of finite tensor cells
        peak_indices : list of argmax indices, sorted by fitness descending
        peak_fitnesses : corresponding fitness values
        global_peak : index of the single global optimum
        basin_sizes : list of basin sizes (count of cells that reach each peak
            via greedy single-step ascent)
        largest_basin_fraction : fraction of cells flowing to the largest
            greedy-ascent basin
        basin_entropy : Shannon entropy of greedy-ascent basin probabilities,
            normalised to [0, 1] when at least two basins exist

    Raw `n_optima` is meaningful on a fixed complete factorial space. For
    comparisons across proteins or alphabet sizes, use `peak_fraction` and the
    basin summaries instead.
    """
    finite_mask = np.isfinite(tensor)
    n_finite = int(finite_mask.sum())
    if n_finite == 0:
        return {
            "n_optima": 0,
            "peak_fraction": float("nan"),
            "peak_indices": [],
            "peak_fitnesses": [],
            "global_peak": None,
            "basin_sizes": [],
            "largest_basin_fraction": float("nan"),
            "basin_entropy": float("nan"),
            "n_variants": 0,
        }

    tensor = np.where(finite_mask, tensor, float(np.nanmin(tensor) - 1e6))
    shape = tensor.shape
    is_peak = np.zeros(shape, dtype=bool)
    for idx in np.ndindex(*shape):
        if not finite_mask[idx]:
            continue
        f = tensor[idx]
        if all((not finite_mask[n]) or tensor[n] < f for n in _neighbours_of(idx, shape)):
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
        if not finite_mask[idx]:
            continue
        final = ascend(idx)
        if final in peak_id:
            basin_counts[peak_id[final]] += 1

    if basin_counts and n_finite:
        basin_counts_arr = np.array(basin_counts, dtype=np.float64)
        largest_basin_fraction = float(basin_counts_arr.max() / n_finite)
        assigned_total = float(basin_counts_arr.sum())
        basin_probs = basin_counts_arr / assigned_total if assigned_total > 0 else basin_counts_arr
        nonzero = basin_probs[basin_probs > 0]
        raw_entropy = float(-(nonzero * np.log(nonzero)).sum())
        basin_entropy = raw_entropy / float(np.log(len(nonzero))) if len(nonzero) > 1 else 0.0
    else:
        largest_basin_fraction = float("nan")
        basin_entropy = float("nan")

    return {
        "n_optima": len(peak_coords),
        "peak_fraction": len(peak_coords) / n_finite,
        "peak_indices": peak_coords,
        "peak_fitnesses": peak_fitnesses,
        "global_peak": peak_coords[0] if peak_coords else None,
        "basin_sizes": basin_counts,
        "largest_basin_fraction": largest_basin_fraction,
        "basin_entropy": basin_entropy,
        "n_variants": n_finite,
    }


def sampled_neighbourhood_ruggedness(
    tensor: NDArray[np.float64],
    n_samples: int | None = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate local ruggedness from sampled Hamming-1 neighbourhoods.

    This is a topology-aware metric that still works when exact peak counting
    is too expensive. It samples tensor cells and evaluates all of their
    single-mutation neighbours, returning:

    - `local_optimum_rate`: fraction of sampled cells that are strict local
      maxima within their full Hamming-1 neighbourhood.
    - `mean_uphill_fraction`: mean fraction of neighbours fitter than the
      source. Lower values mean stronger local trapping.
    - `mean_best_step`: mean fitness gain of the best available single-step
      move, clipped at zero.
    """
    finite = np.argwhere(np.isfinite(tensor))
    if finite.size == 0:
        return {
            "local_optimum_rate": float("nan"),
            "mean_uphill_fraction": float("nan"),
            "mean_best_step": float("nan"),
            "n_sampled": 0.0,
        }

    rng = np.random.default_rng(seed)
    if n_samples is None or n_samples >= len(finite):
        sampled = finite
    else:
        sampled = finite[rng.choice(len(finite), size=n_samples, replace=False)]

    n_local = 0
    uphill_fracs = []
    best_steps = []
    shape = tensor.shape
    for raw_idx in sampled:
        idx = tuple(int(x) for x in raw_idx)
        f = float(tensor[idx])
        neighbours = [
            float(tensor[n]) for n in _neighbours_of(idx, shape) if np.isfinite(tensor[n])
        ]
        if not neighbours:
            continue
        gains = np.array(neighbours, dtype=np.float64) - f
        uphill = gains > 0
        if not np.any(uphill):
            n_local += 1
        uphill_fracs.append(float(uphill.mean()))
        best_steps.append(float(max(0.0, gains.max())))

    n_counted = len(uphill_fracs)
    return {
        "local_optimum_rate": n_local / n_counted if n_counted else float("nan"),
        "mean_uphill_fraction": float(np.mean(uphill_fracs)) if n_counted else float("nan"),
        "mean_best_step": float(np.mean(best_steps)) if n_counted else float("nan"),
        "n_sampled": float(n_counted),
    }


def sampled_landscape_neighbourhoods(
    landscape,
    n_samples: int = 256,
    n_mutations: int | tuple[int, int] = (1, 5),
    max_neighbours: int | None = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate optima and uphill-move structure on a full mutascape oracle.

    Exact `N_optima` requires enumerating the full Cartesian product and is only
    feasible for small complete factorial libraries. This function samples
    source variants from a live landscape, evaluates their Hamming-1
    neighbourhoods, and reports scale-free topology diagnostics:

    - `local_optimum_rate`: sampled analogue of `peak_fraction`.
    - `mean_uphill_fraction`: how much of the local neighbourhood is uphill.
    - `mean_best_step`: average best single-step improvement.
    - `mean_neighbour_count`: average number of neighbours considered.

    When a neighbourhood is too large, neighbours are subsampled uniformly.
    """
    rng = np.random.default_rng(seed)
    variants = [
        dict(v.mutations)
        for v in landscape.generate(n_mutations=n_mutations, n=n_samples, seed=seed)
    ]
    positions = tuple(int(p) for p in landscape.positions)

    n_local = 0
    uphill_fracs = []
    best_steps = []
    neighbour_counts = []

    for mutations in variants:
        fitness = float(landscape.evaluate(mutations))
        neighbours = _landscape_neighbours(landscape, mutations, positions)
        if max_neighbours is not None and len(neighbours) > max_neighbours:
            pick = rng.choice(len(neighbours), size=max_neighbours, replace=False)
            neighbours = [neighbours[int(i)] for i in pick]
        if not neighbours:
            continue

        neighbour_fitness = landscape.evaluate_batch(neighbours)
        gains = neighbour_fitness - fitness
        uphill = gains > 0
        if not np.any(uphill):
            n_local += 1
        uphill_fracs.append(float(uphill.mean()))
        best_steps.append(float(max(0.0, gains.max())))
        neighbour_counts.append(float(len(neighbours)))

    n_counted = len(uphill_fracs)
    return {
        "local_optimum_rate": n_local / n_counted if n_counted else float("nan"),
        "mean_uphill_fraction": float(np.mean(uphill_fracs)) if n_counted else float("nan"),
        "mean_best_step": float(np.mean(best_steps)) if n_counted else float("nan"),
        "mean_neighbour_count": (float(np.mean(neighbour_counts)) if n_counted else float("nan")),
        "n_sampled": float(n_counted),
    }


def _landscape_neighbours(landscape, mutations: dict[int, str], positions: tuple[int, ...]):
    """Construct Hamming-1 neighbours for a mutascape genotype."""
    neighbours = []
    mutated = set(mutations)

    for pos in positions:
        wt = landscape.wild_type[pos]
        allowed = sorted(set(landscape.single_site_effects.get(pos, {})) | {wt})
        if pos in mutated:
            # Revert this mutation to WT.
            reverted = dict(mutations)
            del reverted[pos]
            neighbours.append(reverted)

            # Change this residue to another non-WT amino acid at the same site.
            for aa in allowed:
                if aa != wt and aa != mutations[pos]:
                    changed = dict(mutations)
                    changed[pos] = aa
                    neighbours.append(changed)
        else:
            # Add one mutation at an unmutated site.
            for aa in allowed:
                if aa != wt:
                    added = dict(mutations)
                    added[pos] = aa
                    neighbours.append(added)

    return neighbours


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


def ssmula_attributes(
    tensor: NDArray[np.float64],
    active_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute the SSMuLA navigability attributes for a fitness tensor.

    Li et al. (Cell Systems 2025) identify six landscape attributes that
    predict MLDE benefit. This returns a subset directly computable from a
    fitness tensor:

    - `frac_active`: fraction of variants with fitness > `active_threshold`
      (defaults to 0 = WT fitness). Proxies SSMuLA's activity density.
    - `frac_local_optima`: fraction of variants that are strict local maxima
      on the Hamming-1 neighbourhood graph.
    - `kurtosis`: Fisher kurtosis of the fitness distribution (peaked vs
      flat-tailed). Positive = heavy tails.
    - `kde_peak_count`: number of local maxima in a Gaussian KDE of the
      fitness distribution, found via `scipy.signal.find_peaks`. Proxies
      multi-modality of the fitness distribution.

    Pass `active_threshold` explicitly for non-WT-referenced landscapes.
    """
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde

    finite = tensor[np.isfinite(tensor)].ravel()
    if finite.size == 0:
        return {
            "frac_active": float("nan"),
            "frac_local_optima": float("nan"),
            "kurtosis": float("nan"),
            "kde_peak_count": float("nan"),
        }

    frac_active = float((finite > active_threshold).mean())

    optima = local_optima_from_tensor(tensor)
    frac_local_optima = float(optima["peak_fraction"])

    kurt = float(sp_stats.kurtosis(finite, fisher=True, bias=False))

    if finite.size < 3 or np.std(finite) == 0:
        kde_peaks = 0
    else:
        kde = gaussian_kde(finite)
        grid = np.linspace(finite.min(), finite.max(), 512)
        density = kde(grid)
        peaks, _ = find_peaks(density)
        kde_peaks = int(len(peaks))

    return {
        "frac_active": frac_active,
        "frac_local_optima": frac_local_optima,
        "kurtosis": kurt,
        "kde_peak_count": float(kde_peaks),
    }


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
