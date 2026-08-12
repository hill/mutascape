"""Structure-conditioned metrics for sparse multi-mutant landscapes.

These metrics test whether epistasis is concentrated in variants whose
mutated positions are structurally close. They are designed for sparse
full-protein datasets where exact Walsh spectra and complete peak counts are
not available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from mutascape.contacts import ContactMap


@dataclass(frozen=True)
class ContactRegression:
    """Partial regression summary for contact-conditioned epistasis."""

    outcome: str
    contact_feature: str
    n: int
    beta: float
    partial_r2: float
    spearman_r: float
    spearman_p: float
    permutation_p: float | None


def add_contact_features(
    df: pd.DataFrame,
    contact_map: ContactMap,
    *,
    distance_scale: float = 8.0,
) -> pd.DataFrame:
    """Attach contact features to a DataFrame with a `mutations` column.

    The returned features are computed only from the positions mutated in each
    variant:

    - `contact_count`: number of mutated residue pairs within the contact map.
    - `contact_fraction`: contact_count divided by possible mutated pairs.
    - `contact_proximity`: mean exp(-distance / distance_scale) over mutated pairs.
    - `largest_contact_component_fraction`: size of the largest induced contact
      component divided by mutation depth.
    - `all_positions_resolved`: whether all mutated positions are in the structure.
    """
    pos_to_idx = {int(p): i for i, p in enumerate(contact_map.positions)}
    distances = contact_map.distances
    if distances is None:
        raise ValueError("contact_map.distances is required for structural metrics")

    rows: list[dict[str, float | bool]] = []
    for mutations in df["mutations"]:
        positions = sorted(int(p) for p in mutations)
        depth = len(positions)
        resolved = [p for p in positions if p in pos_to_idx]
        all_resolved = len(resolved) == depth

        if len(resolved) < 2:
            rows.append(
                {
                    "resolved_depth": float(len(resolved)),
                    "all_positions_resolved": all_resolved,
                    "possible_contact_pairs": 0.0,
                    "contact_count": 0.0,
                    "contact_fraction": np.nan,
                    "contact_proximity": np.nan,
                    "mean_pair_distance": np.nan,
                    "min_pair_distance": np.nan,
                    "largest_contact_component_fraction": np.nan,
                }
            )
            continue

        pair_distances = []
        contact_count = 0
        adjacency = {p: set() for p in resolved}
        for i, p_i in enumerate(resolved):
            idx_i = pos_to_idx[p_i]
            for p_j in resolved[i + 1 :]:
                idx_j = pos_to_idx[p_j]
                d = float(distances[idx_i, idx_j])
                pair_distances.append(d)
                if bool(contact_map.matrix[idx_i, idx_j]):
                    contact_count += 1
                    adjacency[p_i].add(p_j)
                    adjacency[p_j].add(p_i)

        possible = len(pair_distances)
        largest = _largest_component_size(adjacency)
        pair_distances_arr = np.array(pair_distances, dtype=np.float64)
        rows.append(
            {
                "resolved_depth": float(len(resolved)),
                "all_positions_resolved": all_resolved,
                "possible_contact_pairs": float(possible),
                "contact_count": float(contact_count),
                "contact_fraction": float(contact_count / possible) if possible else np.nan,
                "contact_proximity": float(np.mean(np.exp(-pair_distances_arr / distance_scale))),
                "mean_pair_distance": float(np.mean(pair_distances_arr)),
                "min_pair_distance": float(np.min(pair_distances_arr)),
                "largest_contact_component_fraction": float(largest / len(resolved)),
            }
        )

    features = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def contact_epistasis_regression(
    df: pd.DataFrame,
    *,
    outcome: str = "abs_epistatic_deviation",
    contact_feature: str = "contact_proximity",
    controls: tuple[str, ...] = ("depth", "abs_additive_prediction"),
    n_permutations: int = 0,
    seed: int = 42,
) -> ContactRegression:
    """Estimate contact-conditioned enrichment of epistasis.

    Fits two linear models after z-scoring all finite columns:

    - reduced: outcome ~ controls
    - full: outcome ~ controls + contact_feature

    The reported `beta` is the standardised contact coefficient in the full
    model. `partial_r2` is the fraction of reduced-model residual sum of
    squares explained by the contact feature. A positive beta means larger
    epistatic deviations are enriched in structurally close variants.
    """
    cols = [outcome, contact_feature, *controls]
    work = df.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < len(cols) + 3:
        return ContactRegression(
            outcome=outcome,
            contact_feature=contact_feature,
            n=len(work),
            beta=float("nan"),
            partial_r2=float("nan"),
            spearman_r=float("nan"),
            spearman_p=float("nan"),
            permutation_p=None,
        )

    y = _zscore(work[outcome].to_numpy(dtype=np.float64))
    contact = _zscore(work[contact_feature].to_numpy(dtype=np.float64))
    control_matrix = np.column_stack(
        [
            np.ones(len(work)),
            *(_zscore(work[c].to_numpy(dtype=np.float64)) for c in controls),
        ]
    )
    full_matrix = np.column_stack([control_matrix, contact])

    reduced_coef, *_ = np.linalg.lstsq(control_matrix, y, rcond=None)
    full_coef, *_ = np.linalg.lstsq(full_matrix, y, rcond=None)
    reduced_resid = y - control_matrix @ reduced_coef
    full_resid = y - full_matrix @ full_coef
    reduced_sse = float(np.sum(reduced_resid**2))
    full_sse = float(np.sum(full_resid**2))
    partial_r2 = (
        max(0.0, (reduced_sse - full_sse) / reduced_sse) if reduced_sse > 0 else float("nan")
    )
    beta = float(full_coef[-1])
    spearman_r, spearman_p = sp_stats.spearmanr(work[contact_feature], work[outcome])

    permutation_p = None
    if n_permutations > 0:
        permutation_p = _permutation_pvalue(
            y=y,
            contact=contact,
            controls=control_matrix,
            observed_beta=beta,
            depth=work["depth"].to_numpy() if "depth" in work else None,
            n_permutations=n_permutations,
            seed=seed,
        )

    return ContactRegression(
        outcome=outcome,
        contact_feature=contact_feature,
        n=len(work),
        beta=beta,
        partial_r2=partial_r2,
        spearman_r=float(spearman_r),
        spearman_p=float(spearman_p),
        permutation_p=permutation_p,
    )


def contact_bin_profile(
    df: pd.DataFrame,
    *,
    value_col: str = "abs_epistatic_deviation",
    contact_feature: str = "contact_proximity",
    n_bins: int = 4,
) -> pd.DataFrame:
    """Mean outcome in quantile bins of a contact feature."""
    work = df[[contact_feature, value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return pd.DataFrame(columns=["bin", "n", "contact_mean", "value_mean", "value_sem"])
    bins = pd.qcut(work[contact_feature], q=n_bins, labels=False, duplicates="drop")
    work = work.assign(bin=bins)
    grouped = work.groupby("bin", observed=True)
    return grouped.agg(
        n=(value_col, "size"),
        contact_mean=(contact_feature, "mean"),
        value_mean=(value_col, "mean"),
        value_sem=(value_col, _standard_error),
    ).reset_index()


def _largest_component_size(adjacency: dict[int, set[int]]) -> int:
    seen: set[int] = set()
    best = 0
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        best = max(best, size)
    return best


def _standard_error(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    std = float(np.std(values))
    if std == 0.0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _permutation_pvalue(
    *,
    y: np.ndarray,
    contact: np.ndarray,
    controls: np.ndarray,
    observed_beta: float,
    depth: np.ndarray | None,
    n_permutations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    extreme = 0
    if depth is None:
        groups = [np.arange(len(contact))]
    else:
        groups = [np.where(depth == d)[0] for d in sorted(set(depth))]

    for _ in range(n_permutations):
        shuffled = contact.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(shuffled[idx])
        full = np.column_stack([controls, shuffled])
        coef, *_ = np.linalg.lstsq(full, y, rcond=None)
        if abs(float(coef[-1])) >= abs(observed_beta):
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))
