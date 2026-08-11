"""Evaluate whether structural priors help on sparse full-protein variants.

This script tests the claim that structural priors matter for larger proteins
where most residue pairs should not interact directly. It uses GFP sparse
multi-mutants as the empirical target, then evaluates mutascape variants on the
same genotypes under several ablations:

- no pairwise contacts, both reference-noise and marginally calibrated
- true 8A structural contacts
- shuffled contacts with the same sparsity
- dense all-to-all pairwise interactions
- a distance-decay structural prior

Outputs:
  notebooks/validation/output_structural/gfp_structural_prior_ablation_summary.csv
  notebooks/validation/output_structural/gfp_structural_prior_ablation_bins.csv
  notebooks/validation/output_structural/gfp_structural_prior_ablation.{png,pdf}
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    AA_INDEX,
    AMINO_ACIDS,
    CouplingTensor,
    RandomEpistasis,
    SignConsistentEpistasis,
)
from mutascape.landscape import Landscape
from mutascape.nonlinearity import GlobalEpistasisSigmoid
from mutascape.validation.real_data import (
    load_proteingym_multimutant,
    load_single_site_effects,
)
from mutascape.validation.structural_metrics import (
    add_contact_features,
    contact_bin_profile,
    contact_epistasis_regression,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data/gfp"
OUTPUT_DIR = REPO_ROOT / "notebooks/validation/output_structural"
PROTEINGYM_CSV = (
    REPO_ROOT.parent
    / "mutantopt/experiments/datasets/proteingym/GFP_AEQVI_Sarkisyan_2016.csv"
)
DMS_CSV = DATA_DIR / "gfp.csv"
DISTANCES_NPZ = DATA_DIR / "gfp_distances.npz"
WT_FASTA = DATA_DIR / "gfp_wt.fasta"

DEPTHS = tuple(range(2, 8))
WT_FITNESS = 3.72
SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    """One structural-prior ablation to evaluate."""

    key: str
    label: str
    kind: str
    sigma: float
    contact_source: str
    alpha: float = 1.5
    noise_sigma: float = 0.3
    distance_scale: float = 8.0
    max_distance: float = 20.0


MODEL_SPECS = (
    ModelSpec(
        key="real",
        label="Empirical GFP",
        kind="real",
        sigma=0.0,
        contact_source="real",
    ),
    ModelSpec(
        key="no_pairwise_reference",
        label="No pairwise, noise 0.30",
        kind="random",
        sigma=0.0,
        contact_source="real",
    ),
    ModelSpec(
        key="no_pairwise_calibrated",
        label="No pairwise, noise 0.45",
        kind="random",
        sigma=0.0,
        contact_source="real",
        noise_sigma=0.45,
    ),
    ModelSpec(
        key="true_contacts_random_0p20_n0p45",
        label="True contacts, random 0.20",
        kind="random",
        sigma=0.20,
        contact_source="real",
        noise_sigma=0.45,
    ),
    ModelSpec(
        key="true_contacts_random_0p30_n0p45",
        label="True contacts, random 0.30",
        kind="random",
        sigma=0.30,
        contact_source="real",
        noise_sigma=0.45,
    ),
    ModelSpec(
        key="shuffled_random_0p30_n0p45",
        label="Shuffled contacts, random 0.30",
        kind="random",
        sigma=0.30,
        contact_source="shuffled",
        noise_sigma=0.45,
    ),
    ModelSpec(
        key="dense_random_0p05_n0p45",
        label="All pairs, random 0.05",
        kind="random",
        sigma=0.05,
        contact_source="dense",
        noise_sigma=0.45,
    ),
    ModelSpec(
        key="distance_decay_random_0p10_n0p45",
        label="Distance decay, random 0.10",
        kind="distance_decay_random",
        sigma=0.10,
        contact_source="real",
        noise_sigma=0.45,
        distance_scale=6.0,
        max_distance=20.0,
    ),
    ModelSpec(
        key="distance_decay_sign_0p20",
        label="Distance decay, sign 0.20",
        kind="distance_decay_sign",
        sigma=0.20,
        contact_source="real",
        distance_scale=6.0,
        max_distance=20.0,
    ),
)

PLOT_KEYS = (
    "real",
    "no_pairwise_reference",
    "no_pairwise_calibrated",
    "true_contacts_random_0p20_n0p45",
    "shuffled_random_0p30_n0p45",
    "dense_random_0p05_n0p45",
    "distance_decay_random_0p10_n0p45",
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    effects = load_single_site_effects(DMS_CSV)
    real_contact_map = _load_contact_map(DISTANCES_NPZ, threshold=8.0)
    contact_maps = {
        "real": real_contact_map,
        "shuffled": _shuffled_contact_map(real_contact_map, seed=SEED),
        "dense": _dense_contact_map(real_contact_map),
    }

    real_df = load_proteingym_multimutant(
        PROTEINGYM_CSV,
        DMS_CSV,
        wt_fitness=WT_FITNESS,
        min_depth=min(DEPTHS),
        max_depth=max(DEPTHS),
    )
    real_df = _prepare_observations(real_df, real_contact_map)
    print(f"Retained {len(real_df)} structurally resolved GFP variants")

    rows = []
    bin_rows = []
    frames = {"real": real_df.assign(model_key="real", model_label="GFP empirical target")}

    real_reg = contact_epistasis_regression(
        real_df,
        contact_feature="contact_proximity",
        n_permutations=999,
        seed=SEED,
    )
    real_curve = contact_bin_profile(real_df, n_bins=5)
    real_residual_curve = contact_bin_profile(
        real_df,
        value_col="abs_epi_control_residual",
        n_bins=5,
    )

    rows.append(
        _summary_row(
            spec=MODEL_SPECS[0],
            df=real_df,
            real_df=real_df,
            regression=real_reg,
            real_regression=real_reg,
            real_curve=real_curve,
            real_residual_curve=real_residual_curve,
        )
    )
    bin_rows.extend(_bin_rows(MODEL_SPECS[0], real_curve, metric="absolute_deviation"))
    bin_rows.extend(
        _bin_rows(
            MODEL_SPECS[0],
            real_residual_curve,
            metric="control_residual",
        )
    )

    for spec in MODEL_SPECS[1:]:
        print(f"Evaluating {spec.label}")
        landscape = _build_landscape(
            contact_map=contact_maps[spec.contact_source],
            effects=effects,
            spec=spec,
        )
        synth_df = _evaluate_same_variants(landscape, real_df)
        synth_df = _prepare_observations(synth_df, real_contact_map)
        frames[spec.key] = synth_df.assign(model_key=spec.key, model_label=spec.label)

        reg = contact_epistasis_regression(
            synth_df,
            contact_feature="contact_proximity",
            n_permutations=199,
            seed=SEED,
        )
        curve = contact_bin_profile(synth_df, n_bins=5)

        rows.append(
            _summary_row(
                spec=spec,
                df=synth_df,
                real_df=real_df,
                regression=reg,
                real_regression=real_reg,
                real_curve=real_curve,
                real_residual_curve=real_residual_curve,
            )
        )
        residual_curve = contact_bin_profile(
            synth_df,
            value_col="abs_epi_control_residual",
            n_bins=5,
        )
        bin_rows.extend(_bin_rows(spec, curve, metric="absolute_deviation"))
        bin_rows.extend(_bin_rows(spec, residual_curve, metric="control_residual"))

    summary = pd.DataFrame(rows)
    bins = pd.DataFrame(bin_rows)
    summary = _add_composite_scores(summary)

    summary_path = OUTPUT_DIR / "gfp_structural_prior_ablation_summary.csv"
    bins_path = OUTPUT_DIR / "gfp_structural_prior_ablation_bins.csv"
    summary.to_csv(summary_path, index=False)
    bins.to_csv(bins_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {bins_path}")

    _plot_ablation(
        summary=summary,
        bins=bins,
        frames=frames,
        real_regression=real_reg,
        out_base=OUTPUT_DIR / "gfp_structural_prior_ablation",
    )

    display_cols = [
        "model_key",
        "beta",
        "partial_r2",
        "permutation_p",
        "mean_abs_deviation",
        "deviation_w1",
        "residual_curve_rmse",
        "beta_error",
        "contact_score",
        "composite_score",
    ]
    print(summary[display_cols].round(4).to_string(index=False))


def _load_contact_map(path: Path, *, threshold: float) -> ContactMap:
    data = np.load(path)
    return ContactMap.from_distance_matrix(
        data["distances"],
        data["positions"].tolist(),
        threshold=threshold,
    )


def _shuffled_contact_map(contact_map: ContactMap, *, seed: int) -> ContactMap:
    rng = np.random.default_rng(seed)
    n = contact_map.n_positions
    possible = np.array(list(combinations(range(n), 2)))
    chosen = rng.choice(len(possible), size=contact_map.n_contacts, replace=False)
    matrix = np.zeros((n, n), dtype=bool)
    for i, j in possible[chosen]:
        matrix[i, j] = True
        matrix[j, i] = True
    return ContactMap(
        matrix=matrix,
        positions=contact_map.positions,
        threshold=contact_map.threshold,
        distances=contact_map.distances,
    )


def _dense_contact_map(contact_map: ContactMap) -> ContactMap:
    n = contact_map.n_positions
    matrix = np.ones((n, n), dtype=bool)
    np.fill_diagonal(matrix, False)
    return ContactMap(
        matrix=matrix,
        positions=contact_map.positions,
        threshold=contact_map.threshold,
        distances=contact_map.distances,
    )


def _build_landscape(
    *,
    contact_map: ContactMap,
    effects: dict[int, dict[str, float]],
    spec: ModelSpec,
) -> Landscape:
    if spec.kind == "sign":
        epistasis = SignConsistentEpistasis(sigma=spec.sigma, seed=SEED)
        couplings = epistasis.generate(contact_map, effects)
    elif spec.kind == "random":
        epistasis = RandomEpistasis(sigma=spec.sigma, seed=SEED)
        couplings = epistasis.generate(contact_map, effects)
    elif spec.kind == "distance_decay_sign":
        couplings = _distance_decay_sign_couplings(
            contact_map=contact_map,
            effects=effects,
            sigma=spec.sigma,
            seed=SEED,
            distance_scale=spec.distance_scale,
            max_distance=spec.max_distance,
        )
    elif spec.kind == "distance_decay_random":
        couplings = _distance_decay_random_couplings(
            contact_map=contact_map,
            sigma=spec.sigma,
            seed=SEED,
            distance_scale=spec.distance_scale,
            max_distance=spec.max_distance,
        )
    else:
        raise ValueError(f"Unknown model kind {spec.kind!r}")

    fitted = GlobalEpistasisSigmoid.fit_from_singles(effects, combinatorial_upside=3.0)
    singles_min = min(effect for pos in effects.values() for effect in pos.values())
    nonlinearity = GlobalEpistasisSigmoid(
        floor=fitted.floor,
        ceiling=fitted.ceiling,
        scale=fitted.scale,
        floor_hard=singles_min,
    )

    wt_sequence = str(next(SeqIO.parse(str(WT_FASTA), "fasta")).seq)
    wild_type = {pos: wt_sequence[pos - 1] for pos in effects}
    return Landscape(
        wild_type=wild_type,
        single_site_effects=effects,
        contact_map=contact_map,
        couplings=couplings,
        alpha=spec.alpha,
        noise_sigma=spec.noise_sigma,
        seed=SEED,
        wt_sequence=wt_sequence,
        nonlinearity=nonlinearity,
    )


def _distance_decay_sign_couplings(
    *,
    contact_map: ContactMap,
    effects: dict[int, dict[str, float]],
    sigma: float,
    seed: int,
    distance_scale: float,
    max_distance: float,
) -> CouplingTensor:
    if contact_map.distances is None:
        raise ValueError("distance-decay couplings require distances")

    rng = np.random.default_rng(seed)
    couplings: dict[tuple[int, int], NDArray[np.float64]] = {}
    n_aa = len(AMINO_ACIDS)
    distances = contact_map.distances

    for idx_i, pos_i in enumerate(contact_map.positions):
        for idx_j, pos_j in enumerate(contact_map.positions[idx_i + 1 :], start=idx_i + 1):
            d = float(distances[idx_i, idx_j])
            if d > max_distance:
                continue
            weight = 1.0 if d <= contact_map.threshold else np.exp(
                -(d - contact_map.threshold) / distance_scale
            )
            matrix = np.zeros((n_aa, n_aa), dtype=np.float64)
            effects_i = effects.get(pos_i, {})
            effects_j = effects.get(pos_j, {})
            for a_idx, aa_a in enumerate(AMINO_ACIDS):
                h_a = effects_i.get(aa_a, 0.0)
                for b_idx, aa_b in enumerate(AMINO_ACIDS):
                    h_b = effects_j.get(aa_b, 0.0)
                    if h_a == 0.0 or h_b == 0.0:
                        continue
                    sign = np.sign(h_a + h_b)
                    matrix[a_idx, b_idx] = sign * abs(rng.normal(0.0, sigma * weight))
            couplings[(int(pos_i), int(pos_j))] = matrix

    return CouplingTensor(
        couplings=couplings,
        positions=contact_map.positions,
        aa_index=AA_INDEX,
    )


def _distance_decay_random_couplings(
    *,
    contact_map: ContactMap,
    sigma: float,
    seed: int,
    distance_scale: float,
    max_distance: float,
) -> CouplingTensor:
    if contact_map.distances is None:
        raise ValueError("distance-decay couplings require distances")

    rng = np.random.default_rng(seed)
    couplings: dict[tuple[int, int], NDArray[np.float64]] = {}
    n_aa = len(AMINO_ACIDS)
    distances = contact_map.distances

    for idx_i, pos_i in enumerate(contact_map.positions):
        for idx_j, pos_j in enumerate(contact_map.positions[idx_i + 1 :], start=idx_i + 1):
            d = float(distances[idx_i, idx_j])
            if d > max_distance:
                continue
            weight = 1.0 if d <= contact_map.threshold else np.exp(
                -(d - contact_map.threshold) / distance_scale
            )
            couplings[(int(pos_i), int(pos_j))] = rng.normal(
                0.0,
                sigma * weight,
                size=(n_aa, n_aa),
            )

    return CouplingTensor(
        couplings=couplings,
        positions=contact_map.positions,
        aa_index=AA_INDEX,
    )


def _prepare_observations(df: pd.DataFrame, contact_map: ContactMap) -> pd.DataFrame:
    out = df.copy()
    out["abs_epistatic_deviation"] = out["epistatic_deviation"].abs()
    out["abs_additive_prediction"] = out["additive_prediction"].abs()
    out = add_contact_features(out, contact_map)
    out = out[out["all_positions_resolved"]].reset_index(drop=True)
    return _add_control_residual(out)


def _add_control_residual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    y = out["abs_epistatic_deviation"].to_numpy(dtype=np.float64)
    controls = np.column_stack([
        np.ones(len(out)),
        out["depth"].to_numpy(dtype=np.float64),
        out["abs_additive_prediction"].to_numpy(dtype=np.float64),
    ])
    coef, *_ = np.linalg.lstsq(controls, y, rcond=None)
    out["abs_epi_control_residual"] = y - controls @ coef
    return out


def _evaluate_same_variants(landscape: Landscape, real_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mutations in real_df["mutations"]:
        variant = {int(pos): aa for pos, aa in mutations.items()}
        fitness = _fast_pairwise_fitness(landscape, variant)
        additive = landscape.additive_fitness(variant)
        rows.append({
            "mutations": variant,
            "fitness": fitness,
            "additive_prediction": additive,
            "depth": len(variant),
            "epistatic_deviation": fitness - additive,
        })
    return pd.DataFrame(rows)


def _fast_pairwise_fitness(landscape: Landscape, mutations: dict[int, str]) -> float:
    additive_preimage = 0.0
    for pos, aa in mutations.items():
        additive_preimage += landscape._preimage_effects.get(pos, {}).get(aa, 0.0)

    epistatic = 0.0
    wild_type = landscape.wild_type
    couplings = landscape.couplings
    for pos_i, pos_j in combinations(sorted(mutations), 2):
        aa_i = mutations[pos_i]
        aa_j = mutations[pos_j]
        wt_i = wild_type[pos_i]
        wt_j = wild_type[pos_j]
        coupling = couplings.get_coupling(pos_i, pos_j, aa_i, aa_j)
        coupling -= couplings.get_coupling(pos_i, pos_j, wt_i, aa_j)
        coupling -= couplings.get_coupling(pos_i, pos_j, aa_i, wt_j)
        coupling += couplings.get_coupling(pos_i, pos_j, wt_i, wt_j)
        epistatic += coupling

    raw = additive_preimage + landscape.alpha * epistatic
    if landscape.noise_sigma > 0:
        raw += landscape._noise_for_mutations(mutations)

    fitness = float(landscape.nonlinearity.forward(raw))
    if landscape.readout_texture_sigma > 0 and landscape._readout_texture_applies(fitness):
        fitness += landscape.readout_texture_sigma * landscape._readout_texture_value(mutations)
    return float(fitness)


def _summary_row(
    *,
    spec: ModelSpec,
    df: pd.DataFrame,
    real_df: pd.DataFrame,
    regression,
    real_regression,
    real_curve: pd.DataFrame,
    real_residual_curve: pd.DataFrame,
) -> dict[str, float | int | str | None]:
    curve = contact_bin_profile(df, n_bins=5)
    residual_curve = contact_bin_profile(
        df,
        value_col="abs_epi_control_residual",
        n_bins=5,
    )
    return {
        "model_key": spec.key,
        "model_label": spec.label,
        "kind": spec.kind,
        "contact_source": spec.contact_source,
        "sigma": spec.sigma,
        "alpha": spec.alpha,
        "noise_sigma": spec.noise_sigma,
        "n": len(df),
        "contact_density": float(df["contact_count"].sum() / df["possible_contact_pairs"].sum()),
        "beta": regression.beta,
        "partial_r2": regression.partial_r2,
        "spearman_r": regression.spearman_r,
        "spearman_p": regression.spearman_p,
        "permutation_p": regression.permutation_p,
        "mean_abs_deviation": float(df["abs_epistatic_deviation"].mean()),
        "real_mean_abs_deviation": float(real_df["abs_epistatic_deviation"].mean()),
        "mean_abs_deviation_error": abs(
            float(df["abs_epistatic_deviation"].mean())
            - float(real_df["abs_epistatic_deviation"].mean())
        ),
        "deviation_w1": float(wasserstein_distance(
            real_df["abs_epistatic_deviation"],
            df["abs_epistatic_deviation"],
        )),
        "signed_deviation_w1": float(wasserstein_distance(
            real_df["epistatic_deviation"],
            df["epistatic_deviation"],
        )),
        "beta_error": abs(regression.beta - real_regression.beta),
        "partial_r2_error": abs(regression.partial_r2 - real_regression.partial_r2),
        "curve_rmse": _curve_rmse(curve, real_curve),
        "residual_curve_rmse": _curve_rmse(residual_curve, real_residual_curve),
        "residual_contact_contrast": _contact_contrast(residual_curve),
        "residual_contact_contrast_error": abs(
            _contact_contrast(residual_curve) - _contact_contrast(real_residual_curve)
        ),
    }


def _curve_rmse(curve: pd.DataFrame, real_curve: pd.DataFrame) -> float:
    merged = real_curve[["bin", "value_mean"]].merge(
        curve[["bin", "value_mean"]],
        on="bin",
        suffixes=("_real", "_model"),
    )
    if merged.empty:
        return float("nan")
    diff = merged["value_mean_model"] - merged["value_mean_real"]
    return float(np.sqrt(np.mean(np.square(diff))))


def _contact_contrast(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    ordered = curve.sort_values("bin")
    return float(ordered["value_mean"].iloc[-1] - ordered["value_mean"].iloc[0])


def _bin_rows(
    spec: ModelSpec,
    curve: pd.DataFrame,
    *,
    metric: str,
) -> list[dict[str, float | int | str]]:
    rows = []
    for _, row in curve.iterrows():
        rows.append({
            "model_key": spec.key,
            "model_label": spec.label,
            "metric": metric,
            "bin": int(row["bin"]),
            "n": int(row["n"]),
            "contact_mean": float(row["contact_mean"]),
            "value_mean": float(row["value_mean"]),
            "value_sem": float(row["value_sem"]),
        })
    return rows


def _add_composite_scores(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    components = [
        "deviation_w1",
        "signed_deviation_w1",
        "residual_curve_rmse",
        "beta_error",
        "mean_abs_deviation_error",
    ]
    for col in components:
        scale = out.loc[out["model_key"] != "real", col].replace(0, np.nan).median()
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out[f"{col}_scaled"] = out[col] / scale
    out["composite_score"] = out[[f"{col}_scaled" for col in components]].sum(axis=1)
    out.loc[out["model_key"] == "real", "composite_score"] = 0.0

    contact_components = [
        "beta_error",
        "partial_r2_error",
        "residual_curve_rmse",
        "residual_contact_contrast_error",
    ]
    for col in contact_components:
        scale = out.loc[out["model_key"] != "real", col].replace(0, np.nan).median()
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out[f"{col}_contact_scaled"] = out[col] / scale
    out["contact_score"] = out[
        [f"{col}_contact_scaled" for col in contact_components]
    ].sum(axis=1)
    out.loc[out["model_key"] == "real", "contact_score"] = 0.0
    return out


def _plot_ablation(
    *,
    summary: pd.DataFrame,
    bins: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    real_regression,
    out_base: Path,
) -> None:
    plot_summary = summary[summary["model_key"].isin(PLOT_KEYS)].copy()
    plot_summary["model_label"] = pd.Categorical(
        plot_summary["model_label"],
        categories=[
            spec.label for spec in MODEL_SPECS if spec.key in PLOT_KEYS
        ],
        ordered=True,
    )
    plot_summary = plot_summary.sort_values("model_label")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_beta, ax_curve, ax_score, ax_dist = axes.ravel()

    colors = _palette()
    beta_colors = [colors[k] for k in plot_summary["model_key"]]
    ax_beta.barh(plot_summary["model_label"], plot_summary["beta"], color=beta_colors)
    ax_beta.axvline(real_regression.beta, color="black", linestyle=":", linewidth=2)
    ax_beta.set_xlabel("Standardised contact coefficient")
    ax_beta.set_title("Contact-conditioned epistasis")

    plot_bins = bins[
        (bins["model_key"].isin(PLOT_KEYS)) & (bins["metric"] == "control_residual")
    ].copy()
    for key in PLOT_KEYS:
        part = plot_bins[plot_bins["model_key"] == key].sort_values("bin")
        if part.empty:
            continue
        label = part["model_label"].iloc[0]
        ax_curve.errorbar(
            part["bin"],
            part["value_mean"],
            yerr=part["value_sem"],
            marker="o",
            linewidth=2,
            capsize=2,
            label=label,
            color=colors[key],
        )
    ax_curve.set_xlabel("Contact-proximity quantile")
    ax_curve.set_ylabel("Mean residual absolute epistasis")
    ax_curve.set_title("Contact response after controls")
    ax_curve.legend(fontsize=8, frameon=False)

    score_data = plot_summary[plot_summary["model_key"] != "real"]
    ax_score.barh(score_data["model_label"], score_data["contact_score"], color=[
        colors[k] for k in score_data["model_key"]
    ])
    ax_score.set_xlabel("Contact-specific distance to GFP")
    ax_score.set_title("Structural ablation score, lower is better")

    dist_keys = (
        "real",
        "no_pairwise_calibrated",
        "true_contacts_random_0p20_n0p45",
        "shuffled_random_0p30_n0p45",
    )
    bins_grid = np.linspace(0, 2.5, 80)
    for key in dist_keys:
        values = frames[key]["abs_epistatic_deviation"].clip(upper=2.5)
        ax_dist.hist(
            values,
            bins=bins_grid,
            density=True,
            histtype="step",
            linewidth=2,
            color=colors[key],
            label=frames[key]["model_label"].iloc[0],
        )
    ax_dist.set_xlabel("Absolute epistatic deviation")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Marginal epistasis scale")
    ax_dist.legend(fontsize=8, frameon=False)

    fig.suptitle("GFP full-protein structural-prior ablation", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {out_base.with_suffix('.png')}")
    print(f"Wrote {out_base.with_suffix('.pdf')}")


def _palette() -> dict[str, str]:
    return {
        "real": "#111111",
        "no_pairwise_reference": "#9CA3AF",
        "no_pairwise_calibrated": "#4B5563",
        "true_contacts_random_0p20_n0p45": "#2563EB",
        "true_contacts_random_0p30_n0p45": "#1D4ED8",
        "shuffled_random_0p30_n0p45": "#DC2626",
        "dense_random_0p05_n0p45": "#7C3AED",
        "distance_decay_random_0p10_n0p45": "#D97706",
        "distance_decay_sign_0p20": "#059669",
    }


if __name__ == "__main__":
    main()
