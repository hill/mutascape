"""Mutascape v2 whole-protein GFP validation with fitness-matched subsampling.

Extends ``scripts/run_gfp_v2_validation.py`` to correct for the Sarkisyan
selection bias. The Sarkisyan multi-mutant library was pre-selected for
functional variants, so high-depth Sarkisyan variants are biased toward the
functional tail. Comparing raw synthetic samples (uniformly drawn from the
multi-mutant cube) against this biased real distribution may under-state
v1's smoothness gap.

Correction: per depth, bin the real fitness distribution into ``n_bins``,
then stratified-sample a synthetic subset whose per-bin counts match real.
Metrics are then computed on the matched samples.

Reports the same depth-resolved metrics as the unmatched version, plus a
diagnostic showing the fitness-distribution alignment per depth.

Outputs:
  - notebooks/validation/output_gfp_v2/gfp_v2_validation_matched_summary.csv
  - notebooks/validation/output_gfp_v2/gfp_v2_fitness_match_diagnostic.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.epistasis import MSACouplingsEpistasis
from mutascape.validation.calibration import build_landscape
from mutascape.validation.metrics import compute_all_metrics, roughness_slope_ratio
from mutascape.validation.real_data import (
    load_proteingym_multimutant,
    load_single_site_effects,
    sample_synthetic_data,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data/gfp"
OUTPUT_DIR = REPO_ROOT / "notebooks/validation/output_gfp_v2"
PROTEINGYM_CSV = (
    REPO_ROOT.parent / "mutantopt/experiments/datasets/proteingym/GFP_AEQVI_Sarkisyan_2016.csv"
)

DMS_CSV = DATA_DIR / "gfp.csv"
DISTANCES_NPZ = DATA_DIR / "gfp_distances.npz"
WT_FASTA = DATA_DIR / "gfp_wt.fasta"
MSA_COUPLINGS_NPZ = DATA_DIR / "gfp_msa_couplings.npz"
DEPTHS = list(range(2, 8))
N_OVERSAMPLE_PER_DEPTH = 30_000  # oversampled before stratified matching
N_BINS = 20
SEED = 42

CONFIGS = [
    ("v1_no_couplings", None),
    ("v2_scale_0.10", 0.10),
    ("v2_scale_0.30", 0.30),
    ("v2_scale_0.50", 0.50),
    ("v2_scale_1.00", 1.00),
]


def fitness_matched_subsample(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    n_bins: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified resample of ``synth_df`` so per-depth fitness histograms match ``real_df``.

    Returns
    -------
    matched_synth : DataFrame
        Subset of synth_df whose per-depth fitness histograms match real_df.
    diagnostic : DataFrame
        Per-depth match diagnostic: bin range, target counts, attained counts.
    """
    rng = np.random.default_rng(seed)
    matched_rows: list[pd.DataFrame] = []
    diag_rows: list[dict] = []
    for d in sorted(real_df["depth"].unique()):
        real_d = real_df[real_df["depth"] == d]
        synth_d = synth_df[synth_df["depth"] == d]
        if len(real_d) == 0 or len(synth_d) == 0:
            continue
        lo = float(min(real_d["fitness"].min(), synth_d["fitness"].min()))
        hi = float(max(real_d["fitness"].max(), synth_d["fitness"].max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        real_counts, _ = np.histogram(real_d["fitness"], bins=bins)
        synth_bins = np.clip(np.digitize(synth_d["fitness"], bins) - 1, 0, n_bins - 1)
        achieved = 0
        deficit_bins = 0
        for b in range(n_bins):
            target = int(real_counts[b])
            if target == 0:
                continue
            bin_mask = synth_bins == b
            available_idx = np.where(bin_mask)[0]
            if len(available_idx) == 0:
                deficit_bins += 1
                continue
            replace = len(available_idx) < target
            pick = rng.choice(available_idx, size=target, replace=replace)
            matched_rows.append(synth_d.iloc[pick])
            achieved += target
        diag_rows.append(
            {
                "depth": int(d),
                "n_real": int(len(real_d)),
                "n_synth_available": int(len(synth_d)),
                "n_matched": achieved,
                "n_empty_bins_in_synth": deficit_bins,
                "fitness_lo": lo,
                "fitness_hi": hi,
            }
        )
    matched = pd.concat(matched_rows, ignore_index=True) if matched_rows else synth_df.iloc[0:0]
    return matched, pd.DataFrame(diag_rows)


def _build_v1(singles_min: float):
    return build_landscape(
        DMS_CSV,
        DISTANCES_NPZ,
        WT_FASTA,
        alpha=1.5,
        sigma=0.0,
        seed=SEED,
        nonlinearity="fit",
        scale=None,
        floor_hard=singles_min,
        combinatorial_upside=3.0,
        noise_sigma=0.3,
    )


def _build_v2(singles_min: float, msa_scale: float):
    epistasis = MSACouplingsEpistasis.from_npz(MSA_COUPLINGS_NPZ, scale=msa_scale)
    return build_landscape(
        DMS_CSV,
        DISTANCES_NPZ,
        WT_FASTA,
        alpha=1.5,
        seed=SEED,
        nonlinearity="fit",
        scale=None,
        floor_hard=singles_min,
        combinatorial_upside=3.0,
        noise_sigma=0.3,
        msa_couplings=epistasis,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    effects = load_single_site_effects(DMS_CSV)
    singles_min = min(effect for pos in effects.values() for effect in pos.values())

    print("Loading Sarkisyan multi-mutant data ...")
    real_df = load_proteingym_multimutant(PROTEINGYM_CSV, DMS_CSV, wt_fitness=3.72)
    real_df = real_df[real_df["depth"].isin(DEPTHS)].reset_index(drop=True)
    real_metrics = compute_all_metrics(real_df, effects, DEPTHS)
    rs_real = roughness_slope_ratio(real_df)

    rows: list[dict] = []
    diag_all: list[pd.DataFrame] = []
    print(
        f"\n{'config':18s} {'depth':>5s} {'real_n':>7s} {'matched_n':>9s} "
        f"{'real_add':>9s} {'matched_add':>11s} {'real_sign':>10s} "
        f"{'matched_sign':>13s}"
    )
    print("-" * 100)

    for label, scale in CONFIGS:
        if scale is None:
            print(f"Building {label} ...")
            landscape = _build_v1(singles_min)
        else:
            print(f"Building {label} (msa_coupling_scale={scale}) ...")
            landscape = _build_v2(singles_min, scale)

        synth_df = sample_synthetic_data(landscape, DEPTHS, n_per_depth=N_OVERSAMPLE_PER_DEPTH)
        matched_df, diag = fitness_matched_subsample(real_df, synth_df, n_bins=N_BINS, seed=SEED)
        diag.insert(0, "config", label)
        diag_all.append(diag)

        matched_metrics = compute_all_metrics(matched_df, effects, DEPTHS)
        rs_matched = roughness_slope_ratio(matched_df)

        for d in DEPTHS:
            real_add = real_metrics["additive_correlation"].get(d)
            mat_add = matched_metrics["additive_correlation"].get(d)
            real_sign = real_metrics["sign_epistasis"].get(d)
            mat_sign = matched_metrics["sign_epistasis"].get(d)
            real_n = real_metrics["fitness_moments"].get(d, {}).get("n")
            mat_n = matched_metrics["fitness_moments"].get(d, {}).get("n")
            row = {
                "config": label,
                "scale": scale if scale is not None else 0.0,
                "depth": d,
                "real_n": real_n,
                "matched_n": mat_n,
                "real_additive_correlation": real_add,
                "matched_additive_correlation": mat_add,
                "additive_diff": (mat_add or 0) - (real_add or 0),
                "real_sign_epistasis": real_sign,
                "matched_sign_epistasis": mat_sign,
                "sign_diff": (mat_sign or 0) - (real_sign or 0),
                "real_dev_mae": real_metrics["deviation_stats"].get(d, {}).get("mae"),
                "matched_dev_mae": matched_metrics["deviation_stats"].get(d, {}).get("mae"),
                "roughness_slope_real": rs_real,
                "roughness_slope_matched": rs_matched,
            }
            rows.append(row)
            print(
                f"{label:18s} {d:>5d} {real_n or 0:>7d} {mat_n or 0:>9d} "
                f"{real_add or 0:>9.3f} {mat_add or 0:>11.3f} "
                f"{real_sign or 0:>10.3f} {mat_sign or 0:>13.3f}"
            )

    summary = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "gfp_v2_validation_matched_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    diag_df = pd.concat(diag_all, ignore_index=True)
    diag_csv = OUTPUT_DIR / "gfp_v2_fitness_match_diagnostic.csv"
    diag_df.to_csv(diag_csv, index=False)
    print(f"wrote {diag_csv}")

    print("\nFitness-matched mean abs deviation from Sarkisyan (depths 2-7):")
    agg = (
        summary.groupby("config")[["additive_diff", "sign_diff"]]
        .apply(lambda df: df.abs().mean())
        .reset_index()
    )
    agg["roughness_slope_matched"] = (
        summary.groupby("config")["roughness_slope_matched"].first().values
    )
    agg["roughness_slope_real"] = float(rs_real)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
