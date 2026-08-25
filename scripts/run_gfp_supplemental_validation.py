"""Run the current headline pipeline on sparse GFP multi-mutant data.

This is the supplementary generalisation check for the paper: same core
architecture as the GB1/TrpB headline configs (censored global-epistasis
function + sigma=0 + noise_sigma=0.3), evaluated on Sarkisyan et al.'s
full-protein sparse multi-mutant dataset where complete-space metrics such as
Walsh decomposition are not tractable.

Outputs:
  - notebooks/validation/output_gfp/gfp_reference_validation_summary.{csv,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mutascape.validation.calibration import build_landscape
from mutascape.validation.figures import fig_validation_summary, save_fig
from mutascape.validation.metrics import compute_all_metrics, roughness_slope_ratio
from mutascape.validation.real_data import (
    load_proteingym_multimutant,
    load_single_site_effects,
    sample_synthetic_data,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data/gfp"
OUTPUT_DIR = REPO_ROOT / "notebooks/validation/output_gfp"
PROTEINGYM_CSV = (
    REPO_ROOT.parent / "mutantopt/experiments/datasets/proteingym/GFP_AEQVI_Sarkisyan_2016.csv"
)

DMS_CSV = DATA_DIR / "gfp.csv"
DISTANCES_NPZ = DATA_DIR / "gfp_distances.npz"
WT_FASTA = DATA_DIR / "gfp_wt.fasta"
DEPTHS = list(range(2, 8))
N_PER_DEPTH = 3000
SEED = 42


def _summary_rows(real_metrics: dict, synth_metrics: dict) -> list[dict]:
    rows = []
    for depth in DEPTHS:
        rows.append(
            {
                "depth": depth,
                "real_n": real_metrics["fitness_moments"].get(depth, {}).get("n"),
                "synth_n": synth_metrics["fitness_moments"].get(depth, {}).get("n"),
                "real_additive_correlation": real_metrics["additive_correlation"].get(depth),
                "synth_additive_correlation": synth_metrics["additive_correlation"].get(depth),
                "real_sign_epistasis": real_metrics["sign_epistasis"].get(depth),
                "synth_sign_epistasis": synth_metrics["sign_epistasis"].get(depth),
                "real_deviation_mae": real_metrics["deviation_stats"].get(depth, {}).get("mae"),
                "synth_deviation_mae": synth_metrics["deviation_stats"].get(depth, {}).get("mae"),
                "real_fitness_mean": real_metrics["fitness_moments"].get(depth, {}).get("mean"),
                "synth_fitness_mean": synth_metrics["fitness_moments"].get(depth, {}).get("mean"),
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    effects = load_single_site_effects(DMS_CSV)
    singles_min = min(effect for pos in effects.values() for effect in pos.values())

    landscape = build_landscape(
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

    real_df = load_proteingym_multimutant(PROTEINGYM_CSV, DMS_CSV, wt_fitness=3.72)
    real_metrics = compute_all_metrics(real_df, effects, DEPTHS)

    synth_df = sample_synthetic_data(landscape, DEPTHS, n_per_depth=N_PER_DEPTH)
    synth_metrics = compute_all_metrics(synth_df, effects, DEPTHS)

    summary = pd.DataFrame(_summary_rows(real_metrics, synth_metrics))
    summary["real_r_over_s"] = roughness_slope_ratio(real_df)
    summary["synth_r_over_s"] = roughness_slope_ratio(synth_df)
    out_csv = OUTPUT_DIR / "gfp_reference_validation_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    fig = fig_validation_summary(
        real_metrics,
        synth_metrics,
        best_alpha=1.5,
        depths=DEPTHS,
        protein_name="GFP",
        synth_label="mutascape headline pipeline",
    )
    save_fig(fig, "gfp_reference_validation_summary", OUTPUT_DIR)

    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
