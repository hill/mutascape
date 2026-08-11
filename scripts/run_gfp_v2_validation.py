"""Whole-protein GFP validation for the Mutascape v2 (DMS-anchored Potts) reference.

Companion to ``run_gfp_supplemental_validation.py`` (the v1 / no-couplings baseline).
Builds Mutascape landscapes with MSA-fitted Potts couplings at a sweep of
``msa_coupling_scale`` values, samples multi-mutant variants from each, and
compares depth-resolved landscape statistics to the sparse multi-mutant
Sarkisyan et al. GFP dataset.

Reports:
  - additive correlation per depth (Sarkisyan vs synthetic v1 / v2 at each scale)
  - sign-epistasis fraction per depth
  - fitness distribution moments per depth
  - roughness-to-slope ratio (overall)

The user-facing question this answers: does swapping random Gaussian couplings
for MSA-fitted Potts couplings reduce the v1 GFP smoothness bias on real
multi-mutant data?

Outputs:
  - notebooks/validation/output_gfp_v2/gfp_v2_validation_summary.csv
"""

from __future__ import annotations

from pathlib import Path

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
    REPO_ROOT.parent
    / "mutantopt/experiments/datasets/proteingym/GFP_AEQVI_Sarkisyan_2016.csv"
)

DMS_CSV = DATA_DIR / "gfp.csv"
DISTANCES_NPZ = DATA_DIR / "gfp_distances.npz"
WT_FASTA = DATA_DIR / "gfp_wt.fasta"
MSA_COUPLINGS_NPZ = DATA_DIR / "gfp_msa_couplings.npz"
DEPTHS = list(range(2, 8))
N_PER_DEPTH = 3000
SEED = 42

# v2 coupling-scale sweep. scale=0 reduces v2 to v1.
SCALE_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def _build_v1(singles_min: float):
    return build_landscape(
        DMS_CSV, DISTANCES_NPZ, WT_FASTA,
        alpha=1.5, sigma=0.0, seed=SEED,
        nonlinearity="fit", scale=None, floor_hard=singles_min,
        combinatorial_upside=3.0, noise_sigma=0.3,
    )


def _build_v2(singles_min: float, msa_scale: float):
    epistasis = MSACouplingsEpistasis.from_npz(MSA_COUPLINGS_NPZ, scale=msa_scale)
    return build_landscape(
        DMS_CSV, DISTANCES_NPZ, WT_FASTA,
        alpha=1.5, seed=SEED,
        nonlinearity="fit", scale=None, floor_hard=singles_min,
        combinatorial_upside=3.0, noise_sigma=0.3,
        msa_couplings=epistasis,
    )


def _summary_row(real_metrics: dict, synth_metrics: dict, label: str, depth: int) -> dict:
    return {
        "config": label,
        "depth": depth,
        "real_n": real_metrics["fitness_moments"].get(depth, {}).get("n"),
        "synth_n": synth_metrics["fitness_moments"].get(depth, {}).get("n"),
        "real_additive_correlation": real_metrics["additive_correlation"].get(depth),
        "synth_additive_correlation": synth_metrics["additive_correlation"].get(depth),
        "additive_diff": (
            (synth_metrics["additive_correlation"].get(depth) or 0)
            - (real_metrics["additive_correlation"].get(depth) or 0)
        ),
        "real_sign_epistasis": real_metrics["sign_epistasis"].get(depth),
        "synth_sign_epistasis": synth_metrics["sign_epistasis"].get(depth),
        "sign_diff": (
            (synth_metrics["sign_epistasis"].get(depth) or 0)
            - (real_metrics["sign_epistasis"].get(depth) or 0)
        ),
        "real_dev_mae": real_metrics["deviation_stats"].get(depth, {}).get("mae"),
        "synth_dev_mae": synth_metrics["deviation_stats"].get(depth, {}).get("mae"),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    effects = load_single_site_effects(DMS_CSV)
    singles_min = min(effect for pos in effects.values() for effect in pos.values())

    print("Loading Sarkisyan multi-mutant data ...")
    real_df = load_proteingym_multimutant(PROTEINGYM_CSV, DMS_CSV, wt_fitness=3.72)
    real_metrics = compute_all_metrics(real_df, effects, DEPTHS)
    real_rs = roughness_slope_ratio(real_df)

    rows = []
    print(f"\n{'config':24s} {'depth':>5s} {'real_add':>9s} {'synth_add':>10s} {'real_sign':>10s} {'synth_sign':>11s}")
    print("-" * 75)

    # v1 baseline
    print("Building v1 (no couplings)...")
    lc_v1 = _build_v1(singles_min)
    synth_v1 = sample_synthetic_data(lc_v1, DEPTHS, n_per_depth=N_PER_DEPTH)
    m_v1 = compute_all_metrics(synth_v1, effects, DEPTHS)
    rs_v1 = roughness_slope_ratio(synth_v1)
    for d in DEPTHS:
        row = _summary_row(real_metrics, m_v1, "v1_no_couplings", d)
        row["roughness_slope"] = rs_v1
        rows.append(row)
        print(
            f"{'v1_no_couplings':24s} {d:>5d} "
            f"{row['real_additive_correlation'] or 0:>9.3f} "
            f"{row['synth_additive_correlation'] or 0:>10.3f} "
            f"{row['real_sign_epistasis'] or 0:>10.3f} "
            f"{row['synth_sign_epistasis'] or 0:>11.3f}"
        )

    # v2 sweep
    for scale in SCALE_SWEEP:
        label = f"v2_msa_scale_{scale:.2f}"
        print(f"Building v2 at msa_coupling_scale={scale}...")
        lc_v2 = _build_v2(singles_min, scale)
        synth_v2 = sample_synthetic_data(lc_v2, DEPTHS, n_per_depth=N_PER_DEPTH)
        m_v2 = compute_all_metrics(synth_v2, effects, DEPTHS)
        rs_v2 = roughness_slope_ratio(synth_v2)
        for d in DEPTHS:
            row = _summary_row(real_metrics, m_v2, label, d)
            row["roughness_slope"] = rs_v2
            rows.append(row)
            print(
                f"{label:24s} {d:>5d} "
                f"{row['real_additive_correlation'] or 0:>9.3f} "
                f"{row['synth_additive_correlation'] or 0:>10.3f} "
                f"{row['real_sign_epistasis'] or 0:>10.3f} "
                f"{row['synth_sign_epistasis'] or 0:>11.3f}"
            )

    summary = pd.DataFrame(rows)
    summary["real_roughness_slope"] = rs_real if (rs_real := real_rs) else None
    out_csv = OUTPUT_DIR / "gfp_v2_validation_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # Per-config aggregated diff summary
    print("\nAggregated mean-abs deviations from real (across depths 2-7):")
    agg = (
        summary.groupby("config")[["additive_diff", "sign_diff"]]
        .apply(lambda df: df.abs().mean())
        .reset_index()
    )
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
