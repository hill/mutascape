"""Validate Mutascape v2 against the Chen 2023 CreiLOV combinatorial library.

This is the headline whole-protein validation per the 2024-2026 literature
scan: a *combinatorial multi-site* dataset on a protein for which we already
have v2 MSA-Potts couplings and DMS singles. Closes the gap that "v2 was
validated only on sparse GFP 2016 data" critique.

Library: Chen et al. 2023 (ACS Synth. Biol. 12, 1461) "SOLD library" -- 20
specific mutations across 15 specific residues, 184,320 theoretical
combinations, 165,428 measured by FACS-seq. Hosted by CombinGym
(sitonglab/CombinGym).

Method:
  - Parse the HGVS-style mutants (3-letter codes); convert to position/AA
    pairs and cross-reference with Mutascape's single-site DMS.
  - Centre the real fitness on WT (subtract WT log-fluorescence).
  - For each variant, evaluate Mutascape v1 (no couplings) and Mutascape v2
    (MSA-Potts) at several coupling scales.
  - Report:
      (a) Overall Spearman/Pearson correlation between Mutascape predictions
          and real fitness, by config.
      (b) Per-depth correlation (does v2 hold up at higher depths?).
      (c) Per-depth additive-correlation gap (real vs Mutascape additive
          baseline) to confirm v2 reduces over-additivity on real data.

Outputs:
  notebooks/validation/output_creilov_combinatorial/creilov_combinatorial_*.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from mutascape.epistasis import MSACouplingsEpistasis
from mutascape.validation.calibration import build_landscape

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data/creilov"
OUTPUT_DIR = REPO_ROOT / "notebooks/validation/output_creilov_combinatorial"

DMS_CSV = DATA_DIR / "creilov.csv"
DISTANCES_NPZ = DATA_DIR / "creilov_distances.npz"
WT_FASTA = DATA_DIR / "creilov_wt.fasta"
MSA_COUPLINGS_NPZ = DATA_DIR / "creilov_msa_couplings.npz"
COMBINATORIAL_XLSX = DATA_DIR / "creilov_combinatorial_raw.xlsx"

SEED = 42

# Three-letter to one-letter amino acid map.
THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}
MUTATION_RE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")


def parse_mutant(mutant_str: str) -> list[tuple[int, str, str]]:
    """Parse HGVS-style multi-mutant string into [(position, wt, mut), ...]."""
    if pd.isna(mutant_str) or mutant_str == "wt":
        return []
    parts = re.split(r"[;,]", mutant_str)
    mutations = []
    for part in parts:
        part = part.strip()
        m = MUTATION_RE.match(part)
        if m is None:
            raise ValueError(f"Cannot parse mutation: {part!r} (in {mutant_str!r})")
        wt = THREE_TO_ONE[m.group(1)]
        pos = int(m.group(2))
        mut = THREE_TO_ONE[m.group(3)]
        mutations.append((pos, wt, mut))
    return mutations


CONFIGS = [
    ("v1_no_couplings", None),
    ("v2_scale_0.10", 0.10),
    ("v2_scale_0.30", 0.30),
    ("v2_scale_0.50", 0.50),
    ("v2_scale_1.00", 1.00),
]


def _build_v1():
    return build_landscape(
        DMS_CSV,
        DISTANCES_NPZ,
        WT_FASTA,
        alpha=1.5,
        sigma=0.0,
        seed=SEED,
        nonlinearity="fit",
        combinatorial_upside=3.0,
        noise_sigma=0.0,
    )


def _build_v2(msa_scale: float):
    epistasis = MSACouplingsEpistasis.from_npz(MSA_COUPLINGS_NPZ, scale=msa_scale)
    return build_landscape(
        DMS_CSV,
        DISTANCES_NPZ,
        WT_FASTA,
        alpha=1.5,
        seed=SEED,
        nonlinearity="fit",
        combinatorial_upside=3.0,
        noise_sigma=0.0,
        msa_couplings=epistasis,
    )


def _evaluate_variants(
    landscape, variants_as_mut_lists: list[list[tuple[int, str, str]]]
) -> np.ndarray:
    """Evaluate Mutascape landscape on each variant. Returns fitness array."""
    wt = landscape.wild_type
    out = np.empty(len(variants_as_mut_lists), dtype=np.float64)
    for i, muts in enumerate(variants_as_mut_lists):
        seq = dict(wt)
        for pos, _wt_aa, mut_aa in muts:
            seq[pos] = mut_aa
        out[i] = landscape.evaluate(seq)
    return out


def _additive_prediction(landscape, variants_as_mut_lists):
    """Sum of single-site DMS effects for each variant (additive baseline)."""
    out = np.empty(len(variants_as_mut_lists), dtype=np.float64)
    for i, muts in enumerate(variants_as_mut_lists):
        s = 0.0
        for pos, _wt_aa, mut_aa in muts:
            s += landscape.single_site_effects[pos].get(mut_aa, 0.0)
        out[i] = s
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Chen 2023 CreiLOV combinatorial data ...")
    df = pd.read_excel(COMBINATORIAL_XLSX, sheet_name="Sheet1")
    wt_log = float(df[df["n_mut"] == 0]["mean_log"].iloc[0])
    df["real_fitness"] = df["mean_log"] - wt_log
    print(f"  N variants: {len(df)}, WT log-fluorescence: {wt_log:.3f}")
    print(f"  depth distribution: {df['n_mut'].value_counts().sort_index().to_dict()}")

    print("Parsing HGVS mutants...")
    df["mutations"] = df["mutant"].apply(parse_mutant)
    df = df[df["n_mut"] > 0].reset_index(drop=True)  # drop WT row
    df = df[df["n_mut"] <= 10].reset_index(drop=True)  # depths 1-10 are well-represented
    print(f"  after filtering 1 ≤ depth ≤ 10: {len(df)} variants")

    # Subsample to keep validation tractable: max N_PER_DEPTH variants per depth.
    # Correlations stabilise well below the full N at each depth and this lets the
    # validation finish in a few minutes per config rather than ~10.
    N_PER_DEPTH = 3000
    rng = np.random.default_rng(SEED)
    keep = []
    for d, idx_group in df.groupby("n_mut").groups.items():
        idx_list = list(idx_group)
        if len(idx_list) > N_PER_DEPTH:
            idx_list = list(rng.choice(idx_list, size=N_PER_DEPTH, replace=False))
        keep.extend(idx_list)
    df = df.loc[sorted(keep)].reset_index(drop=True)
    print(f"  after subsampling to ≤{N_PER_DEPTH}/depth: {len(df)} variants")
    print(f"  depth distribution: {df['n_mut'].value_counts().sort_index().to_dict()}")

    variants = df["mutations"].tolist()
    real = df["real_fitness"].to_numpy()
    depths = df["n_mut"].to_numpy()
    depths_unique = sorted(set(depths))

    summary_rows: list[dict] = []
    per_variant_csv = OUTPUT_DIR / "creilov_combinatorial_predictions.csv"
    out_records = pd.DataFrame({"mutant": df["mutant"], "n_mut": depths, "real_fitness": real})

    additive_pred = None

    for label, scale in CONFIGS:
        if scale is None:
            print(f"Building {label} ...")
            landscape = _build_v1()
        else:
            print(f"Building {label} (msa_coupling_scale={scale}) ...")
            landscape = _build_v2(scale)

        if additive_pred is None:
            additive_pred = _additive_prediction(landscape, variants)
            out_records["additive_prediction"] = additive_pred

        pred = _evaluate_variants(landscape, variants)
        out_records[label] = pred

        # Overall correlations
        overall_p, _ = pearsonr(real, pred)
        overall_s, _ = spearmanr(real, pred)

        # Per-depth
        for d in depths_unique:
            mask = depths == d
            if mask.sum() < 5:
                continue
            r_d = real[mask]
            p_d = pred[mask]
            a_d = additive_pred[mask]
            try:
                pearson_d, _ = pearsonr(r_d, p_d)
            except ValueError:
                pearson_d = np.nan
            try:
                spearman_d, _ = spearmanr(r_d, p_d)
            except ValueError:
                spearman_d = np.nan
            try:
                real_add_corr, _ = pearsonr(r_d, a_d)
            except ValueError:
                real_add_corr = np.nan
            try:
                synth_add_corr, _ = pearsonr(p_d, a_d)
            except ValueError:
                synth_add_corr = np.nan
            summary_rows.append(
                {
                    "config": label,
                    "scale": scale if scale is not None else 0.0,
                    "depth": int(d),
                    "n": int(mask.sum()),
                    "overall_pearson_to_real": overall_p,
                    "overall_spearman_to_real": overall_s,
                    "pearson_to_real": pearson_d,
                    "spearman_to_real": spearman_d,
                    "real_additive_correlation": real_add_corr,
                    "synth_additive_correlation": synth_add_corr,
                    "synth_minus_real_additive_diff": synth_add_corr - real_add_corr,
                }
            )
        print(f"  Overall Pearson(real, pred) = {overall_p:.3f}, Spearman = {overall_s:.3f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "creilov_combinatorial_summary.csv", index=False)
    print(f"\nwrote {OUTPUT_DIR}/creilov_combinatorial_summary.csv ({len(summary)} rows)")

    out_records.to_csv(per_variant_csv, index=False)
    print(f"wrote {per_variant_csv} ({len(out_records)} rows)")

    # Aggregate summary across depths 2-7
    agg_depths = [d for d in depths_unique if 2 <= d <= 7]
    agg = (
        summary[summary["depth"].isin(agg_depths)]
        .groupby("config")
        .agg(
            mean_pearson_per_depth=("pearson_to_real", "mean"),
            mean_spearman_per_depth=("spearman_to_real", "mean"),
            mean_synth_minus_real_add_diff=(
                "synth_minus_real_additive_diff",
                lambda x: x.abs().mean(),
            ),
            overall_pearson=("overall_pearson_to_real", "first"),
            overall_spearman=("overall_spearman_to_real", "first"),
        )
        .reset_index()
    )
    print("\nAggregated per-config (depths 2-7):")
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
