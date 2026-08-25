"""Mutascape v2 contact-prediction sanity check.

For an MSA-fitted Potts coupling tensor to be biologically meaningful, the
strongest pairwise couplings should concentrate at short 3D distances. This
is the foundational observation behind direct coupling analysis (Morcos et
al. 2011, Marks et al. 2011): co-evolving residues tend to be in physical
contact.

This script computes, for CreiLOV and GFP:
  - Frobenius norm of each (i, j) coupling block (after zero-sum gauge)
  - The structural Cα-Cα distance between i and j (from the prepared
    distances .npz)

It then reports:
  - Spearman correlation between coupling magnitude and distance (negative
    expected: stronger couplings at shorter distances)
  - Top-K precision: of the K most strongly coupled pairs, what fraction
    are within the 8 Å contact threshold
  - PR-style curve points (precision at K = 50, 100, 200, 500, 1000)

A negative Spearman and top-K precision well above the contact-density
baseline support the interpretation that the v2 MSA-Potts couplings carry
structural signal rather than noise.

Outputs:
  - notebooks/validation/output_v2_contact_check/contact_prediction_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks/validation/output_v2_contact_check"

PROTEINS = {
    "creilov": {
        "couplings_npz": REPO_ROOT / "data/creilov/creilov_msa_couplings.npz",
        "distances_npz": REPO_ROOT / "data/creilov/creilov_distances.npz",
    },
    "gfp": {
        "couplings_npz": REPO_ROOT / "data/gfp/gfp_msa_couplings.npz",
        "distances_npz": REPO_ROOT / "data/gfp/gfp_distances.npz",
    },
}
CONTACT_THRESHOLD_A = 8.0


def load_coupling_distance(couplings_npz: Path, distances_npz: Path) -> pd.DataFrame:
    cdata = np.load(couplings_npz)
    ddata = np.load(distances_npz)
    J = cdata["coupling_matrix"]  # (L, L, K, K)
    c_pos = list(int(p) for p in cdata["positions"])
    d_pos = list(int(p) for p in ddata["positions"])
    D = ddata["distances"]  # (Nd, Nd)

    pos_to_d = {p: i for i, p in enumerate(d_pos)}
    L = J.shape[0]

    rows = []
    for i in range(L):
        pi = c_pos[i]
        for j in range(i + 1, L):
            pj = c_pos[j]
            if pi not in pos_to_d or pj not in pos_to_d:
                continue
            dist = float(D[pos_to_d[pi], pos_to_d[pj]])
            frob = float(np.linalg.norm(J[i, j]))
            rows.append(
                {
                    "pos_i": pi,
                    "pos_j": pj,
                    "distance_A": dist,
                    "coupling_frob": frob,
                    "is_contact": bool(dist <= CONTACT_THRESHOLD_A),
                }
            )
    return pd.DataFrame(rows)


def precision_at_k(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    df_sorted = df.sort_values("coupling_frob", ascending=False).reset_index(drop=True)
    rows = []
    for k in ks:
        top = df_sorted.head(k)
        precision = float(top["is_contact"].mean()) if len(top) else float("nan")
        rows.append(
            {"k": k, "precision": precision, "n_contact_in_top": int(top["is_contact"].sum())}
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    ks = [50, 100, 200, 500, 1000, 2000]
    for protein, paths in PROTEINS.items():
        df = load_coupling_distance(paths["couplings_npz"], paths["distances_npz"])
        df.to_csv(OUTPUT_DIR / f"{protein}_coupling_vs_distance.csv", index=False)

        # Baseline contact density: fraction of all pairs that are within threshold.
        contact_density = float(df["is_contact"].mean())

        # Negative Spearman expected: shorter distance correlates with larger coupling.
        rho, p = spearmanr(df["distance_A"], df["coupling_frob"])

        prec = precision_at_k(df, ks)
        prec.insert(0, "protein", protein)
        prec["baseline_contact_density"] = contact_density
        prec["lift_over_baseline"] = prec["precision"] / contact_density

        print(f"\n=== {protein} ===")
        print(
            f"  pairs total: {len(df)}, pairs within 8 Å: "
            f"{int(df['is_contact'].sum())} ({contact_density:.3%})"
        )
        print(f"  Spearman (distance vs |J|_F): rho={rho:.3f}, p={p:.2e}")
        print(prec.to_string(index=False))

        summary_rows.append(
            {
                "protein": protein,
                "n_pairs": len(df),
                "n_contacts": int(df["is_contact"].sum()),
                "contact_density": contact_density,
                "spearman_rho": float(rho),
                "spearman_p": float(p),
            }
        )
        # Append the precision rows in a flat output
        prec.to_csv(OUTPUT_DIR / f"{protein}_precision_at_k.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "contact_prediction_summary.csv", index=False)


if __name__ == "__main__":
    main()
