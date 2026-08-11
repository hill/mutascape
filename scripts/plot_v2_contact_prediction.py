"""Plot the v2 MSA-Potts contact-prediction sanity check.

Produces:
  - A scatter of |J|_F vs 3D distance for CreiLOV and GFP, colour-coded by
    contact vs non-contact.
  - A precision-at-K curve (top-K coupling pairs, fraction that are within
    8 Å vs the baseline contact density).

Reads the CSVs produced by ``run_v2_contact_prediction_check.py``.
Outputs PDFs to ``notebooks/validation/output_v2_contact_check/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "notebooks/validation/output_v2_contact_check"

CONTACT_THRESHOLD_A = 8.0


def main() -> None:
    summary = pd.read_csv(OUT / "contact_prediction_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), tight_layout=True)

    for col, protein in enumerate(["creilov", "gfp"]):
        df = pd.read_csv(OUT / f"{protein}_coupling_vs_distance.csv")
        prec = pd.read_csv(OUT / f"{protein}_precision_at_k.csv")
        meta = summary[summary["protein"] == protein].iloc[0]

        ax = axes[0, col]
        # Highlight the top-decile of coupling magnitude to make the
        # contact-prediction signal visible against the dense background.
        top_thresh = df["coupling_frob"].quantile(0.9)
        background = df[df["coupling_frob"] < top_thresh]
        top = df[df["coupling_frob"] >= top_thresh]
        ax.scatter(
            background["distance_A"], background["coupling_frob"],
            s=2, alpha=0.10, color="#888888", label="bottom 90% of pairs",
        )
        ax.scatter(
            top.loc[~top["is_contact"], "distance_A"], top.loc[~top["is_contact"], "coupling_frob"],
            s=8, alpha=0.5, color="#3498db", label="top 10%, non-contact",
        )
        ax.scatter(
            top.loc[top["is_contact"], "distance_A"], top.loc[top["is_contact"], "coupling_frob"],
            s=10, alpha=0.7, color="#c0392b", label="top 10%, contact (≤8 Å)",
        )
        ax.axvline(CONTACT_THRESHOLD_A, color="black", lw=0.7, ls="--", alpha=0.6)
        ax.set_xlabel("Cα-Cα distance (Å)")
        ax.set_ylabel(r"$\|J_{ij}\|_F$ (gauge-fixed)")
        rho = meta["spearman_rho"]
        ax.set_title(f"{protein.upper()}: Spearman ρ = {rho:.3f}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, df["distance_A"].max())

        ax = axes[1, col]
        ax.plot(prec["k"], prec["precision"], "o-", color="#2980b9", label="top-K precision")
        baseline = float(meta["contact_density"])
        ax.axhline(baseline, color="black", lw=0.7, ls="--", label=f"random baseline = {baseline:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("Top-K most strongly coupled pairs (K)")
        ax.set_ylabel("Fraction within 8 Å contact")
        ax.set_title(f"{protein.upper()}: contact-prediction precision")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

    fig.savefig(OUT / "msa_contact_prediction.pdf")
    fig.savefig(OUT / "msa_contact_prediction.png", dpi=160)
    print(f"wrote {OUT}/msa_contact_prediction.pdf")
    print(f"wrote {OUT}/msa_contact_prediction.png")


if __name__ == "__main__":
    main()
