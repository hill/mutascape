"""Prepare GFP dataset for mutascape.

Source data
    DMS CSV: mutantopt/experiments/datasets/proteingym/GFP_AEQVI_Sarkisyan_2016_singles.csv
    Single-site fluorescence measurements from avGFP DMS (Sarkisyan et al., 2016).
    Variant notation is simple (e.g. "K3R"). Fitness column is "DMS_score"
    (log10 brightness). WT score is 3.72 (Gonzalez-Somermeyer 2022).

    WT FASTA: mutantopt/experiments/datasets/proteingym/GFP_AEQVI_Sarkisyan_2016_singles_wt.fasta
    237-residue avGFP sequence from Aequorea victoria.

    PDB structure: 2WUR (superfolder GFP, chain A)
    Downloaded from RCSB: https://files.rcsb.org/download/2WUR.pdb
    Note: 2WUR is sfGFP which has a few mutations vs avGFP. For contact map
    purposes this is fine since the backbone fold is identical.

Processing steps
    1. Parse simple variant notation to (position, wildtype_aa, mutant_aa).
    2. Subtract WT fitness (DMS_score = 3.72) so effects are relative to WT.
    3. Extract C-beta contact map from PDB 2WUR chain A and save the
       pairwise distance matrix with 1-indexed positions matching the
       DMS data.

Outputs
    gfp.csv: position, wildtype, mutant, fitness (relative to WT)
    gfp_wt.fasta: wild-type sequence
    gfp_distances.npz: pairwise C-beta distance matrix
"""

import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape import ContactMap

OUTPUT_DIR = Path(__file__).resolve().parent
MUTANTOPT_DIR = (
    OUTPUT_DIR.parent.parent.parent / "mutantopt" / "experiments" / "datasets" / "proteingym"
)

WT_FITNESS = 3.72  # log10 brightness (Gonzalez-Somermeyer 2022)


def prepare_dms():
    """Convert ProteinGym GFP CSV to mutascape format."""
    raw = pd.read_csv(MUTANTOPT_DIR / "GFP_AEQVI_Sarkisyan_2016_singles.csv")

    # Read WT sequence
    from Bio import SeqIO

    wt_record = next(
        SeqIO.parse(MUTANTOPT_DIR / "GFP_AEQVI_Sarkisyan_2016_singles_wt.fasta", "fasta")
    )
    wt_seq = str(wt_record.seq)

    rows = []
    for _, row in raw.iterrows():
        m = re.match(r"([A-Z])(\d+)([A-Z])", row["mutant"])
        if not m:
            continue
        wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
        if pos > len(wt_seq) or wt_seq[pos - 1] != wt_aa:
            continue
        rows.append(
            {
                "position": pos,
                "wildtype": wt_aa,
                "mutant": mut_aa,
                "fitness": row["DMS_score"] - WT_FITNESS,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "gfp.csv", index=False)
    print(f"DMS: {len(out)} variants, {out['position'].nunique()} positions")
    print(f"  fitness range: {out['fitness'].min():.3f} to {out['fitness'].max():.3f}")

    # Copy WT FASTA
    shutil.copy(
        MUTANTOPT_DIR / "GFP_AEQVI_Sarkisyan_2016_singles_wt.fasta",
        OUTPUT_DIR / "gfp_wt.fasta",
    )
    print(f"  WT sequence: {len(wt_seq)} residues")


def download_pdb():
    """Download PDB 2WUR (superfolder GFP) from RCSB."""
    pdb_path = OUTPUT_DIR / "gfp_2wur.pdb"
    if pdb_path.exists():
        print(f"PDB already exists: {pdb_path}")
        return pdb_path

    url = "https://files.rcsb.org/download/2WUR.pdb"
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, pdb_path)
    print(f"  Saved to {pdb_path}")
    return pdb_path


def prepare_contact_map():
    """Extract contact map from PDB 2WUR."""
    pdb_path = download_pdb()
    cm = ContactMap.from_pdb(pdb_path, threshold=8.0, chain_id="A")

    # 2WUR residue numbering starts at 1 and matches the DMS positions
    # directly (both use 1-indexed positions into the 237-residue sequence).
    # Just restrict to positions present in both PDB and DMS.
    dms = pd.read_csv(OUTPUT_DIR / "gfp.csv")
    dms_positions = sorted(dms["position"].unique())

    available = [p for p in dms_positions if p in set(cm.positions)]
    sub = cm.submap(available)

    np.savez(
        OUTPUT_DIR / "gfp_distances.npz",
        distances=sub.distances,
        positions=np.array(available),
    )
    print(f"Contact map: {sub.n_positions} positions, {sub.n_contacts} contacts")


if __name__ == "__main__":
    prepare_dms()
    prepare_contact_map()
    print("Done.")
