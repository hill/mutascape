"""Prepare CreiLOV dataset for mutascape.

Source data
-----------
- DMS CSV: mutantopt/experiments/datasets/creilov/dms.csv
  Single-site fluorescence measurements from CreiLOV DMS experiment.
  Variant notation is HGVS (e.g. "p.Arg60Asp"). Fitness column is
  "log_mean" (log10 of mean fluorescence across 3 replicates).

- WT FASTA: mutantopt/experiments/datasets/creilov/wt.fasta
  119-residue CreiLOV LOV-domain sequence from FPbase.

- PDB structure: PDB 4I6G (chain A)
  X-ray crystal structure of Chlamydomonas reinhardtii cryptochrome-2.
  The CreiLOV domain is a subset of this larger protein. Downloaded
  from RCSB: https://files.rcsb.org/download/4I6G.pdb

Processing steps
----------------
1. Parse HGVS variant notation to (position, wildtype_aa, mutant_aa).
   Skip stop codon variants ("Ter").

2. Subtract wild-type fitness (log_mean = 4.1704) so all effects are
   relative to WT. This is required because mutascape expects effects
   where WT = 0.0 by definition.

3. Extract C-beta contact map from PDB 4I6G chain A. The PDB numbering
   starts at residue 21 (the LOV domain begins at position 21 in the
   full cryptochrome sequence). DMS positions are 2-119 (1-indexed into
   the 119-residue LOV domain sequence). The offset is 19:
   DMS position N maps to PDB residue N+19.

4. Extract the distance submatrix for PDB positions 21-138 (corresponding
   to DMS positions 2-119), then relabel with DMS numbering.

Outputs
-------
- creilov.csv: position, wildtype, mutant, fitness (relative to WT)
- creilov_wt.fasta: wild-type sequence
- creilov_distances.npz: pairwise C-beta distance matrix with DMS positions
- creilov_4i6g.pdb: raw PDB structure (for reference)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1

from mutascape import ContactMap

MUTANTOPT_DIR = Path("../../mutantopt/experiments/datasets/creilov")
OUTPUT_DIR = Path(__file__).parent


def prepare_dms():
    """Convert mutantopt DMS CSV to mutascape format."""
    raw = pd.read_csv(MUTANTOPT_DIR / "dms.csv")

    # WT fitness for baseline subtraction
    wt_fitness = raw[raw["variant"] == "wt"]["log_mean"].values[0]

    raw = raw[raw["variant"] != "wt"].copy()

    rows = []
    for _, row in raw.iterrows():
        v = row["variant"].replace("p.", "")
        wt_3 = v[:3]
        mut_3 = v[-3:]

        # Skip non-standard AAs (stop codons, etc.)
        if wt_3 not in protein_letters_3to1 or mut_3 not in protein_letters_3to1:
            continue

        pos = int(v[3:-3])
        wt_1 = protein_letters_3to1[wt_3]
        mut_1 = protein_letters_3to1[mut_3]
        effect = row["log_mean"] - wt_fitness

        rows.append({
            "position": pos,
            "wildtype": wt_1,
            "mutant": mut_1,
            "fitness": effect,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "creilov.csv", index=False)
    print(f"DMS: {len(out)} variants, {out['position'].nunique()} positions")
    print(f"  fitness range: {out['fitness'].min():.3f} to {out['fitness'].max():.3f}")


def prepare_contact_map():
    """Extract contact map from PDB with DMS-compatible numbering."""
    cm = ContactMap.from_pdb(OUTPUT_DIR / "creilov_4i6g.pdb", threshold=8.0, chain_id="A")

    # PDB positions 21-138 correspond to DMS positions 2-119 (offset = 19)
    offset = 19
    dms_positions = list(range(2, 120))
    pdb_positions = [p + offset for p in dms_positions]

    sub = cm.submap(pdb_positions)

    # Rebuild with DMS numbering
    lov_cm = ContactMap.from_distance_matrix(sub.distances, dms_positions, threshold=8.0)

    np.savez(
        OUTPUT_DIR / "creilov_distances.npz",
        distances=sub.distances,
        positions=np.array(dms_positions),
    )
    print(f"Contact map: {lov_cm.n_positions} positions, {lov_cm.n_contacts} contacts")


if __name__ == "__main__":
    prepare_dms()
    prepare_contact_map()
    print("Done.")
