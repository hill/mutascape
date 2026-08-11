"""Prepare GB1 dataset for mutascape.

Source data
    DMS CSV: SPG1_STRSG_Wu_2016.csv from the ProteinGym substitutions release
    (https://proteingym.org, CC-BY-4.0). Point PROTEINGYM_DMS_DIR at your local
    copy of DMS_ProteinGym_substitutions/, or edit DEFAULT_PROTEINGYM_DIR below.
    4-position complete combinatorial DMS (Wu et al. 2016). 149,360 variants
    over positions V39/D40/G41/V54 in the 56-residue GB1 (IgG-binding domain).
    ProteinGym uses full Streptococcus G precursor numbering (448 residues), so
    the mutated positions appear as 265/266/267/280 in the raw CSV. We translate
    to the canonical 1-56 GB1 domain numbering (offset = -226) throughout.

    DMS_score is an enrichment ratio (selected / input) with WT = 1.0. Subtract
    WT_FITNESS = 1.0 to get fitness relative to WT (WT = 0.0, nonfunctional = -1.0).

    PDB structure: 2QMT (1.05 A X-ray structure of GB1 monomer, chain A, residues 1-56).
    Downloaded from RCSB. 2GI9 (GB1/Fc complex) is equivalent but its chain layout
    is more complex; 2QMT is the clean choice for a 56-residue domain contact map.

Processing steps
    1. Read the full combinatorial CSV, filter to variants where every mutation
       lies in {265, 266, 267, 280}, translate positions by -226.
    2. Write gb1_combinatorial.csv with mutation notation in domain numbering
       (e.g. 'V39G:D40A') and fitness relative to WT.
    3. Extract single-position variants into gb1.csv (mutascape DMS format).
    4. Save the 56-residue GB1 domain FASTA (residues 227-282 of the precursor).
    5. Download PDB 2QMT, extract chain A, compute C-beta distance matrix,
       save restricted to positions present in DMS data.

Outputs
    gb1.csv: position, wildtype, mutant, fitness (relative to WT) for the 4 sites
    gb1_combinatorial.csv: mutant, fitness, depth for all multi-mutant variants
    gb1_wt.fasta: 56-residue GB1 domain sequence
    gb1_2qmt.pdb: GB1 monomer structure
    gb1_distances.npz: pairwise C-beta distance matrix over all domain positions
"""

import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape import ContactMap

OUTPUT_DIR = Path(__file__).resolve().parent

# Local copy of the ProteinGym DMS_ProteinGym_substitutions/ directory. Override
# with the PROTEINGYM_DMS_DIR environment variable; the default assumes a sibling
# `proteingym` checkout next to this repository.
DEFAULT_PROTEINGYM_DIR = Path.home() / "proteingym" / "DMS_ProteinGym_substitutions"
PROTEINGYM_DIR = Path(os.environ.get("PROTEINGYM_DMS_DIR", DEFAULT_PROTEINGYM_DIR))
PROTEINGYM_CSV = PROTEINGYM_DIR / "SPG1_STRSG_Wu_2016.csv"

# ProteinGym uses full Streptococcus G precursor numbering. The GB1 IgG-binding
# domain is residues 227-282 in that numbering, i.e. positions 1-56 after the
# offset below. Wu et al. screen 4 sites: V39/D40/G41/V54 (domain numbering).
PROTEINGYM_OFFSET = 226
DOMAIN_START_PG = 227  # first residue of GB1 domain in precursor numbering
DOMAIN_END_PG = 282    # last residue
MUTATED_POSITIONS_PG = {265, 266, 267, 280}  # positions screened by Wu et al.

WT_FITNESS = 1.0  # ProteinGym DMS_score = enrichment ratio; WT enrichment is 1
PDB_ID = "2QMT"


def load_proteingym_table() -> pd.DataFrame:
    """Read the raw ProteinGym CSV and return only rows mutated exclusively
    within the 4 Wu et al. positions."""
    df = pd.read_csv(PROTEINGYM_CSV)
    keep = []
    for mutant in df["mutant"]:
        ok = True
        for m in mutant.split(":"):
            match = re.match(r"([A-Z])(\d+)([A-Z])", m)
            if not match:
                ok = False
                break
            pos_pg = int(match.group(2))
            if pos_pg not in MUTATED_POSITIONS_PG:
                ok = False
                break
        keep.append(ok)
    return df[keep].reset_index(drop=True)


def translate_mutant(mutant_str: str) -> str:
    """Convert ProteinGym mutation notation to GB1 domain numbering."""
    parts = []
    for m in mutant_str.split(":"):
        match = re.match(r"([A-Z])(\d+)([A-Z])", m)
        wt, pos_pg, mut = match.group(1), int(match.group(2)), match.group(3)
        parts.append(f"{wt}{pos_pg - PROTEINGYM_OFFSET}{mut}")
    return ":".join(parts)


def extract_wt_sequence(pg_df: pd.DataFrame) -> str:
    """Reconstruct the 56-residue GB1 domain WT sequence from any mutant row."""
    row = pg_df.iloc[0]
    mutated_seq = row["mutated_sequence"]
    wt = list(mutated_seq)
    for m in row["mutant"].split(":"):
        match = re.match(r"([A-Z])(\d+)([A-Z])", m)
        wt_aa, pos_pg, mut_aa = match.group(1), int(match.group(2)), match.group(3)
        assert mutated_seq[pos_pg - 1] == mut_aa
        wt[pos_pg - 1] = wt_aa
    full_wt = "".join(wt)
    return full_wt[DOMAIN_START_PG - 1 : DOMAIN_END_PG]


def prepare_dms(pg_df: pd.DataFrame, wt_seq: str) -> None:
    """Write gb1.csv (singles only) and gb1_combinatorial.csv (all multi-mutants)."""
    translated = pg_df["mutant"].apply(translate_mutant)
    fitness = pg_df["DMS_score"].astype(float) - WT_FITNESS
    depth = translated.apply(lambda s: s.count(":") + 1)

    combinatorial = pd.DataFrame({
        "mutant": translated,
        "fitness": fitness,
        "depth": depth,
    })
    combinatorial.to_csv(OUTPUT_DIR / "gb1_combinatorial.csv", index=False)
    print(f"Combinatorial: {len(combinatorial)} variants")
    print(f"  depth distribution: {combinatorial['depth'].value_counts().sort_index().to_dict()}")
    print(f"  fitness range: {fitness.min():.3f} to {fitness.max():.3f}")

    singles_rows = []
    for mutant_str, fit in zip(translated, fitness):
        if ":" in mutant_str:
            continue
        match = re.match(r"([A-Z])(\d+)([A-Z])", mutant_str)
        wt_aa, pos, mut_aa = match.group(1), int(match.group(2)), match.group(3)
        assert wt_seq[pos - 1] == wt_aa
        singles_rows.append({
            "position": pos,
            "wildtype": wt_aa,
            "mutant": mut_aa,
            "fitness": float(fit),
        })
    singles = pd.DataFrame(singles_rows)
    singles.to_csv(OUTPUT_DIR / "gb1.csv", index=False)
    print(f"Singles: {len(singles)} variants, {singles['position'].nunique()} positions")


def write_wt_fasta(wt_seq: str) -> None:
    fasta_path = OUTPUT_DIR / "gb1_wt.fasta"
    header = ">GB1_WT Streptococcus G IgG-binding domain B1 (residues 1-56)"
    fasta_path.write_text(f"{header}\n{wt_seq}\n")
    print(f"WT FASTA: {len(wt_seq)} residues -> {fasta_path.name}")


def download_pdb() -> Path:
    pdb_path = OUTPUT_DIR / f"gb1_{PDB_ID.lower()}.pdb"
    if pdb_path.exists():
        print(f"PDB already exists: {pdb_path.name}")
        return pdb_path
    url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, pdb_path)
    print(f"  saved to {pdb_path.name}")
    return pdb_path


def prepare_contact_map(wt_seq: str) -> None:
    pdb_path = download_pdb()
    cm = ContactMap.from_pdb(pdb_path, threshold=8.0, chain_id="A")
    print(f"PDB contact map: {cm.n_positions} positions, {cm.n_contacts} contacts, "
          f"residue range {min(cm.positions)}-{max(cm.positions)}")

    # Verify numbering matches the GB1 domain (PDB residues 1-56 map to our WT seq)
    pdb_positions = list(cm.positions)
    domain_positions = [p for p in pdb_positions if 1 <= p <= len(wt_seq)]
    if len(domain_positions) < 50:
        raise RuntimeError(
            f"PDB {PDB_ID} chain A residue numbering does not match domain 1-56 "
            f"(got {len(domain_positions)} positions in expected range)"
        )

    sub = cm.submap(domain_positions)
    np.savez(
        OUTPUT_DIR / "gb1_distances.npz",
        distances=sub.distances,
        positions=np.array(domain_positions),
    )
    print(f"Saved contact map: {sub.n_positions} positions, {sub.n_contacts} contacts")


if __name__ == "__main__":
    pg_df = load_proteingym_table()
    print(f"Loaded {len(pg_df)} rows from ProteinGym (filtered to Wu et al. 4 positions)")

    wt_seq = extract_wt_sequence(pg_df)
    print(f"WT domain: {wt_seq}")
    expected_positions = {39: "V", 40: "D", 41: "G", 54: "V"}
    for pos, aa in expected_positions.items():
        assert wt_seq[pos - 1] == aa, f"WT seq at position {pos} is {wt_seq[pos-1]}, expected {aa}"

    write_wt_fasta(wt_seq)
    prepare_dms(pg_df, wt_seq)
    prepare_contact_map(wt_seq)
    print("Done.")
