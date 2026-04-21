"""Prepare TrpB4 dataset for mutascape (Johnston et al. 2024).

Source data
    Combinatorial landscape CSV: trpb4_scaled.csv (from SSMuLA Zenodo bundle,
    zenodo.org/records/13910506). Parent Tm9D8* (engineered Thermotoga maritima
    TrpB) screened at 4 active-site positions V183, F184, V227, S228. 20^4 =
    160,000 theoretical variants, 159,129 measured after filtering stop codons.

    The `fitness` column in the scaled CSV is growth rate normalised so that
    the best-observed variant (AIKG) has fitness 1.0. The engineered parent
    VFVS sits at 0.408. We subtract the parent fitness so the mutascape
    convention "WT = 0" applies: parent becomes 0.0, best becomes +0.592,
    and the most-depleted variants sit around -0.57.

    PDB structure: 8VHH (engineered Tm9D8* holoenzyme, homodimer 397 aa per
    chain). Chain A, residue numbering already matches positions 183/184/
    227/228.

Processing steps
    1. Read trpb4_scaled.csv, drop rows with any stop codon ('*' in AAs).
    2. Subtract parent fitness to recentre to WT = 0.
    3. Write trpb_combinatorial.csv in the same mutant/fitness/depth schema
       used by gb1_combinatorial.csv (ProteinGym-style mutation strings:
       "V183A:F184I" etc.). Parent row (muts == "WT") is dropped.
    4. Extract single-position variants into trpb.csv (position, wildtype,
       mutant, fitness).
    5. Save the 397-residue Tm9D8* WT sequence from PDB 8VHH chain A.
    6. Compute C-beta contact map on the four mutated positions.

Outputs
    trpb.csv: position, wildtype, mutant, fitness (relative to parent)
    trpb_combinatorial.csv: mutant, fitness, depth for all multi-mutant variants
    trpb_wt.fasta: 397-residue Tm9D8* sequence (from PDB chain A)
    trpb_8vhh.pdb: engineered Tm9D8* structure (already downloaded)
    trpb_distances.npz: pairwise C-beta distance matrix on chain A positions
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mutascape import ContactMap

OUTPUT_DIR = Path(__file__).resolve().parent

SCALED_CSV = OUTPUT_DIR / "trpb4_scaled.csv"
PDB_PATH = OUTPUT_DIR / "trpb_8vhh.pdb"

MUTATED_POSITIONS = (183, 184, 227, 228)
PARENT_AA_AT_POSITION = {183: "V", 184: "F", 227: "V", 228: "S"}


def _extract_wt_sequence() -> str:
    from Bio import PDB
    from Bio.PDB.Polypeptide import index_to_one, three_to_index

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("8vhh", str(PDB_PATH))
    chain = next(iter(structure))["A"]
    residues = {
        r.id[1]: index_to_one(three_to_index(r.get_resname()))
        for r in chain
        if PDB.is_aa(r, standard=True)
    }
    min_pos, max_pos = min(residues), max(residues)
    # Build a sequence indexed 1..max_pos, filling gaps with 'X' for safety.
    seq = ["X"] * max_pos
    for p, aa in residues.items():
        seq[p - 1] = aa
    wt = "".join(seq)
    for p, expected in PARENT_AA_AT_POSITION.items():
        assert wt[p - 1] == expected, (
            f"PDB 8VHH at position {p} is {wt[p - 1]}, expected {expected}"
        )
    return wt


def prepare_dms() -> None:
    df = pd.read_csv(SCALED_CSV)
    df = df[df["# Stop"] == 0].reset_index(drop=True)
    parent_row = df[df["muts"] == "WT"]
    assert len(parent_row) == 1, f"expected exactly one parent row, got {len(parent_row)}"
    parent_fitness = float(parent_row["fitness"].iloc[0])
    print(f"parent fitness before re-centring: {parent_fitness:.4f}")

    df = df[df["muts"] != "WT"].reset_index(drop=True)
    fitness_rel = df["fitness"].astype(float) - parent_fitness
    depth = df["n_mut"].astype(int)

    combinatorial = pd.DataFrame({
        "mutant": df["muts"],
        "fitness": fitness_rel,
        "depth": depth,
    })
    combinatorial.to_csv(OUTPUT_DIR / "trpb_combinatorial.csv", index=False)
    print(f"Combinatorial: {len(combinatorial)} variants")
    print(
        "  depth distribution: "
        f"{combinatorial['depth'].value_counts().sort_index().to_dict()}"
    )
    print(f"  fitness range: {fitness_rel.min():.3f} to {fitness_rel.max():.3f}")

    singles_rows = []
    singles_mask = combinatorial["depth"] == 1
    for mutant_str, fit in zip(combinatorial.loc[singles_mask, "mutant"], fitness_rel[singles_mask]):
        import re
        match = re.match(r"([A-Z])(\d+)([A-Z])", mutant_str)
        assert match is not None, f"could not parse single mutant: {mutant_str}"
        wt_aa, pos, mut_aa = match.group(1), int(match.group(2)), match.group(3)
        assert PARENT_AA_AT_POSITION[pos] == wt_aa, (
            f"mutant {mutant_str} inconsistent with parent at position {pos}"
        )
        singles_rows.append({
            "position": pos,
            "wildtype": wt_aa,
            "mutant": mut_aa,
            "fitness": float(fit),
        })
    singles = pd.DataFrame(singles_rows)
    singles.to_csv(OUTPUT_DIR / "trpb.csv", index=False)
    print(
        f"Singles: {len(singles)} variants, "
        f"{singles['position'].nunique()} positions, "
        f"range [{singles['fitness'].min():.3f}, {singles['fitness'].max():.3f}]"
    )


def write_wt_fasta(wt_seq: str) -> None:
    fasta_path = OUTPUT_DIR / "trpb_wt.fasta"
    header = (
        ">TrpB_Tm9D8 Thermotoga maritima TrpB engineered variant "
        "(parent for Johnston 2024 4-site landscape)"
    )
    fasta_path.write_text(f"{header}\n{wt_seq}\n")
    print(f"WT FASTA: {len(wt_seq)} residues -> {fasta_path.name}")


def prepare_contact_map(wt_seq: str) -> None:
    cm = ContactMap.from_pdb(PDB_PATH, threshold=8.0, chain_id="A")
    print(
        f"PDB contact map: {cm.n_positions} positions, {cm.n_contacts} contacts, "
        f"residue range {min(cm.positions)}-{max(cm.positions)}"
    )
    np.savez(
        OUTPUT_DIR / "trpb_distances.npz",
        distances=cm.distances,
        positions=np.array(cm.positions),
    )
    # Also print the 4-site submap in case the user wants to sanity-check.
    sub = cm.submap(list(MUTATED_POSITIONS))
    print(
        f"Four-site submap: {sub.n_contacts} contacts out of {4 * 3 // 2} possible pairs"
    )
    for i, p in enumerate(sub.positions):
        for q in sub.positions[i + 1 :]:
            d = sub.distances[sub.positions.index(p), sub.positions.index(q)]
            print(f"  {p}-{q}: {d:.2f} Å  contact={d <= 8.0}")


if __name__ == "__main__":
    assert SCALED_CSV.exists(), (
        f"expected {SCALED_CSV.name} to be in place. "
        "Extract it from the SSMuLA Zenodo bundle at "
        "https://zenodo.org/records/13910506 (path data/TrpB/scale2max/TrpB4.csv)."
    )
    assert PDB_PATH.exists(), (
        f"expected {PDB_PATH.name} to be in place. "
        "Download from https://files.rcsb.org/download/8VHH.pdb."
    )
    wt_seq = _extract_wt_sequence()
    print(f"WT length: {len(wt_seq)}, positions of interest: "
          + ", ".join(f"{p}={wt_seq[p - 1]}" for p in MUTATED_POSITIONS))
    write_wt_fasta(wt_seq)
    prepare_dms()
    prepare_contact_map(wt_seq)
    print("Done.")
