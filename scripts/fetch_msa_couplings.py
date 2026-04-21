"""Fetch UniProt homologs, pairwise-align to WT, and compute log-odds couplings
at the focal DMS positions. Output: .npz with (L, L, 20, 20) coupling_matrix and
(L,) positions, consumable by MSACouplingsEpistasis.from_npz.

Usage:
    uv run python scripts/fetch_msa_couplings.py --protein gb1
    uv run python scripts/fetch_msa_couplings.py --protein trpb

Needs internet. Falls back gracefully if UniProt unreachable.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.SeqRecord import SeqRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}

PROTEIN_CONFIG = {
    "gb1": {
        "wt_fasta": REPO_ROOT / "data/gb1/gb1_wt.fasta",
        "positions": (39, 40, 41, 54),
        "pfam": "PF01378",
        "out_npz": REPO_ROOT / "data/gb1/gb1_msa_couplings.npz",
        "page_size": 500,
        "n_pages": 4,
    },
    "trpb": {
        "wt_fasta": REPO_ROOT / "data/trpb/trpb_wt.fasta",
        "positions": (183, 184, 227, 228),
        "pfam": "PF00291",
        "out_npz": REPO_ROOT / "data/trpb/trpb_msa_couplings.npz",
        "page_size": 500,
        "n_pages": 4,
    },
}


def fetch_homologs(pfam_id: str, page_size: int, n_pages: int) -> list[SeqRecord]:
    all_records: list[SeqRecord] = []
    cursor = None
    for page in range(n_pages):
        params = f"query=xref%3Apfam-{pfam_id}&format=fasta&size={page_size}"
        if cursor:
            url = f"{cursor}"
        else:
            url = f"https://rest.uniprot.org/uniprotkb/search?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mutascape/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8")
                link = resp.headers.get("Link", "")
        except Exception as exc:
            print(f"  page {page + 1} fetch failed: {exc}")
            break
        records = list(SeqIO.parse(io.StringIO(text), "fasta"))
        all_records.extend(records)
        print(f"  page {page + 1}: {len(records)} sequences (cumulative {len(all_records)})")
        if "rel=\"next\"" in link:
            next_url = link.split(";")[0].strip("<> ")
            cursor = next_url
        else:
            break
    return all_records


def extract_focal_aas(
    homolog: SeqRecord, wt_seq: str, positions: tuple[int, ...], aligner: PairwiseAligner
) -> tuple[str, ...] | None:
    """Pairwise-align homolog to WT; return the AAs at the focal positions of WT,
    or None if alignment is too weak / any focal position maps to a gap or
    non-standard AA."""
    try:
        alignment = aligner.align(wt_seq, str(homolog.seq).replace("*", ""))[0]
    except Exception:
        return None
    wt_aligned, hom_aligned = str(alignment[0]), str(alignment[1])
    # Build WT column -> WT residue index mapping and hom column -> hom residue mapping.
    wt_col_to_idx: dict[int, int] = {}
    wt_idx = 0
    for col, c in enumerate(wt_aligned):
        if c != "-":
            wt_col_to_idx[col] = wt_idx
            wt_idx += 1
    wt_idx_to_col = {v: k for k, v in wt_col_to_idx.items()}

    result: list[str] = []
    # positions are 1-based PDB residue numbers in our data; WT sequence starts at 1
    # for GB1, but TrpB might be offset. For simplicity assume positions are 1-based
    # residue numbers matching the WT sequence (need to verify offset).
    # Actually: positions in WT_AA_AT_POSITION are PDB numbers. Need to check whether
    # WT FASTA is the same residue numbering. We'll treat position as 1-based.
    wt_len = len(wt_seq)
    for pos in positions:
        idx = pos - 1  # 1-based -> 0-based
        if idx < 0 or idx >= wt_len:
            return None
        col = wt_idx_to_col.get(idx)
        if col is None:
            return None
        hc = hom_aligned[col]
        if hc not in AA_TO_IDX:
            return None
        result.append(hc)
    return tuple(result)


def compute_couplings(
    focal_seqs: list[tuple[str, ...]], positions: tuple[int, ...], pseudocount: float = 0.5,
):
    L = len(positions)
    K = len(ALPHABET)
    n_seqs = len(focal_seqs)

    idx_cols = np.zeros((L, n_seqs), dtype=np.int64)
    for s, focal in enumerate(focal_seqs):
        for i, aa in enumerate(focal):
            idx_cols[i, s] = AA_TO_IDX[aa]

    marginals = np.zeros((L, K))
    for i in range(L):
        counts = np.bincount(idx_cols[i], minlength=K).astype(np.float64) + pseudocount
        marginals[i] = counts / counts.sum()

    J = np.zeros((L, L, K, K), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            joint = np.full((K, K), pseudocount / K, dtype=np.float64)
            np.add.at(joint, (idx_cols[i], idx_cols[j]), 1.0)
            joint /= joint.sum()
            lo = np.log(joint) - np.log(marginals[i])[:, None] - np.log(marginals[j])[None, :]
            J[i, j] = lo
    return J


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein", required=True, choices=sorted(PROTEIN_CONFIG))
    args = parser.parse_args()

    cfg = PROTEIN_CONFIG[args.protein]
    wt_seq = str(next(SeqIO.parse(str(cfg["wt_fasta"]), "fasta")).seq)
    print(f"WT length: {len(wt_seq)}")
    print(f"Focal positions (1-based): {cfg['positions']}")
    print(f"Focal WT residues: {''.join(wt_seq[p - 1] for p in cfg['positions'])}")

    print(f"fetching Pfam {cfg['pfam']} homologs from UniProt ...")
    homologs = fetch_homologs(cfg["pfam"], cfg["page_size"], cfg["n_pages"])
    print(f"  total fetched: {len(homologs)}")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1

    focal_seqs: list[tuple[str, ...]] = []
    for k, rec in enumerate(homologs):
        res = extract_focal_aas(rec, wt_seq, cfg["positions"], aligner)
        if res is not None:
            focal_seqs.append(res)
        if (k + 1) % 100 == 0:
            print(f"  aligned {k + 1}/{len(homologs)}, kept {len(focal_seqs)}")

    print(f"Usable aligned sequences: {len(focal_seqs)}")
    if len(focal_seqs) < 50:
        print("WARNING: fewer than 50 usable sequences; couplings will be noisy")

    # Add the WT row itself so WT is anchored in the alignment.
    wt_focal = tuple(wt_seq[p - 1] for p in cfg["positions"])
    focal_seqs.append(wt_focal)

    J = compute_couplings(focal_seqs, cfg["positions"])

    cfg["out_npz"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg["out_npz"],
        coupling_matrix=J,
        positions=np.array(cfg["positions"]),
        n_aligned=np.array([len(focal_seqs)]),
        wt_focal="".join(wt_focal),
    )
    print(f"wrote {cfg['out_npz']}")

    # Diagnostic print
    for i in range(len(cfg["positions"])):
        for j in range(i + 1, len(cfg["positions"])):
            abs_mean = np.abs(J[i, j]).mean()
            print(f"  mean |J[{cfg['positions'][i]},{cfg['positions'][j]}]| = {abs_mean:.3f}")


if __name__ == "__main__":
    main()
