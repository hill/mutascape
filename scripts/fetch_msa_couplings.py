"""Fetch UniProt homologs, pairwise-align to WT, and compute log-odds couplings
at the focal DMS positions. Output: .npz with (L, L, 20, 20) coupling_matrix and
(L,) positions, consumable by MSACouplingsEpistasis.from_npz.

Usage:
    uv run python scripts/fetch_msa_couplings.py --protein gb1
    uv run python scripts/fetch_msa_couplings.py --protein trpb

Needs internet. Falls back gracefully if UniProt unreachable.
"""

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
        "whole_protein": False,
    },
    "trpb": {
        "wt_fasta": REPO_ROOT / "data/trpb/trpb_wt.fasta",
        "positions": (183, 184, 227, 228),
        "pfam": "PF00291",
        "out_npz": REPO_ROOT / "data/trpb/trpb_msa_couplings.npz",
        "page_size": 500,
        "n_pages": 4,
        "whole_protein": False,
    },
    "creilov": {
        # PF00785 (PAS_2) is deprecated. PF00989 (PAS) is the canonical Pfam
        # for the LOV/PAS fold; pairwise-alignment to WT filters down to the
        # actual LOV1-similar homologues.
        "wt_fasta": REPO_ROOT / "data/creilov/creilov_wt.fasta",
        "positions": None,
        "pfam": "PF00989",
        "out_npz": REPO_ROOT / "data/creilov/creilov_msa_couplings.npz",
        "page_size": 500,
        "n_pages": 8,
        "whole_protein": True,
    },
    "gfp": {
        "wt_fasta": REPO_ROOT / "data/gfp/gfp_wt.fasta",
        "positions": None,
        "pfam": "PF01353",
        "out_npz": REPO_ROOT / "data/gfp/gfp_msa_couplings.npz",
        "page_size": 500,
        "n_pages": 8,
        "whole_protein": True,
    },
    # tem1 data lives in mutantopt/experiments/datasets/tem1/.
    # Wire it after copying or symlinking the WT FASTA.
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
        if 'rel="next"' in link:
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
    focal_seqs: list[tuple[str, ...]],
    positions: tuple[int, ...],
    pseudocount: float = 0.5,
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


def hamming_reweighting(idx_matrix: np.ndarray, identity_threshold: float = 0.8) -> np.ndarray:
    """Compute per-sequence weights to downweight phylogenetic redundancy.

    Each sequence's weight is 1 / (cluster size at identity_threshold). This is
    the standard plmDCA reweighting (Morcos et al. 2011; Ekeberg et al. 2013).
    """
    n_seqs, L = idx_matrix.shape
    distance_threshold = 1.0 - identity_threshold
    cluster_sizes = np.zeros(n_seqs, dtype=np.float64)
    # Chunks limit memory use. Total work is O(N^2 L).
    chunk = max(1, min(256, n_seqs))
    for start in range(0, n_seqs, chunk):
        block = idx_matrix[start : start + chunk]
        # broadcast Hamming distance (block_size, n_seqs, L) -> (block_size, n_seqs)
        diff_frac = (block[:, None, :] != idx_matrix[None, :, :]).mean(axis=2)
        cluster_sizes[start : start + chunk] = (diff_frac < distance_threshold).sum(axis=1)
    weights = 1.0 / cluster_sizes
    return weights


def zero_sum_gauge(J: np.ndarray) -> np.ndarray:
    """Project couplings into the zero-sum (Ising) gauge.

    For each position-pair (i, j), shift the 20x20 sub-matrix so that row sums
    and column sums are zero. This removes the additive single-site component
    that would otherwise be colinear with the DMS-anchored fields h_i.
    """
    row_mean = J.mean(axis=-2, keepdims=True)
    col_mean = J.mean(axis=-1, keepdims=True)
    grand_mean = J.mean(axis=(-2, -1), keepdims=True)
    return J - row_mean - col_mean + grand_mean


def compute_couplings_reweighted(
    idx_matrix: np.ndarray,
    gap_mask: np.ndarray | None = None,
    pseudocount: float = 0.5,
    identity_threshold: float = 0.8,
    apply_gauge: bool = True,
) -> tuple[np.ndarray, float]:
    """Whole-protein log-odds couplings with Hamming reweighting and zero-sum gauge.

    Args:
        idx_matrix: (n_seqs, L) integer matrix of amino acid indices (0..K-1
            for valid AAs; gaps may share any sentinel value because
            gap_mask separately tracks validity).
        gap_mask: (n_seqs, L) bool where True means the AA is valid (not a
            gap or non-standard letter). When None, all positions are valid.
            Used to mask sequences out of marginals and joints per-position
            and per-pair, so a homologue with an indel at position 30 still
            contributes to marginals and joints at every other position.
        pseudocount: Laplace smoothing.
        identity_threshold: cluster sequences above this fraction-identity.
        apply_gauge: project J into the zero-sum gauge after computation.

    Returns:
        (J, n_eff) where J has shape (L, L, K, K) and n_eff is the sum of
        sequence weights after reweighting.
    """
    n_seqs, L = idx_matrix.shape
    K = len(ALPHABET)
    if gap_mask is None:
        gap_mask = np.ones((n_seqs, L), dtype=bool)

    weights = hamming_reweighting(idx_matrix, identity_threshold=identity_threshold)
    n_eff = float(weights.sum())

    marginals = np.zeros((L, K), dtype=np.float64) + pseudocount
    for i in range(L):
        valid_i = gap_mask[:, i]
        if valid_i.any():
            np.add.at(marginals[i], idx_matrix[valid_i, i], weights[valid_i])
    marginals /= marginals.sum(axis=1, keepdims=True)
    log_marg = np.log(marginals)

    J = np.zeros((L, L, K, K), dtype=np.float64)
    for i in range(L):
        valid_i = gap_mask[:, i]
        for j in range(i + 1, L):
            valid_ij = valid_i & gap_mask[:, j]
            joint = np.full((K, K), pseudocount / K, dtype=np.float64)
            if valid_ij.any():
                np.add.at(
                    joint, (idx_matrix[valid_ij, i], idx_matrix[valid_ij, j]), weights[valid_ij]
                )
            joint /= joint.sum()
            lo = np.log(joint) - log_marg[i][:, None] - log_marg[j][None, :]
            J[i, j] = lo
            J[j, i] = lo.T

    if apply_gauge:
        J = zero_sum_gauge(J)
    return J, n_eff


def extract_whole_protein_aas(
    homolog: SeqRecord, wt_seq: str, aligner: PairwiseAligner
) -> tuple[str, ...] | None:
    """Return the AA the homolog has at every WT residue, or None if the
    alignment is too weak. Gaps / non-standard AAs use a sentinel index handled
    at the row-keep stage downstream.
    """
    try:
        alignment = aligner.align(wt_seq, str(homolog.seq).replace("*", ""))[0]
    except Exception:
        return None
    wt_aligned, hom_aligned = str(alignment[0]), str(alignment[1])
    wt_idx = 0
    wt_idx_to_col: dict[int, int] = {}
    for col, c in enumerate(wt_aligned):
        if c != "-":
            wt_idx_to_col[wt_idx] = col
            wt_idx += 1
    result: list[str] = []
    for idx in range(len(wt_seq)):
        col = wt_idx_to_col.get(idx)
        if col is None:
            return None
        result.append(hom_aligned[col])
    return tuple(result)


def _build_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    return aligner


def _run_focal(cfg) -> None:
    wt_seq = str(next(SeqIO.parse(str(cfg["wt_fasta"]), "fasta")).seq)
    print(f"WT length: {len(wt_seq)}")
    print(f"Focal positions (1-based): {cfg['positions']}")
    print(f"Focal WT residues: {''.join(wt_seq[p - 1] for p in cfg['positions'])}")

    print(f"fetching Pfam {cfg['pfam']} homologs from UniProt ...")
    homologs = fetch_homologs(cfg["pfam"], cfg["page_size"], cfg["n_pages"])
    print(f"  total fetched: {len(homologs)}")

    aligner = _build_aligner()
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

    for i in range(len(cfg["positions"])):
        for j in range(i + 1, len(cfg["positions"])):
            abs_mean = np.abs(J[i, j]).mean()
            print(f"  mean |J[{cfg['positions'][i]},{cfg['positions'][j]}]| = {abs_mean:.3f}")


def _run_whole_protein(cfg) -> None:
    wt_seq = str(next(SeqIO.parse(str(cfg["wt_fasta"]), "fasta")).seq)
    L = len(wt_seq)
    print(f"WT length: {L}")
    positions = tuple(range(1, L + 1))  # 1-based residue numbers

    print(f"fetching Pfam {cfg['pfam']} homologs from UniProt ...")
    homologs = fetch_homologs(cfg["pfam"], cfg["page_size"], cfg["n_pages"])
    print(f"  total fetched: {len(homologs)}")

    aligner = _build_aligner()
    aligned_rows: list[tuple[str, ...]] = []
    for k, rec in enumerate(homologs):
        res = extract_whole_protein_aas(rec, wt_seq, aligner)
        if res is not None:
            aligned_rows.append(res)
        if (k + 1) % 100 == 0:
            print(f"  aligned {k + 1}/{len(homologs)}, kept {len(aligned_rows)}")
    print(f"Usable aligned sequences (before column filtering): {len(aligned_rows)}")

    # Anchor WT in the alignment.
    aligned_rows.append(tuple(wt_seq))

    # Drop sequences whose alignment to WT is too sparse (e.g. >50% gaps)
    # but allow partial coverage; per-pair gap masking handles the rest.
    coverage_threshold = 0.5
    keep_seqs = []
    for row in aligned_rows:
        valid = sum(1 for c in row if c in AA_TO_IDX)
        if valid / len(row) >= coverage_threshold:
            keep_seqs.append(row)
    n_kept = len(keep_seqs)
    print(f"Sequences with ≥{coverage_threshold:.0%} AA coverage: {n_kept}")
    if n_kept < 100:
        print("WARNING: fewer than 100 usable sequences; couplings will be noisy")

    GAP_IDX = len(ALPHABET)  # sentinel for hamming-distance purposes
    idx_matrix = np.full((n_kept, L), GAP_IDX, dtype=np.int64)
    gap_mask = np.zeros((n_kept, L), dtype=bool)
    for s, row in enumerate(keep_seqs):
        for c, aa in enumerate(row):
            if aa in AA_TO_IDX:
                idx_matrix[s, c] = AA_TO_IDX[aa]
                gap_mask[s, c] = True

    avg_coverage = float(gap_mask.mean())
    print(f"  average per-position coverage: {avg_coverage:.2%}")

    print("Computing reweighted, gauge-fixed couplings (this can take a few minutes for L>200)...")
    J, n_eff = compute_couplings_reweighted(
        idx_matrix,
        gap_mask=gap_mask,
        pseudocount=0.5,
        identity_threshold=0.8,
        apply_gauge=True,
    )
    print(f"  N_eff (sum of weights): {n_eff:.1f}")
    abs_mean = float(np.abs(J).mean())
    abs_max = float(np.abs(J).max())
    print(f"  mean |J| = {abs_mean:.4f}, max |J| = {abs_max:.4f}")

    cfg["out_npz"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg["out_npz"],
        coupling_matrix=J,
        positions=np.array(positions, dtype=np.int64),
        n_aligned=np.array([n_kept]),
        n_effective=np.array([n_eff]),
        wt_seq=wt_seq,
        identity_threshold=np.array([0.8]),
        gauge="zero_sum",
    )
    print(f"wrote {cfg['out_npz']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein", required=True, choices=sorted(PROTEIN_CONFIG))
    args = parser.parse_args()

    cfg = PROTEIN_CONFIG[args.protein]
    if cfg.get("whole_protein", False):
        _run_whole_protein(cfg)
    else:
        _run_focal(cfg)


if __name__ == "__main__":
    main()
