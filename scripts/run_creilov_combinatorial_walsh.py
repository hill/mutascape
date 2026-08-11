"""Walsh-Hadamard validation on the Chen 2023 CreiLOV combinatorial library.

Per the 2024-2026 literature scan, the Walsh-Hadamard transform is the
canonical spectral validation for synthetic combinatorial landscapes
(it asks: does the synthetic landscape distribute fitness variance across
interaction orders the same way real data does?). This script applies it
to the Chen et al. 2023 CreiLOV SOLD library.

The Chen library is a heterogeneous product space (positions have 2-5
allowed AAs each; total 2x2x2x2x3x3x2x5x2x2x2x2x2x2x2 = 184,320 cells).
For a clean Walsh-Hadamard, we project the library to a 15-bit binary
cube (WT=0, mut=1 per position), aggregating measured variants per cell.
Multi-AA positions like Q47 (5 alphabet) collapse to "mutated or not".
This loses the per-AA distinction but preserves the position-by-position
interaction structure, which is what the Walsh decomposition reads.

For each landscape (real Chen, the no-coupling baseline, and MSA-coupling
scales), we:
  - Compute the 2^15-cell tensor by aggregating measured variants per
    binary vector (cell value = mean fitness over variants matching).
  - Walsh-Hadamard transform.
  - Normalised variance per interaction order = sum of squared Walsh
    coefficients with that index weight.
  - L1 distance between real and synthetic spectra.

Outputs:
  notebooks/validation/output_creilov_combinatorial/creilov_walsh_summary.csv
  notebooks/validation/output_creilov_combinatorial/creilov_walsh_spectrum.{pdf,png}
"""

from __future__ import annotations

import re
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

CHEN_POSITIONS = (3, 4, 5, 7, 29, 34, 47, 60, 61, 92, 96, 98, 107, 109, 113)
L = len(CHEN_POSITIONS)  # 15
SEED = 42

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
MUTATION_RE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")


def parse_mutant(s: str) -> list[tuple[int, str, str]]:
    if pd.isna(s) or s == "wt":
        return []
    out = []
    for part in re.split(r"[;,]", s):
        m = MUTATION_RE.match(part.strip())
        if not m:
            raise ValueError(f"cannot parse {part!r}")
        out.append((int(m.group(2)), THREE_TO_ONE[m.group(1)], THREE_TO_ONE[m.group(3)]))
    return out


def binary_vector_for_variant(mutations: list[tuple[int, str, str]]) -> tuple[int, ...]:
    """Map a variant's mutations to a binary vector of length 15 (one bit per Chen position)."""
    mut_positions = {p for p, _wt, _mut in mutations}
    return tuple(1 if pos in mut_positions else 0 for pos in CHEN_POSITIONS)


def aggregate_to_binary_cube(df: pd.DataFrame, fitness_col: str) -> np.ndarray:
    """Mean fitness per cell of the 2^15 binary cube. NaN for empty cells."""
    n_cells = 2 ** L
    sums = np.zeros(n_cells, dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    for vec, fit in zip(df["binary_vec"], df[fitness_col]):
        idx = sum(b << i for i, b in enumerate(vec))
        sums[idx] += fit
        counts[idx] += 1
    out = np.full(n_cells, np.nan, dtype=np.float64)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out.reshape((2,) * L)


def fill_missing_with_mean(tensor: np.ndarray) -> np.ndarray:
    """Replace NaN cells with the grand mean of measured cells."""
    flat = tensor.ravel()
    mean = float(np.nanmean(flat))
    out = np.where(np.isnan(flat), mean, flat)
    return out.reshape(tensor.shape)


def walsh_hadamard(tensor: np.ndarray) -> np.ndarray:
    """Discrete Walsh-Hadamard transform on a binary hypercube (in-place style FWHT)."""
    a = tensor.copy().astype(np.float64)
    n_axes = a.ndim
    for axis in range(n_axes):
        # Move axis to front, do (a + b)/sqrt(2), (a - b)/sqrt(2), move back
        a = np.moveaxis(a, axis, 0)
        s = a.shape
        a = a.reshape((2, -1))
        x0, x1 = a[0].copy(), a[1].copy()
        a[0] = (x0 + x1) / np.sqrt(2)
        a[1] = (x0 - x1) / np.sqrt(2)
        a = a.reshape(s)
        a = np.moveaxis(a, 0, axis)
    return a


def variance_per_order(walsh: np.ndarray) -> dict[int, float]:
    """Sum of squared Walsh coefficients grouped by interaction order
    (Hamming weight of the index in binary). Order 0 is the constant; we
    discard it and normalise so that orders 1..L sum to 1.
    """
    L_local = walsh.ndim
    out: dict[int, float] = {k: 0.0 for k in range(L_local + 1)}
    for idx in product(*[range(2)] * L_local):
        order = sum(idx)
        out[order] += float(walsh[idx] ** 2)
    total_nonconst = sum(v for k, v in out.items() if k > 0)
    if total_nonconst > 0:
        return {k: out[k] / total_nonconst for k in range(1, L_local + 1)}
    return {k: 0.0 for k in range(1, L_local + 1)}


def chen_variant_binary_vec_list(df: pd.DataFrame) -> list[tuple[int, ...]]:
    out = []
    for mutations in df["mutations"]:
        out.append(binary_vector_for_variant(mutations))
    return out


def evaluate_landscape_on_variants(landscape, mutations_list: list[list[tuple[int, str, str]]]) -> np.ndarray:
    wt = landscape.wild_type
    out = np.empty(len(mutations_list), dtype=np.float64)
    for i, muts in enumerate(mutations_list):
        seq = dict(wt)
        for pos, _wt_aa, mut_aa in muts:
            seq[pos] = mut_aa
        out[i] = landscape.evaluate(seq)
    return out


def _build_v1():
    return build_landscape(
        DMS_CSV, DISTANCES_NPZ, WT_FASTA,
        alpha=1.5, sigma=0.0, seed=SEED,
        nonlinearity="fit", combinatorial_upside=3.0, noise_sigma=0.0,
    )


def _build_v2(msa_scale: float):
    eps = MSACouplingsEpistasis.from_npz(MSA_COUPLINGS_NPZ, scale=msa_scale)
    return build_landscape(
        DMS_CSV, DISTANCES_NPZ, WT_FASTA,
        alpha=1.5, seed=SEED,
        nonlinearity="fit", combinatorial_upside=3.0, noise_sigma=0.0,
        msa_couplings=eps,
    )


CONFIGS = [
    ("v1_no_couplings", None),
    ("v2_scale_0.02", 0.02),
    ("v2_scale_0.05", 0.05),
    ("v2_scale_0.10", 0.10),
    ("v2_scale_0.15", 0.15),
    ("v2_scale_0.20", 0.20),
    ("v2_scale_0.30", 0.30),
    ("v2_scale_0.50", 0.50),
    ("v2_scale_1.00", 1.00),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Chen 2023 combinatorial data ...")
    df = pd.read_excel(COMBINATORIAL_XLSX, sheet_name="Sheet1")
    wt_log = float(df[df["n_mut"] == 0]["mean_log"].iloc[0])
    df["real_fitness"] = df["mean_log"] - wt_log
    df["mutations"] = df["mutant"].apply(parse_mutant)
    df["binary_vec"] = chen_variant_binary_vec_list(df)
    print(f"  {len(df)} variants total")

    # Real Walsh spectrum
    print("Aggregating real data into binary 2^15 cube ...")
    real_tensor_raw = aggregate_to_binary_cube(df, "real_fitness")
    n_filled = int(np.sum(~np.isnan(real_tensor_raw)))
    print(f"  {n_filled}/{2**L} = {n_filled / 2**L:.1%} cells have measurements")
    real_tensor = fill_missing_with_mean(real_tensor_raw)
    real_walsh = walsh_hadamard(real_tensor)
    real_spectrum = variance_per_order(real_walsh)
    print(f"  Real spectrum (top 6 orders): "
          f"{', '.join(f'O{k}={real_spectrum[k]:.3f}' for k in range(1, 7))}")

    # For each Mutascape config, evaluate on all 184320 binary cells
    # (using a canonical AA assignment per cell). We pick, for each Chen
    # position, the alphabetically-first non-WT mutation in Chen's set. This
    # collapse loses the multi-AA distinction the binary projection already
    # decided to ignore.

    # Build canonical AA per (position, bit). For each Chen position, pick
    # the lexicographically-first non-WT AA among the alphabet Chen actually
    # measured (extracted from the singles in df).
    print("Determining canonical mutant AA per position ...")
    canonical_mut_aa: dict[int, str] = {}
    singles_df = df[df["n_mut"] == 1]
    for pos in CHEN_POSITIONS:
        muts_here = []
        for ms in singles_df["mutations"]:
            for p, wt, mut in ms:
                if p == pos:
                    muts_here.append(mut)
        canonical_mut_aa[pos] = sorted(set(muts_here))[0] if muts_here else "A"
    print(f"  canonical mutant AAs: {canonical_mut_aa}")

    # Enumerate the 2^15 = 32768 cells and build per-cell mutation list
    cells = list(product(*[(0, 1)] * L))
    print(f"  enumerating {len(cells)} binary cells, evaluating each landscape ...")
    cell_mutation_lists: list[list[tuple[int, str, str]]] = []
    wt_at_pos = {pos: None for pos in CHEN_POSITIONS}  # filled below
    # Get WT residue at each Chen position from the DMS file
    dms_df = pd.read_csv(DMS_CSV)
    for pos in CHEN_POSITIONS:
        row = dms_df[dms_df["position"] == pos].iloc[0]
        wt_at_pos[pos] = str(row["wildtype"])
    for vec in cells:
        muts = []
        for bit, pos in zip(vec, CHEN_POSITIONS):
            if bit == 1:
                muts.append((pos, wt_at_pos[pos], canonical_mut_aa[pos]))
        cell_mutation_lists.append(muts)

    rows: list[dict] = []
    spectra: dict[str, dict[int, float]] = {"real": real_spectrum}

    for label, scale in CONFIGS:
        print(f"Evaluating {label} ...")
        landscape = _build_v1() if scale is None else _build_v2(scale)
        synth_values = evaluate_landscape_on_variants(landscape, cell_mutation_lists)
        synth_tensor = synth_values.reshape((2,) * L)
        synth_walsh = walsh_hadamard(synth_tensor)
        synth_spectrum = variance_per_order(synth_walsh)
        spectra[label] = synth_spectrum
        l1 = sum(abs(synth_spectrum[k] - real_spectrum[k]) for k in real_spectrum)
        for k in real_spectrum:
            rows.append({
                "config": label,
                "scale": scale if scale is not None else 0.0,
                "order": k,
                "real_var_frac": real_spectrum[k],
                "synth_var_frac": synth_spectrum[k],
                "diff": synth_spectrum[k] - real_spectrum[k],
            })
        print(f"  L1_Walsh = {l1:.4f}, top orders: "
              f"{', '.join(f'O{k}={synth_spectrum[k]:.3f}' for k in range(1, 7))}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "creilov_walsh_summary.csv", index=False)
    print(f"\nwrote {OUTPUT_DIR}/creilov_walsh_summary.csv")

    # L1_Walsh per config
    print("\nL1_Walsh per config (lower is better):")
    l1_per_config = (
        summary.groupby("config")["diff"]
        .apply(lambda x: float(x.abs().sum()))
        .reset_index()
        .rename(columns={"diff": "L1_Walsh"})
    )
    print(l1_per_config.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.6), tight_layout=True)
    orders = sorted(real_spectrum.keys())
    width = 0.18
    colors = {"real": "#000000", "v1_no_couplings": "#d62728",
              "v2_scale_0.30": "#bcbd22", "v2_scale_0.50": "#17becf", "v2_scale_1.00": "#2ca02c"}
    labels = {
        "real": "real Chen 2023",
        "v1_no_couplings": "No pairwise couplings",
        "v2_scale_0.30": "MSA couplings, scale = 0.30",
        "v2_scale_0.50": "MSA couplings, scale = 0.50",
        "v2_scale_1.00": "MSA couplings, scale = 1.00",
    }
    x = np.arange(len(orders))
    for i, key in enumerate(["real"] + [c for c, _ in CONFIGS]):
        vals = [spectra[key][k] for k in orders]
        ax.bar(x + (i - 2) * width, vals, width, label=labels[key], color=colors[key], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"O{o}" for o in orders])
    ax.set_xlabel("Interaction order (Hamming weight)")
    ax.set_ylabel("Normalised Walsh variance fraction")
    ax.set_title("CreiLOV combinatorial -- Walsh-spectrum match (15-bit binary projection)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(OUTPUT_DIR / "creilov_walsh_spectrum.pdf")
    fig.savefig(OUTPUT_DIR / "creilov_walsh_spectrum.png", dpi=160)
    print(f"wrote {OUTPUT_DIR}/creilov_walsh_spectrum.pdf")


if __name__ == "__main__":
    main()
