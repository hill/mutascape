"""Matched-variant Walsh-Hadamard validation on Chen 2023 CreiLOV combinatorial.

This is the corrected version of the binary-projection Walsh validation.
Difference from `run_creilov_combinatorial_walsh.py`: instead of choosing a
canonical mutant AA per position and evaluating Mutascape on the 32,768
binary cube cells, we evaluate Mutascape on every one of Chen's 165,428
real variants USING THE EXACT SAME (position, AA) ASSIGNMENTS.

Both real and Mutascape values are then aggregated to the same 2^15 = 32,768
binary cube by averaging within each cell. This eliminates the
canonical-AA mismatch that made the first Walsh comparison uninterpretable.

The Walsh-spectrum comparison after this matching is meaningful: given
Chen's 15-position × specific-AA library, does Mutascape's variance
distribution by interaction order match the real measurement?

Outputs:
  notebooks/validation/output_creilov_combinatorial/creilov_walsh_matched_summary.csv
  notebooks/validation/output_creilov_combinatorial/creilov_walsh_matched_spectrum.{pdf,png}
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mutascape.epistasis import MSACouplingsEpistasis
from mutascape.validation.calibration import build_landscape
from mutascape.validation.fourier import (
    binary_cube_spectral_metrics,
    binary_cube_spectrum,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data/creilov"
OUTPUT_DIR = REPO_ROOT / "notebooks/validation/output_creilov_combinatorial"

DMS_CSV = DATA_DIR / "creilov.csv"
DISTANCES_NPZ = DATA_DIR / "creilov_distances.npz"
WT_FASTA = DATA_DIR / "creilov_wt.fasta"
MSA_COUPLINGS_NPZ = DATA_DIR / "creilov_msa_couplings.npz"
COMBINATORIAL_XLSX = DATA_DIR / "creilov_combinatorial_raw.xlsx"

CHEN_POSITIONS = (3, 4, 5, 7, 29, 34, 47, 60, 61, 92, 96, 98, 107, 109, 113)
L = len(CHEN_POSITIONS)
SEED = 42
# Order truncations for the robustness check. The Walsh tail is noise-dominated
# (real CreiLOV carries ~97% of its variance in orders 1-2), so the selected
# scale should be stable whether or not the tail is included.
TRUNCATIONS = (2, 3, 5, None)
CACHE_NPZ = OUTPUT_DIR / "creilov_walsh_matched_cache.npz"  # per-variant evals

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


def binary_index(mutations: list[tuple[int, str, str]]) -> int:
    """Map a variant to a binary cube index ∈ [0, 2^15)."""
    mut_positions = {p for p, _wt, _mut in mutations}
    idx = 0
    for i, pos in enumerate(CHEN_POSITIONS):
        if pos in mut_positions:
            idx |= 1 << i
    return idx


def evaluate_landscape_on_variants(landscape, mutations_list) -> np.ndarray:
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
    eps = MSACouplingsEpistasis.from_npz(MSA_COUPLINGS_NPZ, scale=msa_scale)
    return build_landscape(
        DMS_CSV,
        DISTANCES_NPZ,
        WT_FASTA,
        alpha=1.5,
        seed=SEED,
        nonlinearity="fit",
        combinatorial_upside=3.0,
        noise_sigma=0.0,
        msa_couplings=eps,
    )


CONFIGS = [
    ("v1_no_couplings", None),
    ("v2_scale_0.05", 0.05),
    ("v2_scale_0.10", 0.10),
    ("v2_scale_0.20", 0.20),
    ("v2_scale_0.50", 0.50),
    ("v2_scale_1.00", 1.00),
]


def _load_or_evaluate() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Per-variant real values, cell indices, and synth values per config.

    The 1M landscape evaluations are the expensive step, so they are cached to
    an npz. Delete the cache to force a fresh evaluation.
    """
    if CACHE_NPZ.exists():
        print(f"Loading cached evaluations from {CACHE_NPZ.name} ...")
        data = np.load(CACHE_NPZ)
        real_values = data["real_values"]
        cell_idx = data["cell_idx"]
        synth_by_config = {label: data[f"synth__{label}"] for label, _ in CONFIGS}
        print(f"  {len(real_values)} variants, {len(synth_by_config)} configs")
        return real_values, cell_idx, synth_by_config

    print("Loading Chen 2023 combinatorial data ...")
    df = pd.read_excel(COMBINATORIAL_XLSX, sheet_name="Sheet1")
    wt_log = float(df[df["n_mut"] == 0]["mean_log"].iloc[0])
    df["real_fitness"] = df["mean_log"] - wt_log
    df["mutations"] = df["mutant"].apply(parse_mutant)
    df["cell_idx"] = df["mutations"].apply(binary_index)
    print(f"  {len(df)} variants total")

    mutations_list = df["mutations"].tolist()
    real_values = df["real_fitness"].to_numpy()
    cell_idx = df["cell_idx"].to_numpy()

    synth_by_config: dict[str, np.ndarray] = {}
    for label, scale in CONFIGS:
        print(f"  evaluating {label} on {len(real_values)} Chen variants ...")
        landscape = _build_v1() if scale is None else _build_v2(scale)
        synth_by_config[label] = evaluate_landscape_on_variants(landscape, mutations_list)

    np.savez_compressed(
        CACHE_NPZ,
        real_values=real_values,
        cell_idx=cell_idx,
        **{f"synth__{label}": v for label, v in synth_by_config.items()},
    )
    print(f"  cached evaluations to {CACHE_NPZ.name}")
    return real_values, cell_idx, synth_by_config


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_values, cell_idx, synth_by_config = _load_or_evaluate()

    _, real_spectrum_full = binary_cube_spectrum(real_values, cell_idx, L)
    print(
        "\nReal spectrum (full data): "
        f"{', '.join(f'O{k}={real_spectrum_full[k]:.3f}' for k in range(1, 7))}"
    )

    # Full-data point estimates on the near-complete cube (99.6% coverage) are
    # the trustworthy answer. A variant-resampling bootstrap or held-out split is
    # NOT used: the per-order spectrum is coverage-dependent (fewer variants per
    # cell inflate the noise-driven high-order tail), so resampling biases the
    # metric and even reverses the scale ranking. Robustness is instead shown by
    # truncating the noisy tail to the signal-bearing orders.
    bar_spectra: dict[str, dict[int, float]] = {"real": real_spectrum_full}
    metrics_by_trunc: dict[int | None, dict[str, dict[str, float]]] = {}
    for max_order in TRUNCATIONS:
        metrics_by_trunc[max_order] = {
            label: binary_cube_spectral_metrics(
                real_values, synth_by_config[label], cell_idx, L, max_order=max_order
            )
            for label, _scale in CONFIGS
        }
    for label, _scale in CONFIGS:
        _, bar_spectra[label] = binary_cube_spectrum(synth_by_config[label], cell_idx, L)

    full_metrics = metrics_by_trunc[None]
    selected_label = min(CONFIGS, key=lambda c: full_metrics[c[0]]["order_w1"])[0]
    selected_scale = dict(CONFIGS)[selected_label] or 0.0

    # Truncation robustness: the scale minimising order_w1 at every truncation.
    print("\nScale selection robustness (argmin order_w1 by order truncation):")
    robust = True
    for max_order in TRUNCATIONS:
        best = min(CONFIGS, key=lambda c: metrics_by_trunc[max_order][c[0]]["order_w1"])[0]
        tag = "all" if max_order is None else f"O1-{max_order}"
        robust = robust and (best == selected_label)
        print(f"  {tag:<6} -> {best}")
    print(f"  selection stable across truncations: {robust}")

    rows: list[dict] = []
    for label, scale in CONFIGS:
        row = {
            "config": label,
            "scale": scale if scale is not None else 0.0,
            "selected": label == selected_label,
            "cos_order2": full_metrics[label]["cos_order2"],
            "cos_ge2": full_metrics[label]["cos_ge2"],
        }
        for max_order in TRUNCATIONS:
            tag = "all" if max_order is None else f"o{max_order}"
            row[f"order_w1_{tag}"] = metrics_by_trunc[max_order][label]["order_w1"]
            row[f"walsh_l1_{tag}"] = metrics_by_trunc[max_order][label]["walsh_l1"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "creilov_walsh_matched_summary.csv", index=False)
    print(f"\nwrote {OUTPUT_DIR}/creilov_walsh_matched_summary.csv")

    # Long-format per-order fractions on the full data, for the bar-chart panel.
    order_rows = []
    for label in ["real"] + [c for c, _ in CONFIGS]:
        for k in range(1, L + 1):
            order_rows.append({"config": label, "order": k, "var_frac": bar_spectra[label][k]})
    pd.DataFrame(order_rows).to_csv(OUTPUT_DIR / "creilov_walsh_matched_orders.csv", index=False)

    print(f"\nSelected scale = {selected_scale} ({selected_label}, min full-data order_w1)")
    print("\nper config: order_w1 (full), walsh_l1 (full), cos_order2 (full):")
    for _, r in summary.sort_values("order_w1_all").iterrows():
        star = " *" if r["selected"] else ""
        print(
            f"  {r['config']:<18} order_w1={r['order_w1_all']:.4f}  "
            f"walsh_l1={r['walsh_l1_all']:.4f}  "
            f"cos_order2={r['cos_order2']:+.3f}{star}"
        )

    _plot(bar_spectra, summary, selected_scale)


def _plot(bar_spectra, summary, selected_scale) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), tight_layout=True)

    # Panel 1: per-order variance-fraction bar chart (full data).
    orders = list(range(1, L + 1))
    width = 0.12
    colors = {
        "real": "#000000",
        "v1_no_couplings": "#d62728",
        "v2_scale_0.05": "#ff7f0e",
        "v2_scale_0.10": "#ffbb78",
        "v2_scale_0.20": "#bcbd22",
        "v2_scale_0.50": "#17becf",
        "v2_scale_1.00": "#2ca02c",
    }
    labels = {
        "real": "Chen et al. real",
        "v1_no_couplings": "No pairwise couplings",
        "v2_scale_0.05": "MSA couplings, scale = 0.05",
        "v2_scale_0.10": "MSA couplings, scale = 0.10",
        "v2_scale_0.20": "MSA couplings, scale = 0.20",
        "v2_scale_0.50": "MSA couplings, scale = 0.50",
        "v2_scale_1.00": "MSA couplings, scale = 1.00",
    }
    keys = ["real"] + [c for c, _ in CONFIGS]
    x = np.arange(len(orders))
    for i, key in enumerate(keys):
        vals = [bar_spectra[key][k] for k in orders]
        offset = (i - (len(keys) - 1) / 2) * width
        ax1.bar(x + offset, vals, width, label=labels[key], color=colors[key], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"O{o}" for o in orders])
    ax1.set_xlabel("Interaction order (Hamming weight on binary projection)")
    ax1.set_ylabel("Normalised Walsh variance fraction")
    ax1.set_title("Matched-variant Walsh spectrum")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2: per-order Wasserstein vs coupling scale, the metric that selects
    # the scale on CreiLOV, drawn at several order truncations to show the
    # selection is stable to the noise-dominated high-order tail. The selected
    # scale (min full-data order_w1) is marked.
    # (The order-2 cosine is reported in the CSV but is not the CreiLOV headline:
    # the binary projection averages over the 2-5 AAs Chen tested per position,
    # which washes out the AA-specific coupling pattern the cosine measures. The
    # cosine is the trustworthy structural test on complete-factorial libraries
    # such as GB1 and TrpB.)
    s = summary.sort_values("scale")
    scales = s["scale"].to_numpy()
    trunc_styles = [("o2", "O1-2"), ("o3", "O1-3"), ("o5", "O1-5"), ("all", "all orders")]
    for col, lbl in trunc_styles:
        ax2.plot(scales, s[f"order_w1_{col}"].to_numpy(), marker="o", label=lbl)
    ax2.axvline(
        selected_scale,
        color="#d62728",
        linestyle="--",
        label=f"selected scale = {selected_scale} (min order_w1)",
    )
    ax2.set_xlabel("MSA coupling scale")
    ax2.set_ylabel("Per-order Wasserstein distance (real vs synthetic)")
    ax2.set_title("Spectrum-budget match vs scale (lower is better)")
    ax2.legend(fontsize=8, loc="upper center", title="orders compared")
    ax2.grid(True, alpha=0.3)

    fig.savefig(OUTPUT_DIR / "creilov_walsh_matched_spectrum.pdf")
    fig.savefig(OUTPUT_DIR / "creilov_walsh_matched_spectrum.png", dpi=160)
    print(f"wrote {OUTPUT_DIR}/creilov_walsh_matched_spectrum.pdf")


if __name__ == "__main__":
    main()
