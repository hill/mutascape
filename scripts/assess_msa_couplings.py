"""Compare MSA-derived couplings vs RandomEpistasis on walsh_L1, fitness_W1,
per-order Walsh, and n_optima. For each protein, sweeps the MSA scale factor
at fixed noise_sigma=0.3 (the headline config) and reports whether
evolutionary couplings close the order-3 gap.

Run:
    uv run python scripts/assess_msa_couplings.py
"""

from pathlib import Path

import pandas as pd

from mutascape.epistasis import MSACouplingsEpistasis
from mutascape.nonlinearity import GlobalEpistasisSigmoid
from mutascape.validation.calibration import build_landscape
from mutascape.validation.comparisons import fitness_wasserstein
from mutascape.validation.fourier import (
    fitness_tensor_from_landscape,
    fitness_tensor_from_real,
    observed_alphabets,
    walsh_decompose,
    walsh_l1_distance,
)
from mutascape.validation.metrics import (
    accessible_paths,
    local_optima_from_tensor,
    ssmula_attributes,
)
from mutascape.validation.real_data import (
    load_gb1_combinatorial,
    load_single_site_effects,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42

PROTEIN_CONFIG = {
    "gb1": {
        "dms_csv": REPO_ROOT / "data/gb1/gb1.csv",
        "distances_npz": REPO_ROOT / "data/gb1/gb1_distances.npz",
        "wt_fasta": REPO_ROOT / "data/gb1/gb1_wt.fasta",
        "combinatorial_csv": REPO_ROOT / "data/gb1/gb1_combinatorial.csv",
        "msa_npz": REPO_ROOT / "data/gb1/gb1_msa_couplings.npz",
        "positions": (39, 40, 41, 54),
        "wt_aa_at_position": {39: "V", 40: "D", 41: "G", 54: "V"},
        "output_dir": REPO_ROOT / "notebooks/validation/output_gb1",
        "alpha": 1.5,
        "combinatorial_upside": 3.0,
        "scale_override": 1.0,
        "floor_hard": -1.0,
    },
    "trpb": {
        "dms_csv": REPO_ROOT / "data/trpb/trpb.csv",
        "distances_npz": REPO_ROOT / "data/trpb/trpb_distances.npz",
        "wt_fasta": REPO_ROOT / "data/trpb/trpb_wt.fasta",
        "combinatorial_csv": REPO_ROOT / "data/trpb/trpb_combinatorial.csv",
        "msa_npz": REPO_ROOT / "data/trpb/trpb_msa_couplings.npz",
        "positions": (183, 184, 227, 228),
        "wt_aa_at_position": {183: "V", 184: "F", 227: "V", 228: "S"},
        "output_dir": REPO_ROOT / "notebooks/validation/output_trpb",
        "alpha": 1.5,
        "combinatorial_upside": 3.0,
        "scale_override": None,
        "floor_hard": None,
    },
}

# MSA-coupling scale factors (multiplier on the raw log-odds). A range chosen so
# the effective pairwise contribution lives on a comparable scale to the
# RandomEpistasis sigma=0.3 baseline after applying alpha=1.5.
MSA_SCALES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
NOISE_SIGMA = 0.3  # headline fixed value


def _fit_censored_sigmoid(dms_csv, upside, scale_override, floor_hard):
    singles = load_single_site_effects(dms_csv)
    base = GlobalEpistasisSigmoid.fit_from_singles(singles, combinatorial_upside=upside)
    scale = base.scale if scale_override is None else float(scale_override)
    if floor_hard is None:
        floor_hard = float(min(v for pos in singles.values() for v in pos.values()))
    return GlobalEpistasisSigmoid(
        floor=base.floor,
        ceiling=base.ceiling,
        scale=scale,
        floor_hard=floor_hard,
    )


def _real_tensor_and_walsh(cfg):
    real_df = load_gb1_combinatorial(
        cfg["combinatorial_csv"],
        cfg["dms_csv"],
        positions=cfg["positions"],
    )
    alphabets = observed_alphabets(real_df, cfg["positions"], cfg["wt_aa_at_position"])
    real_tensor, _ = fitness_tensor_from_real(
        real_df,
        cfg["positions"],
        cfg["wt_aa_at_position"],
        alphabets,
    )
    return real_tensor, walsh_decompose(real_tensor), alphabets


def _metrics_row(label, ls, cfg, real_tensor, real_walsh, alphabets):
    t = fitness_tensor_from_landscape(ls, cfg["positions"], alphabets)
    w = walsh_decompose(t)
    o = local_optima_from_tensor(t)
    p = accessible_paths(t, n_sources=3000, n_rsign_samples=20000, seed=SEED)
    ss = ssmula_attributes(t)
    return {
        "config": label,
        "walsh_1": w[1],
        "walsh_2": w[2],
        "walsh_3": w[3],
        "walsh_4": w[4],
        "walsh_L1": walsh_l1_distance(real_walsh, w),
        "fitness_W1": fitness_wasserstein(real_tensor, t),
        "n_optima": o["n_optima"],
        "accessible_frac": p["mean_accessible_fraction"],
        "ssmula_kurtosis": ss["kurtosis"],
    }


def run(protein):
    cfg = PROTEIN_CONFIG[protein]
    print(f"\n=== {protein.upper()} ===")
    g = _fit_censored_sigmoid(
        cfg["dms_csv"],
        cfg["combinatorial_upside"],
        cfg["scale_override"],
        cfg["floor_hard"],
    )
    real_tensor, real_walsh, alphabets = _real_tensor_and_walsh(cfg)

    rows = []
    rows.append(
        {
            "config": "REAL",
            "walsh_1": real_walsh[1],
            "walsh_2": real_walsh[2],
            "walsh_3": real_walsh[3],
            "walsh_4": real_walsh[4],
            "walsh_L1": float("nan"),
            "fitness_W1": float("nan"),
            "n_optima": local_optima_from_tensor(real_tensor)["n_optima"],
            "accessible_frac": accessible_paths(
                real_tensor,
                n_sources=3000,
                n_rsign_samples=20000,
                seed=SEED,
            )["mean_accessible_fraction"],
            "ssmula_kurtosis": ssmula_attributes(real_tensor)["kurtosis"],
        }
    )

    # Baseline: RandomEpistasis at headline noise
    ls_rand = build_landscape(
        cfg["dms_csv"],
        cfg["distances_npz"],
        cfg["wt_fasta"],
        alpha=cfg["alpha"],
        seed=SEED,
        nonlinearity=g,
        noise_sigma=NOISE_SIGMA,
    )
    rows.append(
        _metrics_row(
            "random(sigma=0.3) + noise=0.3",
            ls_rand,
            cfg,
            real_tensor,
            real_walsh,
            alphabets,
        )
    )

    # MSA couplings at various scales
    for msa_scale in MSA_SCALES:
        msa = MSACouplingsEpistasis.from_npz(cfg["msa_npz"], scale=msa_scale)
        ls_msa = build_landscape(
            cfg["dms_csv"],
            cfg["distances_npz"],
            cfg["wt_fasta"],
            alpha=cfg["alpha"],
            seed=SEED,
            nonlinearity=g,
            noise_sigma=NOISE_SIGMA,
            msa_couplings=msa,
        )
        rows.append(
            _metrics_row(
                f"MSA(scale={msa_scale}) + noise=0.3",
                ls_msa,
                cfg,
                real_tensor,
                real_walsh,
                alphabets,
            )
        )

    df = pd.DataFrame(rows)
    out_path = cfg["output_dir"] / f"{protein}_msa_assessment.csv"
    df.to_csv(out_path, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nwrote {out_path}")
    return df


def main():
    run("gb1")
    run("trpb")


if __name__ == "__main__":
    main()
