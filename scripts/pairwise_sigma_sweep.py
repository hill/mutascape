"""Sweep the RandomEpistasis sigma at fixed noise=0.3 to confirm the finding
that random pairwise couplings worsen the Walsh spectrum match.

Run: uv run python scripts/pairwise_sigma_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
from mutascape.validation.metrics import accessible_paths, local_optima_from_tensor
from mutascape.validation.real_data import (
    load_gb1_combinatorial,
    load_single_site_effects,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
NOISE_SIGMA = 0.3
SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)

PROTEIN_CONFIG = {
    "gb1": {
        "dms_csv": REPO_ROOT / "data/gb1/gb1.csv",
        "distances_npz": REPO_ROOT / "data/gb1/gb1_distances.npz",
        "wt_fasta": REPO_ROOT / "data/gb1/gb1_wt.fasta",
        "combinatorial_csv": REPO_ROOT / "data/gb1/gb1_combinatorial.csv",
        "positions": (39, 40, 41, 54),
        "wt_aa_at_position": {39: "V", 40: "D", 41: "G", 54: "V"},
        "alpha": 1.5,
        "upside": 3.0,
        "scale_override": 1.0,
        "floor_hard": -1.0,
        "output_dir": REPO_ROOT / "notebooks/validation/output_gb1",
    },
    "trpb": {
        "dms_csv": REPO_ROOT / "data/trpb/trpb.csv",
        "distances_npz": REPO_ROOT / "data/trpb/trpb_distances.npz",
        "wt_fasta": REPO_ROOT / "data/trpb/trpb_wt.fasta",
        "combinatorial_csv": REPO_ROOT / "data/trpb/trpb_combinatorial.csv",
        "positions": (183, 184, 227, 228),
        "wt_aa_at_position": {183: "V", 184: "F", 227: "V", 228: "S"},
        "alpha": 1.5,
        "upside": 3.0,
        "scale_override": None,
        "floor_hard": None,
        "output_dir": REPO_ROOT / "notebooks/validation/output_trpb",
    },
}


def _fit_g(cfg):
    singles = load_single_site_effects(cfg["dms_csv"])
    base = GlobalEpistasisSigmoid.fit_from_singles(singles, combinatorial_upside=cfg["upside"])
    scale = base.scale if cfg["scale_override"] is None else float(cfg["scale_override"])
    fh = cfg["floor_hard"]
    if fh is None:
        fh = float(min(v for p in singles.values() for v in p.values()))
    return GlobalEpistasisSigmoid(
        floor=base.floor, ceiling=base.ceiling, scale=scale, floor_hard=fh
    )


def _real(cfg):
    real_df = load_gb1_combinatorial(
        cfg["combinatorial_csv"],
        cfg["dms_csv"],
        positions=cfg["positions"],
    )
    alphabets = observed_alphabets(real_df, cfg["positions"], cfg["wt_aa_at_position"])
    t, _ = fitness_tensor_from_real(
        real_df,
        cfg["positions"],
        cfg["wt_aa_at_position"],
        alphabets,
    )
    return t, walsh_decompose(t), alphabets


def run(protein):
    cfg = PROTEIN_CONFIG[protein]
    g = _fit_g(cfg)
    real_tensor, real_walsh, alphabets = _real(cfg)
    rows = [
        {
            "sigma": float("nan"),
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
            "config": "REAL",
        }
    ]
    for sigma in SIGMAS:
        ls = build_landscape(
            cfg["dms_csv"],
            cfg["distances_npz"],
            cfg["wt_fasta"],
            alpha=cfg["alpha"],
            sigma=sigma,
            seed=SEED,
            nonlinearity=g,
            noise_sigma=NOISE_SIGMA,
        )
        t = fitness_tensor_from_landscape(ls, cfg["positions"], alphabets)
        w = walsh_decompose(t)
        o = local_optima_from_tensor(t)
        p = accessible_paths(t, n_sources=3000, n_rsign_samples=20000, seed=SEED)
        rows.append(
            {
                "sigma": sigma,
                "walsh_1": w[1],
                "walsh_2": w[2],
                "walsh_3": w[3],
                "walsh_4": w[4],
                "walsh_L1": walsh_l1_distance(real_walsh, w),
                "fitness_W1": fitness_wasserstein(real_tensor, t),
                "n_optima": o["n_optima"],
                "accessible_frac": p["mean_accessible_fraction"],
                "config": f"random_sigma={sigma}",
            }
        )
    df = pd.DataFrame(rows)
    out = cfg["output_dir"] / f"{protein}_pairwise_sigma_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n=== {protein.upper()} ===")
    print(df.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    for p in ("gb1", "trpb"):
        run(p)
