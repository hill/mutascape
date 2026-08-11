"""Noise-sigma sweep: vary i.i.d. Gaussian noise, measure walsh_L1 + fitness_W1
plus topology and distribution attributes, per protein.

The headline calibration path (additive + pairwise contacts + censored global epistasis
sigmoid) is fixed; only `noise_sigma` varies. Produces the CSV that feeds the
paper's primary figure.

Run:
    uv run python scripts/run_noise_sweep.py --protein gb1
    uv run python scripts/run_noise_sweep.py --protein trpb
"""

from __future__ import annotations

import argparse

import pandas as pd

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
from mutascape.validation.reference_configs import (
    GB1_HEADLINE,
    TRPB_HEADLINE,
    ReferenceConfig,
)

NOISE_SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
PROTEIN_CONFIG: dict[str, ReferenceConfig] = {
    "gb1": GB1_HEADLINE,
    "trpb": TRPB_HEADLINE,
}


def _real_tensor_and_walsh(cfg: ReferenceConfig):
    real_df = cfg.load_real_dataframe()
    alphabets = observed_alphabets(
        real_df,
        cfg.positions,
        cfg.wt_aa_at_position,
    )
    real_tensor, _coverage = fitness_tensor_from_real(
        real_df,
        cfg.positions,
        cfg.wt_aa_at_position,
        alphabets,
    )
    return real_tensor, walsh_decompose(real_tensor), alphabets


def _eval_noise(noise_sigma, cfg, g, real_tensor, real_walsh, alphabets):
    ls = cfg.build_landscape(
        noise_sigma=noise_sigma,
        nonlinearity=g,
    )
    t = fitness_tensor_from_landscape(ls, cfg.positions, alphabets)
    w = walsh_decompose(t)
    o = local_optima_from_tensor(t)
    p = accessible_paths(t, n_sources=3000, n_rsign_samples=20000, seed=cfg.seed)
    ss = ssmula_attributes(t)
    return {
        "noise_sigma": noise_sigma,
        "walsh_1": w[1],
        "walsh_2": w[2],
        "walsh_3": w[3],
        "walsh_4": w[4],
        "walsh_L1": walsh_l1_distance(real_walsh, w),
        "fitness_W1": fitness_wasserstein(real_tensor, t),
        "n_optima": o["n_optima"],
        "peak_fraction": o["peak_fraction"],
        "largest_basin_fraction": o["largest_basin_fraction"],
        "basin_entropy": o["basin_entropy"],
        "accessible_frac": p["mean_accessible_fraction"],
        "ssmula_frac_active": ss["frac_active"],
        "ssmula_frac_optima": ss["frac_local_optima"],
        "ssmula_kurtosis": ss["kurtosis"],
        "ssmula_kde_peaks": ss["kde_peak_count"],
    }


def _real_row(real_tensor, real_walsh, seed: int):
    o = local_optima_from_tensor(real_tensor)
    p = accessible_paths(
        real_tensor,
        n_sources=3000,
        n_rsign_samples=20000,
        seed=seed,
    )
    ss = ssmula_attributes(real_tensor)
    return {
        "noise_sigma": float("nan"),
        "walsh_1": real_walsh[1],
        "walsh_2": real_walsh[2],
        "walsh_3": real_walsh[3],
        "walsh_4": real_walsh[4],
        "walsh_L1": float("nan"),
        "fitness_W1": float("nan"),
        "n_optima": o["n_optima"],
        "peak_fraction": o["peak_fraction"],
        "largest_basin_fraction": o["largest_basin_fraction"],
        "basin_entropy": o["basin_entropy"],
        "accessible_frac": p["mean_accessible_fraction"],
        "ssmula_frac_active": ss["frac_active"],
        "ssmula_frac_optima": ss["frac_local_optima"],
        "ssmula_kurtosis": ss["kurtosis"],
        "ssmula_kde_peaks": ss["kde_peak_count"],
    }


def run_sweep(protein: str, noise_sigmas=NOISE_SIGMAS) -> pd.DataFrame:
    cfg = PROTEIN_CONFIG[protein]
    g = cfg.fit_nonlinearity()
    real_tensor, real_walsh, alphabets = _real_tensor_and_walsh(cfg)

    rows = [{"sweep": "REAL", **_real_row(real_tensor, real_walsh, cfg.seed)}]
    for sigma in noise_sigmas:
        print(f"  noise_sigma={sigma}", flush=True)
        row = _eval_noise(sigma, cfg, g, real_tensor, real_walsh, alphabets)
        rows.append({"sweep": f"noise={sigma}", **row})

    df = pd.DataFrame(rows)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"{protein}_noise_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protein", required=True, choices=sorted(PROTEIN_CONFIG)
    )
    args = parser.parse_args()
    df = run_sweep(args.protein)
    print()
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
