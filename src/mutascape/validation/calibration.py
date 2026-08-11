"""Alpha calibration: find the epistatic strength that best matches real data."""

from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from mutascape import ContactMap, Landscape, RandomEpistasis, SignConsistentEpistasis
from mutascape.epistasis import (
    ContactNKEpistasis,
    EpistasisModel,
    MSACouplingsEpistasis,
)
from mutascape.nonlinearity import Identity, NonLinearity, GlobalEpistasisSigmoid

from .comparisons import composite_validation_score
from .metrics import compute_all_metrics
from .real_data import load_single_site_effects, sample_synthetic_data


def _resolve_nonlinearity_arg(
    nonlinearity,
    dms_csv: str | Path,
    *,
    scale: float | None = None,
    floor_hard: float | None = None,
    combinatorial_upside: float = 1.0,
) -> NonLinearity:
    """Normalise the `nonlinearity` argument used throughout this module.

    Supports: None (Identity), a NonLinearity instance (used as-is), the string
    "fit" (MLE-fit GlobalEpistasisSigmoid from the DMS singles), or a float
    (GlobalEpistasisSigmoid with that scale; floor/ceiling auto-set from singles).
    When fitting from singles, callers can override the fitted `scale`, inject
    a `floor_hard` censoring threshold, and set `combinatorial_upside`.
    """
    if nonlinearity is None:
        if scale is not None or floor_hard is not None or combinatorial_upside != 1.0:
            raise ValueError(
                "scale/floor_hard/combinatorial_upside require nonlinearity='fit' "
                "or an explicit GlobalEpistasisSigmoid instance"
            )
        return Identity()
    if isinstance(nonlinearity, NonLinearity):
        if scale is not None or floor_hard is not None or combinatorial_upside != 1.0:
            raise ValueError(
                "scale/floor_hard/combinatorial_upside cannot be mixed with an "
                "explicit nonlinearity instance"
            )
        return nonlinearity
    if isinstance(nonlinearity, str):
        if nonlinearity == "fit":
            effects = load_single_site_effects(dms_csv)
            fitted = GlobalEpistasisSigmoid.fit_from_singles(
                effects,
                combinatorial_upside=combinatorial_upside,
            )
            return GlobalEpistasisSigmoid(
                floor=fitted.floor,
                ceiling=fitted.ceiling,
                scale=fitted.scale if scale is None else float(scale),
                floor_hard=floor_hard,
            )
        raise ValueError(f"Unknown nonlinearity string {nonlinearity!r}")
    if isinstance(nonlinearity, int | float):
        if floor_hard is not None or combinatorial_upside != 1.0 or scale is not None:
            raise ValueError(
                "Pass either nonlinearity=<float> or scale=..., not both; "
                "floor_hard/combinatorial_upside require nonlinearity='fit'"
            )
        # Auto-set floor/ceiling from singles, use user-supplied scale.
        effects = load_single_site_effects(dms_csv)
        fitted = GlobalEpistasisSigmoid.fit_from_singles(effects)
        return GlobalEpistasisSigmoid(
            floor=fitted.floor,
            ceiling=fitted.ceiling,
            scale=float(nonlinearity),
        )
    raise TypeError(f"Unsupported nonlinearity type: {type(nonlinearity)}")


def build_landscape(
    dms_csv: str | Path,
    distances_npz: str | Path,
    wt_fasta: str | Path,
    alpha: float = 1.0,
    sigma: float = 0.3,
    seed: int = 42,
    epistasis_model: str = "random",
    threshold: float = 8.0,
    nonlinearity: NonLinearity | str | float | None = None,
    scale: float | None = None,
    floor_hard: float | None = None,
    combinatorial_upside: float = 1.0,
    noise_sigma: float = 0.0,
    readout_texture_sigma: float = 0.0,
    readout_texture_order: int = 0,
    readout_texture_iid_fraction: float = 0.0,
    readout_texture_floor: float | None = None,
    readout_texture_floor_band: float = 1e-9,
    readout_texture_seed: int | None = None,
    sigma_nk: float = 0.0,
    alpha_ho: float = 1.0,
    max_neighbours: int | None = None,
    msa_couplings: EpistasisModel | str | Path | None = None,
) -> Landscape:
    """Build a landscape from DMS data and a precomputed distance matrix.

    `nonlinearity` accepts None (default, Identity passthrough), a NonLinearity
    instance, "fit" (auto-fit a GlobalEpistasisSigmoid from the DMS singles), or
    a float (use that sigmoid scale, with floor/ceiling auto-set from singles).
    When `nonlinearity="fit"`, `scale`, `floor_hard`, and
    `combinatorial_upside` expose the paper-critical global-epistasis parameters
    without requiring callers to build the sigmoid by hand.

    `noise_sigma` is the std dev of Gaussian measurement noise added to each
    evaluated variant before the global-epistasis function, in stability-equivalent units.

    `sigma_nk > 0` adds a `ContactNKEpistasis` higher-order component: each
    position contributes a random function over itself plus its structural
    neighbours (capped at `max_neighbours` if given). `alpha_ho` is the scale
    scale parameter that multiplies the NK contribution in `Landscape.evaluate`; ignored
    when `sigma_nk == 0`.

    `msa_couplings`, when set, overrides `epistasis_model` and uses MSA-derived
    pairwise couplings instead of random ones. Accepts an MSACouplingsEpistasis
    instance, or a path to a .npz file (loaded via
    `MSACouplingsEpistasis.from_npz`).
    """
    distances = np.load(distances_npz)
    contact_map = ContactMap.from_distance_matrix(
        distances["distances"],
        distances["positions"].tolist(),
        threshold=threshold,
    )

    if msa_couplings is not None:
        if isinstance(msa_couplings, EpistasisModel):
            epistasis = msa_couplings
        else:
            epistasis = MSACouplingsEpistasis.from_npz(Path(msa_couplings))
    elif epistasis_model == "sign_consistent":
        epistasis = SignConsistentEpistasis(sigma=sigma, seed=seed)
    else:
        epistasis = RandomEpistasis(sigma=sigma, seed=seed)

    wt_seq = str(next(SeqIO.parse(str(wt_fasta), "fasta")).seq)
    nl = _resolve_nonlinearity_arg(
        nonlinearity,
        dms_csv,
        scale=scale,
        floor_hard=floor_hard,
        combinatorial_upside=combinatorial_upside,
    )

    landscape = Landscape.from_dms(
        dms_path=dms_csv,
        contact_map=contact_map,
        epistasis=epistasis,
        alpha=alpha,
        noise_sigma=noise_sigma,
        seed=seed,
        readout_texture_sigma=readout_texture_sigma,
        readout_texture_order=readout_texture_order,
        readout_texture_iid_fraction=readout_texture_iid_fraction,
        readout_texture_floor=readout_texture_floor,
        readout_texture_floor_band=readout_texture_floor_band,
        readout_texture_seed=readout_texture_seed,
        wt_sequence=wt_seq,
        nonlinearity=nl,
    )

    if sigma_nk > 0:
        nk_model = ContactNKEpistasis(
            sigma=sigma_nk, max_neighbours=max_neighbours, seed=seed,
        )
        nk_tensor = nk_model.generate(landscape.contact_map)
        # Re-create the landscape so __post_init__ pins the wildtype; cheap
        # since it only re-does the preimage cache.
        landscape = Landscape(
            wild_type=landscape.wild_type,
            single_site_effects=landscape.single_site_effects,
            contact_map=landscape.contact_map,
            couplings=landscape.couplings,
            alpha=landscape.alpha,
            noise_sigma=landscape.noise_sigma,
            seed=landscape.seed,
            deterministic_noise=landscape.deterministic_noise,
            readout_texture_sigma=landscape.readout_texture_sigma,
            readout_texture_order=landscape.readout_texture_order,
            readout_texture_iid_fraction=landscape.readout_texture_iid_fraction,
            readout_texture_floor=landscape.readout_texture_floor,
            readout_texture_floor_band=landscape.readout_texture_floor_band,
            readout_texture_seed=landscape.readout_texture_seed,
            wt_sequence=landscape.wt_sequence,
            nonlinearity=landscape.nonlinearity,
            higher_order=nk_tensor,
            alpha_ho=alpha_ho,
        )

    return landscape


def alpha_sweep(
    real_metrics: dict,
    single_site_effects: dict[int, dict[str, float]],
    build_fn,
    alphas: list[float] | None = None,
    depths: list[int] | None = None,
    n_per_depth: int = 3000,
) -> pd.DataFrame:
    """Sweep alpha values and compute validation scores.

    Parameters
    build_fn : callable(alpha) -> Landscape
        Function that builds a landscape at the given alpha.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    if depths is None:
        depths = list(range(2, 8))

    results = []

    for alpha in alphas:
        print(f"  alpha={alpha:.2f}", end="", flush=True)

        landscape = build_fn(alpha)
        synth_df = sample_synthetic_data(landscape, depths, n_per_depth=n_per_depth)
        synth_metrics = compute_all_metrics(synth_df, single_site_effects, depths)

        score = composite_validation_score(real_metrics, synth_metrics)

        row = {
            "alpha": alpha,
            "composite_score": score,
        }
        for d in depths:
            if d in synth_metrics["additive_correlation"]:
                row[f"additive_corr_d{d}"] = synth_metrics["additive_correlation"][d]
            if d in synth_metrics["sign_epistasis"]:
                row[f"sign_epi_d{d}"] = synth_metrics["sign_epistasis"][d]

        results.append(row)
        print(f" -> score={score:.4f}")

    return pd.DataFrame(results)


def find_optimal_alpha(sweep_df: pd.DataFrame) -> float:
    """Return the alpha with the lowest composite score."""
    best_idx = sweep_df["composite_score"].idxmin()
    return float(sweep_df.loc[best_idx, "alpha"])


def global_epistasis_sweep(
    real_metrics: dict,
    single_site_effects: dict[int, dict[str, float]],
    build_fn,
    alphas: list[float],
    scales: list[float],
    depths: list[int] | None = None,
    n_per_depth: int = 3000,
) -> pd.DataFrame:
    """Joint 2D sweep over (alpha, sigmoid_scale) for global-epistasis function.

    Parameters
    build_fn : callable(alpha, scale) -> Landscape
        Builds a landscape with the given alpha and sigmoid scale. The
        caller is responsible for setting floor/ceiling (usually via
        GlobalEpistasisSigmoid.fit_from_singles once, then varying scale).

    Returns DataFrame with columns: alpha, scale, composite_score, and
    per-depth additive_corr_d{d} / sign_epi_d{d} for d in depths.
    """
    if depths is None:
        depths = list(range(2, 8))

    rows = []
    for alpha in alphas:
        for scale in scales:
            print(f"  alpha={alpha:.2f} scale={scale:.3f}", end="", flush=True)
            landscape = build_fn(alpha, scale)
            synth_df = sample_synthetic_data(landscape, depths, n_per_depth=n_per_depth)
            synth_metrics = compute_all_metrics(synth_df, single_site_effects, depths)
            score = composite_validation_score(real_metrics, synth_metrics)

            row = {"alpha": alpha, "scale": scale, "composite_score": score}
            for d in depths:
                if d in synth_metrics["additive_correlation"]:
                    row[f"additive_corr_d{d}"] = synth_metrics["additive_correlation"][d]
                if d in synth_metrics["sign_epistasis"]:
                    row[f"sign_epi_d{d}"] = synth_metrics["sign_epistasis"][d]
            rows.append(row)
            print(f" -> score={score:.4f}")

    return pd.DataFrame(rows)


def find_optimal_global_epistasis(sweep_df: pd.DataFrame) -> tuple[float, float]:
    """Return (alpha, scale) with the lowest composite score."""
    best_idx = sweep_df["composite_score"].idxmin()
    row = sweep_df.loc[best_idx]
    return float(row["alpha"]), float(row["scale"])


def refine_alpha(
    real_metrics: dict,
    single_site_effects: dict[int, dict[str, float]],
    build_fn,
    coarse_best: float,
    step: float = 0.05,
    half_range: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    """Fine-grained alpha search around a coarse optimum."""
    fine_alphas = np.arange(
        max(0.0, coarse_best - half_range),
        coarse_best + half_range + step / 2,
        step,
    ).tolist()
    return alpha_sweep(
        real_metrics,
        single_site_effects,
        build_fn,
        alphas=fine_alphas,
        **kwargs,
    )
