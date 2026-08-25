"""Frozen reference configurations for the publication validation pipeline.

These configs define the named GB1 and TrpB headline pipelines used by the
paper, the figure scripts, and the regression tests. The goal is to keep the
publication-critical parameter sets in one place instead of re-encoding them in
notebooks and ad hoc scripts.
"""

from dataclasses import dataclass
from pathlib import Path

from mutascape.nonlinearity import GlobalEpistasisSigmoid

from .calibration import build_landscape
from .comparisons import fitness_wasserstein
from .fourier import (
    fitness_tensor_from_landscape,
    fitness_tensor_from_real,
    observed_alphabets,
    order_wasserstein,
    spectral_cosine,
    walsh_decompose,
    walsh_l1_distance,
    walsh_signed_effects,
)
from .metrics import accessible_paths, local_optima_from_tensor, ssmula_attributes
from .real_data import load_gb1_combinatorial, load_single_site_effects

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_SEED = 42


@dataclass(frozen=True)
class ReferenceConfig:
    """Named, reproducible configuration for a validation protein.

    `msa_couplings_npz`, when set, switches the pairwise epistasis from random
    Gaussian (`RandomEpistasis(sigma)`) to an MSA-fitted Potts coupling tensor
    loaded from disk via `MSACouplingsEpistasis.from_npz`. `msa_coupling_scale`
    multiplies the loaded J to keep its magnitude comparable to the additive
    DMS field (`h_i`) and is independent of the landscape `alpha`.
    """

    name: str
    dms_csv: Path
    distances_npz: Path
    wt_fasta: Path
    combinatorial_csv: Path
    positions: tuple[int, ...]
    wt_aa_at_position: dict[int, str]
    output_dir: Path
    alpha: float = 1.5
    sigma: float = 0.0
    noise_sigma: float = 0.3
    readout_texture_sigma: float = 0.0
    readout_texture_order: int = 0
    readout_texture_iid_fraction: float = 0.0
    readout_texture_floor_band: float = 1e-9
    readout_texture_seed: int | None = None
    combinatorial_upside: float = 3.0
    scale: float | None = None
    floor_hard: float | None = None
    use_singles_min_floor: bool = False
    seed: int = DEFAULT_REFERENCE_SEED
    msa_couplings_npz: Path | None = None
    msa_coupling_scale: float = 1.0

    def resolved_floor_hard(self) -> float | None:
        """Resolve an explicit hard floor for the censored sigmoid."""
        if self.floor_hard is not None:
            return float(self.floor_hard)
        if self.use_singles_min_floor:
            singles = load_single_site_effects(self.dms_csv)
            return float(min(effect for pos in singles.values() for effect in pos.values()))
        return None

    def fit_nonlinearity(self) -> GlobalEpistasisSigmoid:
        """Fit the publication global-epistasis function for this protein."""
        singles = load_single_site_effects(self.dms_csv)
        fitted = GlobalEpistasisSigmoid.fit_from_singles(
            singles,
            combinatorial_upside=self.combinatorial_upside,
        )
        return GlobalEpistasisSigmoid(
            floor=fitted.floor,
            ceiling=fitted.ceiling,
            scale=fitted.scale if self.scale is None else float(self.scale),
            floor_hard=self.resolved_floor_hard(),
        )

    def build_landscape(self, **overrides):
        """Build the named landscape through the shared validation API."""
        from mutascape.epistasis import MSACouplingsEpistasis

        nonlinearity = overrides.pop("nonlinearity", "fit")
        params = {
            "alpha": self.alpha,
            "sigma": self.sigma,
            "noise_sigma": self.noise_sigma,
            "readout_texture_sigma": self.readout_texture_sigma,
            "readout_texture_order": self.readout_texture_order,
            "readout_texture_iid_fraction": self.readout_texture_iid_fraction,
            "readout_texture_floor": self.resolved_floor_hard(),
            "readout_texture_floor_band": self.readout_texture_floor_band,
            "readout_texture_seed": self.readout_texture_seed,
            "seed": self.seed,
            "nonlinearity": nonlinearity,
        }
        if nonlinearity == "fit":
            params.update(
                {
                    "scale": self.scale,
                    "floor_hard": self.resolved_floor_hard(),
                    "combinatorial_upside": self.combinatorial_upside,
                }
            )
        if self.msa_couplings_npz is not None:
            params["msa_couplings"] = MSACouplingsEpistasis.from_npz(
                self.msa_couplings_npz,
                scale=self.msa_coupling_scale,
            )
        params.update(overrides)
        return build_landscape(
            self.dms_csv,
            self.distances_npz,
            self.wt_fasta,
            **params,
        )

    def load_real_dataframe(self):
        """Load the prepared real combinatorial dataset for this protein."""
        return load_gb1_combinatorial(
            self.combinatorial_csv,
            self.dms_csv,
            positions=self.positions,
        )


@dataclass(frozen=True)
class ReferenceEvaluation:
    """Summary metrics for a frozen reference configuration."""

    config: ReferenceConfig
    real_walsh: dict[int, float]
    synth_walsh: dict[int, float]
    walsh_l1: float
    walsh_order_w1: float
    walsh_cos_order2: float
    walsh_cos_ge2: float
    fitness_w1: float
    real_n_optima: int
    synth_n_optima: int
    real_peak_fraction: float
    synth_peak_fraction: float
    real_largest_basin_fraction: float
    synth_largest_basin_fraction: float
    real_basin_entropy: float
    synth_basin_entropy: float
    real_accessible_fraction: float
    synth_accessible_fraction: float
    ssmula_real: dict[str, float]
    ssmula_synth: dict[str, float]


GB1_HEADLINE = ReferenceConfig(
    name="gb1_headline",
    dms_csv=REPO_ROOT / "data/gb1/gb1.csv",
    distances_npz=REPO_ROOT / "data/gb1/gb1_distances.npz",
    wt_fasta=REPO_ROOT / "data/gb1/gb1_wt.fasta",
    combinatorial_csv=REPO_ROOT / "data/gb1/gb1_combinatorial.csv",
    positions=(39, 40, 41, 54),
    wt_aa_at_position={39: "V", 40: "D", 41: "G", 54: "V"},
    output_dir=REPO_ROOT / "notebooks/validation/output_gb1",
    alpha=1.5,
    sigma=0.0,
    noise_sigma=0.3,
    readout_texture_sigma=0.005,
    readout_texture_order=3,
    readout_texture_iid_fraction=0.3,
    combinatorial_upside=3.0,
    scale=1.0,
    floor_hard=-1.0,
)

TRPB_HEADLINE = ReferenceConfig(
    name="trpb_headline",
    dms_csv=REPO_ROOT / "data/trpb/trpb.csv",
    distances_npz=REPO_ROOT / "data/trpb/trpb_distances.npz",
    wt_fasta=REPO_ROOT / "data/trpb/trpb_wt.fasta",
    combinatorial_csv=REPO_ROOT / "data/trpb/trpb_combinatorial.csv",
    positions=(183, 184, 227, 228),
    wt_aa_at_position={183: "V", 184: "F", 227: "V", 228: "S"},
    output_dir=REPO_ROOT / "notebooks/validation/output_trpb",
    alpha=1.5,
    sigma=0.0,
    noise_sigma=0.3,
    readout_texture_sigma=0.003,
    readout_texture_order=4,
    combinatorial_upside=3.0,
    scale=None,
    floor_hard=None,
    use_singles_min_floor=True,
)

GB1_V2 = ReferenceConfig(
    name="gb1_v2",
    dms_csv=REPO_ROOT / "data/gb1/gb1.csv",
    distances_npz=REPO_ROOT / "data/gb1/gb1_distances.npz",
    wt_fasta=REPO_ROOT / "data/gb1/gb1_wt.fasta",
    combinatorial_csv=REPO_ROOT / "data/gb1/gb1_combinatorial.csv",
    positions=(39, 40, 41, 54),
    wt_aa_at_position={39: "V", 40: "D", 41: "G", 54: "V"},
    output_dir=REPO_ROOT / "notebooks/validation/output_gb1_v2",
    alpha=1.5,
    sigma=0.0,
    noise_sigma=0.3,
    readout_texture_sigma=0.005,
    readout_texture_order=3,
    readout_texture_iid_fraction=0.3,
    combinatorial_upside=3.0,
    scale=1.0,
    floor_hard=-1.0,
    msa_couplings_npz=REPO_ROOT / "data/gb1/gb1_msa_couplings.npz",
    msa_coupling_scale=1.0,
)

TRPB_V2 = ReferenceConfig(
    name="trpb_v2",
    dms_csv=REPO_ROOT / "data/trpb/trpb.csv",
    distances_npz=REPO_ROOT / "data/trpb/trpb_distances.npz",
    wt_fasta=REPO_ROOT / "data/trpb/trpb_wt.fasta",
    combinatorial_csv=REPO_ROOT / "data/trpb/trpb_combinatorial.csv",
    positions=(183, 184, 227, 228),
    wt_aa_at_position={183: "V", 184: "F", 227: "V", 228: "S"},
    output_dir=REPO_ROOT / "notebooks/validation/output_trpb_v2",
    alpha=1.5,
    sigma=0.0,
    noise_sigma=0.3,
    readout_texture_sigma=0.003,
    readout_texture_order=4,
    combinatorial_upside=3.0,
    scale=None,
    floor_hard=None,
    use_singles_min_floor=True,
    msa_couplings_npz=REPO_ROOT / "data/trpb/trpb_msa_couplings.npz",
    msa_coupling_scale=1.0,
)

REFERENCE_CONFIGS = {
    GB1_HEADLINE.name: GB1_HEADLINE,
    TRPB_HEADLINE.name: TRPB_HEADLINE,
    GB1_V2.name: GB1_V2,
    TRPB_V2.name: TRPB_V2,
}


def get_reference_config(name: str) -> ReferenceConfig:
    """Return a frozen named reference configuration."""
    try:
        return REFERENCE_CONFIGS[name]
    except KeyError as exc:
        known = ", ".join(sorted(REFERENCE_CONFIGS))
        raise KeyError(f"Unknown reference config {name!r}. Known: {known}") from exc


def build_gb1_headline_landscape(**overrides):
    """Named builder for the GB1 publication reference configuration."""
    return GB1_HEADLINE.build_landscape(**overrides)


def build_trpb_headline_landscape(**overrides):
    """Named builder for the TrpB publication reference configuration."""
    return TRPB_HEADLINE.build_landscape(**overrides)


def evaluate_reference_config(config: ReferenceConfig) -> ReferenceEvaluation:
    """Compute publication headline metrics for a frozen config."""
    real_df = config.load_real_dataframe()
    alphabets = observed_alphabets(real_df, config.positions, config.wt_aa_at_position)
    real_tensor, _coverage = fitness_tensor_from_real(
        real_df,
        config.positions,
        config.wt_aa_at_position,
        alphabets,
    )
    synth_tensor = fitness_tensor_from_landscape(
        config.build_landscape(),
        config.positions,
        alphabets,
    )

    real_walsh = walsh_decompose(real_tensor)
    synth_walsh = walsh_decompose(synth_tensor)

    # Signed Walsh coefficients on the SAME alphabets align element by element,
    # so cosine on an order-masked slice compares the actual interaction pattern
    # (which subsets are coupled, with what sign and relative magnitude). Order 1
    # is shared by construction (both sides use the same DMS singles), so the
    # discriminating signal lives at order >= 2; order 2 is reported as primary.
    real_coeffs, coeff_orders = walsh_signed_effects(real_tensor)
    synth_coeffs, _ = walsh_signed_effects(synth_tensor)
    cos_order2 = spectral_cosine(real_coeffs[coeff_orders == 2], synth_coeffs[coeff_orders == 2])
    cos_ge2 = spectral_cosine(real_coeffs[coeff_orders >= 2], synth_coeffs[coeff_orders >= 2])

    real_paths = accessible_paths(
        real_tensor,
        n_sources=3000,
        n_rsign_samples=20000,
        seed=config.seed,
    )
    synth_paths = accessible_paths(
        synth_tensor,
        n_sources=3000,
        n_rsign_samples=20000,
        seed=config.seed,
    )
    real_optima = local_optima_from_tensor(real_tensor)
    synth_optima = local_optima_from_tensor(synth_tensor)

    return ReferenceEvaluation(
        config=config,
        real_walsh=real_walsh,
        synth_walsh=synth_walsh,
        walsh_l1=walsh_l1_distance(real_walsh, synth_walsh),
        walsh_order_w1=order_wasserstein(real_walsh, synth_walsh),
        walsh_cos_order2=cos_order2,
        walsh_cos_ge2=cos_ge2,
        fitness_w1=fitness_wasserstein(real_tensor, synth_tensor),
        real_n_optima=real_optima["n_optima"],
        synth_n_optima=synth_optima["n_optima"],
        real_peak_fraction=real_optima["peak_fraction"],
        synth_peak_fraction=synth_optima["peak_fraction"],
        real_largest_basin_fraction=real_optima["largest_basin_fraction"],
        synth_largest_basin_fraction=synth_optima["largest_basin_fraction"],
        real_basin_entropy=real_optima["basin_entropy"],
        synth_basin_entropy=synth_optima["basin_entropy"],
        real_accessible_fraction=real_paths["mean_accessible_fraction"],
        synth_accessible_fraction=synth_paths["mean_accessible_fraction"],
        ssmula_real=ssmula_attributes(real_tensor),
        ssmula_synth=ssmula_attributes(synth_tensor),
    )
