"""Frozen reference configurations for the publication validation pipeline.

These configs define the named GB1 and TrpB headline pipelines used by the
paper, the figure scripts, and the regression tests. The goal is to keep the
publication-critical parameter sets in one place instead of re-encoding them in
notebooks and ad hoc scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mutascape.nonlinearity import ThermodynamicSigmoid

from .calibration import build_landscape
from .comparisons import fitness_wasserstein
from .fourier import (
    fitness_tensor_from_landscape,
    fitness_tensor_from_real,
    observed_alphabets,
    walsh_decompose,
    walsh_l1_distance,
)
from .metrics import accessible_paths, local_optima_from_tensor, ssmula_attributes
from .real_data import load_gb1_combinatorial, load_single_site_effects

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_SEED = 42


@dataclass(frozen=True)
class ReferenceConfig:
    """Named, reproducible configuration for a validation protein."""

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
    combinatorial_upside: float = 3.0
    scale: float | None = None
    floor_hard: float | None = None
    use_singles_min_floor: bool = False
    seed: int = DEFAULT_REFERENCE_SEED

    def resolved_floor_hard(self) -> float | None:
        """Resolve an explicit hard floor for the censored sigmoid."""
        if self.floor_hard is not None:
            return float(self.floor_hard)
        if self.use_singles_min_floor:
            singles = load_single_site_effects(self.dms_csv)
            return float(min(effect for pos in singles.values() for effect in pos.values()))
        return None

    def fit_nonlinearity(self) -> ThermodynamicSigmoid:
        """Fit the publication thermodynamic non-linearity for this protein."""
        singles = load_single_site_effects(self.dms_csv)
        fitted = ThermodynamicSigmoid.fit_from_singles(
            singles,
            combinatorial_upside=self.combinatorial_upside,
        )
        return ThermodynamicSigmoid(
            floor=fitted.floor,
            ceiling=fitted.ceiling,
            scale=fitted.scale if self.scale is None else float(self.scale),
            floor_hard=self.resolved_floor_hard(),
        )

    def build_landscape(self, **overrides):
        """Build the named landscape through the shared validation API."""
        nonlinearity = overrides.pop("nonlinearity", "fit")
        params = {
            "alpha": self.alpha,
            "sigma": self.sigma,
            "noise_sigma": self.noise_sigma,
            "seed": self.seed,
            "nonlinearity": nonlinearity,
        }
        if nonlinearity == "fit":
            params.update({
                "scale": self.scale,
                "floor_hard": self.resolved_floor_hard(),
                "combinatorial_upside": self.combinatorial_upside,
            })
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
    fitness_w1: float
    real_n_optima: int
    synth_n_optima: int
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
    combinatorial_upside=3.0,
    scale=None,
    floor_hard=None,
    use_singles_min_floor=True,
)

REFERENCE_CONFIGS = {
    GB1_HEADLINE.name: GB1_HEADLINE,
    TRPB_HEADLINE.name: TRPB_HEADLINE,
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
        fitness_w1=fitness_wasserstein(real_tensor, synth_tensor),
        real_n_optima=real_optima["n_optima"],
        synth_n_optima=synth_optima["n_optima"],
        real_accessible_fraction=real_paths["mean_accessible_fraction"],
        synth_accessible_fraction=synth_paths["mean_accessible_fraction"],
        ssmula_real=ssmula_attributes(real_tensor),
        ssmula_synth=ssmula_attributes(synth_tensor),
    )
