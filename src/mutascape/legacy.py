"""Empirical-prior generator used by the studio and earlier benchmark runs.

``mutascape.from_dms`` builds the singles-only model in ``mutascape.model``.
This module keeps the older posterior so saved ``dms-pairwise-v1`` manifests
and the studio continue to work.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from mutascape.completion import (
    CompletedLandscape,
    DMSPairwiseLandscape,
    LandscapePrior,
    PosteriorLandscape,
)
from mutascape.imputation import complete_singles, restore_completion
from mutascape.inputs import (
    DMSInput,
    Manifest,
    load_dms_frame,
    load_sequence,
    normalise_observations,
)
from mutascape.priors import DEFAULT_PRIOR_ID, load_prior_bundle
from mutascape.schemas import LegacyOracleManifest, generator_id

__all__ = [
    "CompletedLandscape",
    "DMSPairwiseLandscape",
    "LandscapePrior",
    "PosteriorLandscape",
    "from_dms",
    "load_oracle",
]


def from_dms(
    dms: DMSInput,
    *,
    fasta: str | Path | None = None,
    wt_sequence: str | None = None,
    wild_type_fitness: float = 0.0,
    maximum_depth: int | None = None,
    prior_id: str = DEFAULT_PRIOR_ID,
    complete: Literal["esm2"] | None = None,
    esm_model: str = "esm2_t33_650M_UR50D",
) -> PosteriorLandscape:
    """Build an empirical-prior posterior from DMS observations.

    ``dms`` accepts a CSV path or a DataFrame. Supported forms follow.

    - ``position, wildtype, mutant, fitness`` for single mutants.
    - ``variant, fitness`` for WT, singles, and optional combinations.
    - ``mutations, fitness`` where mutations contains mapping objects.

    ``complete="esm2"`` calibrates masked ESM scores on measured singles.
    It fills missing substitutions across the protein and keeps observations exact.
    """
    if complete not in (None, "esm2"):
        raise ValueError("complete must be None or esm2")
    sequence = load_sequence(fasta=fasta, wt_sequence=wt_sequence)
    frame = load_dms_frame(dms)
    observations = normalise_observations(
        frame,
        sequence,
        wild_type_fitness=float(wild_type_fitness),
    )
    _, prior = load_prior_bundle(prior_id)
    generator = PosteriorLandscape.condition(
        sequence,
        observations,
        prior,
        maximum_depth=maximum_depth,
        prior_id=prior_id,
    )
    if complete == "esm2":
        generator = complete_singles(generator, model_name=esm_model)
    return generator


def load_oracle(
    manifest: str | Path | Manifest,
    *,
    dms: DMSInput,
    fasta: str | Path | None = None,
    wt_sequence: str | None = None,
    wild_type_fitness: float | None = None,
) -> DMSPairwiseLandscape:
    """Recreate and verify one ``dms-pairwise-v1`` oracle from its manifest."""
    if isinstance(manifest, Mapping):
        settings = dict(manifest)
    else:
        settings = json.loads(Path(manifest).read_text())
    if generator_id(settings) != "dms-pairwise-v1":
        raise ValueError(f"unsupported oracle generator {generator_id(settings)!r}")
    saved = LegacyOracleManifest.model_validate(settings)
    if not saved.prior or saved.prior == "custom":
        raise ValueError("the manifest does not name a bundled prior")
    if wild_type_fitness is None:
        wild_type_fitness = saved.wild_type_fitness
    generator = from_dms(
        dms,
        fasta=fasta,
        wt_sequence=wt_sequence,
        wild_type_fitness=wild_type_fitness,
        maximum_depth=saved.maximum_depth,
        prior_id=saved.prior,
    )
    if saved.completion is not None:
        generator = restore_completion(generator, saved.completion)
    if generator.source_digest != saved.source_digest:
        raise ValueError("DMS data and wild-type sequence do not match the oracle manifest")
    return generator.sample_dms_pairwise(seed=saved.seed, coupling=saved.coupling)
