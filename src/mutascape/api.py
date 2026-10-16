"""Small public facade for constructing benchmark oracles from DMS data."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from mutascape import legacy
from mutascape.calibrated import QuantileCalibratedLandscape
from mutascape.completion import DMSPairwiseLandscape
from mutascape.imputation import DEFAULT_ESM_MODEL, esm2_completion
from mutascape.inputs import (
    DMSInput,
    Manifest,
    load_dms_frame,
    load_sequence,
    normalise_observations,
    normalise_pilot,
)
from mutascape.model import GENERATOR_ID, DMSGenerator, DMSLandscape
from mutascape.nonlinearity import DEFAULT_SIGMOID_MARGIN
from mutascape.schemas import OracleManifest, generator_id


def from_dms(
    dms: DMSInput,
    *,
    fasta: str | Path | None = None,
    wt_sequence: str | None = None,
    wild_type_fitness: float = 0.0,
    maximum_depth: int | None = None,
    complete: Literal["esm2"] | None = None,
    esm_model: str = DEFAULT_ESM_MODEL,
    calibration: DMSInput | None = None,
    margin: float = DEFAULT_SIGMOID_MARGIN,
) -> DMSGenerator | QuantileCalibratedLandscape:
    """Build a generator from WT, single-site fitness and optional combination measurements.

    ``dms`` accepts a CSV path or a DataFrame. Supported forms follow.

    - ``position, wildtype, mutant, fitness`` for single-site variants.
    - ``variant, fitness`` for WT and single-site variants.
    - ``mutations, fitness`` where mutations contains mapping objects.

    ``complete="esm2"`` calibrates masked ESM scores on single-site measurements.
    It fills missing substitutions across the protein and keeps observations exact.

    ``calibration`` accepts a CSV path or DataFrame of measured multi-mutant
    genotypes sampled uniformly from the search space, using the same assay scale.
    Use ``variant, fitness`` or ``mutations, fitness`` columns. Every substitution
    must have a single-site measurement. These data fit depth effects, interaction
    strength and a fitness-distribution map. WT and single-site values stay exact.
    Calibration measurements fit the model and are not preserved as exact values.

    With calibration, ``generator.draw(seed=...)`` fixes the percentile map using
    100,000 uniformly sampled generated multi-mutants, or all multi-mutants when
    fewer exist. The search space has no default size limit. Pass
    ``reference_size=None`` to ``draw`` for exact enumeration. This mode does not
    support ``complete`` or ``maximum_depth``. Omitting calibration keeps the
    single-site-only generator.

    ``margin`` pads the sigmoid beyond the supplied range, as a fraction of that
    range, so every measurement sits strictly inside the limits.
    """
    if complete not in (None, "esm2"):
        raise ValueError("complete must be None or esm2")
    sequence = load_sequence(fasta=fasta, wt_sequence=wt_sequence)
    observations = normalise_observations(
        load_dms_frame(dms), sequence, wild_type_fitness=float(wild_type_fitness)
    )
    if (observations.depth > 1).any():
        raise ValueError(
            "from_dms takes WT and single-site measurements in dms. "
            "Pass measured combinations with calibration="
        )
    if calibration is not None:
        if complete is not None or maximum_depth is not None:
            raise ValueError("combination calibration does not support complete or maximum_depth")
        return QuantileCalibratedLandscape.fit(
            sequence,
            observations,
            normalise_pilot(load_dms_frame(calibration), sequence),
            margin=margin,
        )
    completion = None
    if complete == "esm2":
        wt = float(wild_type_fitness)
        measured = {
            next(iter(mutations.items())): fitness - wt
            for mutations, fitness in observations[["mutations", "fitness"]].itertuples(index=False)
            if mutations
        }
        completion = esm2_completion(sequence, measured, model_name=esm_model)
    return DMSGenerator(
        DMSLandscape.from_singles(
            sequence,
            observations,
            maximum_depth=maximum_depth,
            completion=completion,
            margin=margin,
        )
    )


def load_oracle(
    manifest: str | Path | Manifest,
    *,
    dms: DMSInput,
    fasta: str | Path | None = None,
    wt_sequence: str | None = None,
    wild_type_fitness: float | None = None,
) -> DMSLandscape | DMSPairwiseLandscape:
    """Recreate and verify one fixed oracle from its saved manifest.

    Manifests from the empirical-prior generator load through ``mutascape.legacy``.
    """
    if isinstance(manifest, Mapping):
        settings = dict(manifest)
    else:
        settings = json.loads(Path(manifest).read_text())
    generator = generator_id(settings)
    if generator == "dms-pairwise-v1":
        return legacy.load_oracle(
            settings,
            dms=dms,
            fasta=fasta,
            wt_sequence=wt_sequence,
            wild_type_fitness=wild_type_fitness,
        )
    if generator != GENERATOR_ID:
        raise ValueError(f"unsupported oracle generator {generator!r}")
    saved = OracleManifest.model_validate(settings)
    if wild_type_fitness is None:
        wild_type_fitness = saved.wild_type_fitness
    sequence = load_sequence(fasta=fasta, wt_sequence=wt_sequence)
    observations = normalise_observations(
        load_dms_frame(dms), sequence, wild_type_fitness=float(wild_type_fitness)
    )
    oracle = DMSLandscape.from_singles(
        sequence,
        observations,
        seed=saved.seed,
        strength=saved.strength,
        mode=saved.mode,
        margin=saved.margin,
        contacts=set(saved.contacts) if saved.contacts is not None else None,
        maximum_depth=saved.maximum_depth,
        completion=saved.completion,
    )
    if oracle.source_digest != saved.source_digest:
        raise ValueError("DMS data and wild-type sequence do not match the oracle manifest")
    return oracle


LandscapeGenerator = DMSGenerator
Oracle = DMSLandscape
