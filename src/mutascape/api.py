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
    load_dms_frame,
    load_sequence,
    normalise_observations,
    normalise_pilot,
)
from mutascape.model import GENERATOR_ID, DMSGenerator, DMSLandscape


def from_dms(
    dms: DMSInput,
    *,
    fasta: str | Path | None = None,
    wt_sequence: str | None = None,
    wild_type_fitness: float = 0.0,
    maximum_depth: int | None = None,
    complete: Literal["esm2"] | None = None,
    esm_model: str = DEFAULT_ESM_MODEL,
    pilot: DMSInput | None = None,
) -> DMSGenerator | QuantileCalibratedLandscape:
    """Build a generator from wild type, measured singles and a wild-type sequence.

    ``dms`` accepts a CSV path or a DataFrame. Supported forms follow.

    - ``position, wildtype, mutant, fitness`` for single mutants.
    - ``variant, fitness`` for WT and singles.
    - ``mutations, fitness`` where mutations contains mapping objects.

    ``complete="esm2"`` calibrates masked ESM scores on measured singles.
    It fills missing substitutions across the protein and keeps observations exact.

    ``pilot`` takes measured multi-mutants sampled uniformly from the library.
    The generator then matches their fitness distribution, and each draw
    enumerates the full library once.
    """
    if complete not in (None, "esm2"):
        raise ValueError("complete must be None or esm2")
    sequence = load_sequence(fasta=fasta, wt_sequence=wt_sequence)
    observations = normalise_observations(
        load_dms_frame(dms), sequence, wild_type_fitness=float(wild_type_fitness)
    )
    if (observations.depth > 1).any():
        raise ValueError(
            "from_dms takes wild type and singles only. Pass measured combinations with pilot="
        )
    if pilot is not None:
        if complete is not None or maximum_depth is not None:
            raise ValueError("pilot calibration does not support complete or maximum_depth")
        return QuantileCalibratedLandscape.fit(
            sequence, observations, normalise_pilot(load_dms_frame(pilot), sequence)
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
            sequence, observations, maximum_depth=maximum_depth, completion=completion
        )
    )


def load_oracle(
    manifest: str | Path | Mapping[str, object],
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
    generator = settings.get("generator")
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
    if wild_type_fitness is None:
        wild_type_fitness = float(settings.get("wild_type_fitness", 0.0))
    sequence = load_sequence(fasta=fasta, wt_sequence=wt_sequence)
    observations = normalise_observations(
        load_dms_frame(dms), sequence, wild_type_fitness=float(wild_type_fitness)
    )
    contacts = settings.get("contacts")
    maximum_depth = settings.get("maximum_depth")
    oracle = DMSLandscape.from_singles(
        sequence,
        observations,
        seed=int(settings["seed"]),
        strength=float(settings["strength"]),
        mode=settings["mode"],
        margin=float(settings["margin"]),
        contacts={tuple(pair) for pair in contacts} if contacts is not None else None,
        maximum_depth=int(maximum_depth) if maximum_depth is not None else None,
        completion=settings.get("completion"),
    )
    if oracle.source_digest != settings.get("source_digest"):
        raise ValueError("DMS data and wild-type sequence do not match the oracle manifest")
    return oracle


LandscapeGenerator = DMSGenerator
Oracle = DMSLandscape
