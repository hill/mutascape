"""Small public facade for constructing task-validated benchmark oracles."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mutascape.completion import DMSPairwiseLandscape, PosteriorLandscape
from mutascape.imputation import complete_singles, restore_completion
from mutascape.mutant import Mutant, parse_variant
from mutascape.priors import DEFAULT_PRIOR_ID, load_prior_bundle

type DMSInput = str | Path | pd.DataFrame


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
    """Build a generator from DMS observations and a wild-type sequence.

    ``dms`` accepts a CSV path or a DataFrame. Supported forms follow.

    - ``position, wildtype, mutant, fitness`` for single mutants.
    - ``variant, fitness`` for WT, singles, and optional combinations.
    - ``mutations, fitness`` where mutations contains mapping objects.

    ``complete="esm2"`` calibrates masked ESM scores on measured singles.
    It fills missing substitutions across the protein and keeps observations exact.
    """
    if complete not in (None, "esm2"):
        raise ValueError("complete must be None or esm2")
    sequence = _load_sequence(fasta=fasta, wt_sequence=wt_sequence)
    frame = _load_dms_frame(dms)
    observations = _normalise_observations(
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
    manifest: str | Path | Mapping[str, object],
    *,
    dms: DMSInput,
    fasta: str | Path | None = None,
    wt_sequence: str | None = None,
    wild_type_fitness: float | None = None,
) -> DMSPairwiseLandscape:
    """Recreate and verify one fixed oracle from its saved manifest."""
    if isinstance(manifest, Mapping):
        settings = dict(manifest)
    else:
        settings = json.loads(Path(manifest).read_text())
    if settings.get("generator") != "dms-pairwise-v1":
        raise ValueError(f"unsupported oracle generator {settings.get('generator')!r}")
    prior_id = str(settings.get("prior", ""))
    if not prior_id or prior_id == "custom":
        raise ValueError("the manifest does not name a bundled prior")
    maximum_depth_value = settings.get("maximum_depth")
    maximum_depth = int(maximum_depth_value) if maximum_depth_value is not None else None
    if wild_type_fitness is None:
        wild_type_fitness = float(settings.get("wild_type_fitness", 0.0))
    generator = from_dms(
        dms,
        fasta=fasta,
        wt_sequence=wt_sequence,
        wild_type_fitness=wild_type_fitness,
        maximum_depth=maximum_depth,
        prior_id=prior_id,
    )
    expected_digest = str(settings.get("source_digest", ""))
    if "completion" in settings:
        generator = restore_completion(generator, settings["completion"])
    if generator.source_digest != expected_digest:
        raise ValueError("DMS data and wild-type sequence do not match the oracle manifest")
    return generator.sample_dms_pairwise(
        seed=int(settings["seed"]),
        coupling=float(settings["coupling"]),
    )


def _load_sequence(*, fasta: str | Path | None, wt_sequence: str | None) -> str:
    if (fasta is None) == (wt_sequence is None):
        raise ValueError("provide exactly one of fasta or wt_sequence")
    if fasta is not None:
        path = Path(fasta)
        if not path.exists():
            raise FileNotFoundError(f"wild-type FASTA does not exist: {path}")
        text = path.read_text()
        sequence = "".join(
            line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">")
        )
    else:
        sequence = "".join(str(wt_sequence).split())
    sequence = sequence.upper()
    if not sequence:
        raise ValueError("wild-type sequence is empty")
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if invalid:
        raise ValueError(f"wild-type sequence contains unsupported residues: {', '.join(invalid)}")
    return sequence


def _load_dms_frame(dms: DMSInput) -> pd.DataFrame:
    if isinstance(dms, pd.DataFrame):
        return dms.copy()
    path = Path(dms)
    if not path.exists():
        raise FileNotFoundError(f"DMS CSV does not exist: {path}")
    return pd.read_csv(path)


def _normalise_observations(
    frame: pd.DataFrame,
    sequence: str,
    *,
    wild_type_fitness: float,
) -> pd.DataFrame:
    renamed = {column: str(column).strip().lower() for column in frame.columns}
    frame = frame.rename(columns=renamed)
    columns = set(frame.columns)
    if {"position", "wildtype", "mutant", "fitness"}.issubset(columns):
        records: list[tuple[Mutant, float]] = [(Mutant(), wild_type_fitness)]
        for row in frame.itertuples(index=False):
            position = int(getattr(row, "position"))
            label = f"{getattr(row, 'wildtype')}{position}{getattr(row, 'mutant')}"
            records.append(
                (
                    parse_variant(label, wild_type_sequence=sequence),
                    _finite_fitness(getattr(row, "fitness"), label),
                )
            )
    elif {"variant", "fitness"}.issubset(columns):
        records = []
        for label, fitness in frame[["variant", "fitness"]].itertuples(index=False, name=None):
            records.append(
                (
                    parse_variant(str(label), wild_type_sequence=sequence),
                    _finite_fitness(fitness, str(label)),
                )
            )
    elif {"mutations", "fitness"}.issubset(columns):
        records = []
        for mutations, fitness in frame[["mutations", "fitness"]].itertuples(
            index=False, name=None
        ):
            mutant = parse_variant(mutations, wild_type_sequence=sequence)
            records.append((mutant, _finite_fitness(fitness, mutant.label)))
    else:
        raise ValueError(
            "DMS input needs position/wildtype/mutant/fitness, variant/fitness, "
            "or mutations/fitness columns"
        )

    seen: set[tuple[tuple[int, str], ...]] = set()
    rows = []
    for mutant, fitness in records:
        key = tuple(sorted(mutant.items()))
        if key in seen:
            raise ValueError(f"DMS input contains a duplicate measurement for {mutant.label}")
        seen.add(key)
        rows.append({"mutations": dict(mutant), "fitness": fitness, "depth": mutant.depth})
    if () not in seen:
        raise ValueError("DMS input must include WT")
    singles = {key[0] for key in seen if len(key) == 1}
    for genotype in seen:
        missing = [mutation for mutation in genotype if mutation not in singles]
        if missing:
            labels = ", ".join(
                f"{sequence[position - 1]}{position}{amino_acid}"
                for position, amino_acid in missing
            )
            raise ValueError(f"DMS input is missing constituent singles for {labels}")
    return pd.DataFrame(rows)


def _finite_fitness(value: object, label: str) -> float:
    try:
        fitness = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fitness for {label} must be numeric") from exc
    if not np.isfinite(fitness):
        raise ValueError(f"fitness for {label} must be finite")
    return fitness


LandscapeGenerator = PosteriorLandscape
Oracle = DMSPairwiseLandscape
