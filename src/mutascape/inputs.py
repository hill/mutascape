"""Parse DMS tables and wild-type sequences into canonical observations."""

from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.mutant import Mutant, parse_variant

type DMSInput = str | Path | pd.DataFrame


def load_sequence(*, fasta: str | Path | None, wt_sequence: str | None) -> str:
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


def load_dms_frame(dms: DMSInput) -> pd.DataFrame:
    if isinstance(dms, pd.DataFrame):
        return dms.copy()
    path = Path(dms)
    if not path.exists():
        raise FileNotFoundError(f"DMS CSV does not exist: {path}")
    return pd.read_csv(path)


def normalise_observations(
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
                    finite_fitness(getattr(row, "fitness"), label),
                )
            )
    elif {"variant", "fitness"}.issubset(columns):
        records = []
        for label, fitness in frame[["variant", "fitness"]].itertuples(index=False, name=None):
            records.append(
                (
                    parse_variant(str(label), wild_type_sequence=sequence),
                    finite_fitness(fitness, str(label)),
                )
            )
    elif {"mutations", "fitness"}.issubset(columns):
        records = []
        for mutations, fitness in frame[["mutations", "fitness"]].itertuples(
            index=False, name=None
        ):
            mutant = parse_variant(mutations, wild_type_sequence=sequence)
            records.append((mutant, finite_fitness(fitness, mutant.label)))
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


def finite_fitness(value: object, label: str) -> float:
    try:
        fitness = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fitness for {label} must be numeric") from exc
    if not np.isfinite(fitness):
        raise ValueError(f"fitness for {label} must be finite")
    return fitness


def normalise_pilot(frame: pd.DataFrame, sequence: str) -> pd.DataFrame:
    """Parse measured multi-mutants supplied for distribution calibration."""
    frame = frame.rename(columns={column: str(column).strip().lower() for column in frame.columns})
    column = next((name for name in ("variant", "mutations") if name in frame.columns), None)
    if column is None or "fitness" not in frame.columns:
        raise ValueError("calibration input needs variant/fitness or mutations/fitness columns")
    rows, seen = [], set()
    for variant, fitness in frame[[column, "fitness"]].itertuples(index=False, name=None):
        mutant = parse_variant(
            str(variant) if column == "variant" else variant, wild_type_sequence=sequence
        )
        key = tuple(sorted(mutant.items()))
        if len(key) < 2:
            raise ValueError(
                f"calibration input must contain only multi-mutant genotypes, found {mutant.label}"
            )
        if key in seen:
            raise ValueError(
                f"calibration input contains a duplicate measurement for {mutant.label}"
            )
        seen.add(key)
        rows.append({"mutations": dict(mutant), "fitness": finite_fitness(fitness, mutant.label)})
    return pd.DataFrame(rows)
