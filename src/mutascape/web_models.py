"""Storage and calculations for the benchmark oracle web service."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import threading
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mutascape.contacts import ContactMap
from mutascape.epistasis import AMINO_ACIDS, RandomEpistasis
from mutascape.landscape import Landscape, _impute_missing_single_sites
from mutascape.mutant import Mutant
from mutascape.nonlinearity import GlobalEpistasisSigmoid, Identity
from mutascape.validation.comparisons import fitness_wasserstein
from mutascape.validation.fourier import (
    fitness_tensor_from_landscape,
    fitness_tensor_from_real,
    observed_alphabets,
    walsh_decompose,
    walsh_l1_distance,
)
from mutascape.validation.metrics import accessible_paths, local_optima_from_tensor
from mutascape.validation.reference_configs import GB1_HEADLINE, TRPB_HEADLINE, ReferenceConfig

CANONICAL_AAS = set(AMINO_ACIDS)
REFERENCE_CONFIGS = {
    "gb1_headline": GB1_HEADLINE,
    "trpb_headline": TRPB_HEADLINE,
}
POSTERIOR_REFERENCE_CONFIGS = {
    "gb1_posterior": "gb1_headline",
    "trpb_posterior": "trpb_headline",
}
DEFAULT_DATABASE = Path(__file__).resolve().parents[2] / ".mutascape" / "web.sqlite3"


@dataclass(frozen=True)
class BenchmarkParameters:
    """Parameters that define one reproducible benchmark oracle."""

    noise_sigma: float = 0.3
    alpha: float = 1.5
    pairwise_sigma: float = 0.0
    combinatorial_upside: float = 3.0
    seed: int = 42

    def validate(self) -> None:
        ranges = {
            "noise_sigma": (self.noise_sigma, 0.0, 5.0),
            "alpha": (self.alpha, 0.0, 10.0),
            "pairwise_sigma": (self.pairwise_sigma, 0.0, 5.0),
            "combinatorial_upside": (self.combinatorial_upside, 1.0, 10.0),
        }
        for name, (value, minimum, maximum) in ranges.items():
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")

    def as_json(self) -> dict[str, float | int]:
        return asdict(self)

    def as_public_json(self) -> dict[str, float | int]:
        return {
            "noiseSigma": self.noise_sigma,
            "alpha": self.alpha,
            "pairwiseSigma": self.pairwise_sigma,
            "combinatorialUpside": self.combinatorial_upside,
            "seed": self.seed,
        }


class ModelStore:
    """Small SQLite store for drafts, immutable models, and preview results."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.environ.get("MUTASCAPE_DB_PATH") or DEFAULT_DATABASE
        self.path = Path(configured)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY,
                    source_json TEXT NOT NULL,
                    uploads_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    uploads_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS preview_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leaderboards (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    landscape_id TEXT NOT NULL,
                    query_budget INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leaderboard_entries (
                    id TEXT PRIMARY KEY,
                    leaderboard_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    successes INTEGER NOT NULL,
                    trials INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(leaderboard_id) REFERENCES leaderboards(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS models_created_at_idx
                    ON models(created_at DESC);
                CREATE INDEX IF NOT EXISTS leaderboard_entries_board_idx
                    ON leaderboard_entries(leaderboard_id, created_at);
                """
            )
            model_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(models)").fetchall()
            }
            if "model_kind" not in model_columns:
                connection.execute(
                    "ALTER TABLE models ADD COLUMN model_kind TEXT NOT NULL DEFAULT 'deterministic'"
                )
            if "prior_id" not in model_columns:
                connection.execute("ALTER TABLE models ADD COLUMN prior_id TEXT")
            if "oracle_seed" not in model_columns:
                connection.execute("ALTER TABLE models ADD COLUMN oracle_seed INTEGER")

    def create_draft(self, source: dict[str, Any], uploads: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC)
        draft_id = f"draft_{uuid.uuid4().hex}"
        expires_at = now + timedelta(hours=24)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO drafts(id, source_json, uploads_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    _canonical_json(source),
                    _canonical_json(uploads),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        return {"id": draft_id, "expiresAt": expires_at.isoformat(), "source": source}

    def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        self.delete_expired_drafts()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM drafts WHERE id = ?",
                (draft_id,),
            ).fetchone()
        return _draft_row(row) if row else None

    def delete_expired_drafts(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM drafts WHERE expires_at <= ?",
                (datetime.now(UTC).isoformat(),),
            )
        return int(cursor.rowcount)

    def publish(
        self,
        draft_id: str,
        name: str,
        parameters: BenchmarkParameters,
        *,
        model_kind: str = "deterministic",
        prior_id: str | None = None,
        oracle_seed: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        draft = self.get_draft(draft_id)
        if draft is None:
            raise KeyError(draft_id)
        if model_kind not in {"deterministic", "posterior"}:
            raise ValueError("model kind must be deterministic or posterior")
        if model_kind == "posterior" and (prior_id is None or oracle_seed is None):
            raise ValueError("posterior models need a prior and oracle seed")
        model_id = stored_model_id(
            draft["source"],
            parameters,
            model_kind=model_kind,
            prior_id=prior_id,
            oracle_seed=oracle_seed,
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
            if existing:
                return _model_row(existing), False
            connection.execute(
                """
                INSERT INTO models(
                    id, name, source_json, uploads_json, parameters_json, created_at,
                    model_kind, prior_id, oracle_seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    name,
                    _canonical_json(draft["source"]),
                    _canonical_json(draft["uploads"]),
                    _canonical_json(parameters.as_json()),
                    now,
                    model_kind,
                    prior_id,
                    oracle_seed,
                ),
            )
            row = connection.execute(
                "SELECT * FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
        assert row is not None
        return _model_row(row), True

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
        return _model_row(row) if row else None

    def list_models(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM models ORDER BY created_at DESC, id"
            ).fetchall()
        return [_model_row(row) for row in rows]

    def get_cached_preview(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response_json FROM preview_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def cache_preview(self, cache_key: str, response: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO preview_cache(cache_key, response_json, created_at)
                VALUES (?, ?, ?)
                """,
                (cache_key, json.dumps(response), datetime.now(UTC).isoformat()),
            )

    def create_leaderboard(
        self,
        name: str,
        landscape_id: str,
        query_budget: int,
    ) -> dict[str, Any]:
        board_id = f"lb_{uuid.uuid4().hex[:16]}"
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO leaderboards(id, name, landscape_id, query_budget, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (board_id, name, landscape_id, query_budget, created_at),
            )
        board = self.get_leaderboard(board_id)
        assert board is not None
        return board

    def get_leaderboard(self, board_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            board = connection.execute(
                "SELECT * FROM leaderboards WHERE id = ?",
                (board_id,),
            ).fetchone()
            if board is None:
                return None
            entries = connection.execute(
                """
                SELECT * FROM leaderboard_entries
                WHERE leaderboard_id = ?
                ORDER BY CAST(successes AS REAL) / trials DESC, trials DESC, created_at
                """,
                (board_id,),
            ).fetchall()
        return _leaderboard_row(board, entries)

    def list_leaderboards(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM leaderboards ORDER BY created_at DESC, id"
            ).fetchall()
        return [board for row in rows if (board := self.get_leaderboard(row["id"]))]

    def add_leaderboard_entry(
        self,
        board_id: str,
        model_name: str,
        successes: int,
        trials: int,
    ) -> dict[str, Any]:
        if self.get_leaderboard(board_id) is None:
            raise KeyError(board_id)
        entry_id = f"run_{uuid.uuid4().hex[:16]}"
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO leaderboard_entries(
                    id, leaderboard_id, model_name, successes, trials, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (entry_id, board_id, model_name, successes, trials, created_at),
            )
        board = self.get_leaderboard(board_id)
        assert board is not None
        return next(entry for entry in board["entries"] if entry["id"] == entry_id)


def _draft_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source": json.loads(row["source_json"]),
        "uploads": json.loads(row["uploads_json"]),
        "createdAt": row["created_at"],
        "expiresAt": row["expires_at"],
    }


def _model_row(row: sqlite3.Row) -> dict[str, Any]:
    source = json.loads(row["source_json"])
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": "custom",
        "modelKind": row["model_kind"],
        "priorId": row["prior_id"],
        "oracleSeed": row["oracle_seed"],
        "source": source,
        "uploads": json.loads(row["uploads_json"]),
        "parameters": json.loads(row["parameters_json"]),
        "createdAt": row["created_at"],
        "positions": source["positions"],
        "wtCompact": source["wtCompact"],
    }


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a 95% Wilson score interval for a binomial success rate."""
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z**2 / trials
    centre = (rate + z**2 / (2 * trials)) / denominator
    spread = z * np.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _leaderboard_row(
    board: sqlite3.Row,
    entries: list[sqlite3.Row],
) -> dict[str, Any]:
    public_entries = []
    for entry in entries:
        successes = int(entry["successes"])
        trials = int(entry["trials"])
        low, high = wilson_interval(successes, trials)
        public_entries.append(
            {
                "id": entry["id"],
                "modelName": entry["model_name"],
                "successes": successes,
                "trials": trials,
                "successRate": successes / trials,
                "confidenceLow": low,
                "confidenceHigh": high,
                "createdAt": entry["created_at"],
            }
        )
    return {
        "id": board["id"],
        "name": board["name"],
        "landscapeId": board["landscape_id"],
        "queryBudget": int(board["query_budget"]),
        "createdAt": board["created_at"],
        "entries": public_entries,
    }


def parse_custom_source(
    dms_csv: str,
    wt_fasta: str,
    structure_text: str | None = None,
    structure_format: str | None = None,
    chain_id: str | None = None,
    structure_inference: str = "none",
    structure_cache_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate uploaded source text and return its normal form and summary."""
    if len(dms_csv.encode()) > 5_000_000:
        raise ValueError("DMS CSV must be at most 5 MB")
    if len(wt_fasta.encode()) > 1_000_000:
        raise ValueError("WT FASTA must be at most 1 MB")
    if structure_text is not None and len(structure_text.encode()) > 20_000_000:
        raise ValueError("structure file must be at most 20 MB")

    wt_sequence = _parse_fasta(wt_fasta)
    schema, rows, observation_rows, wild_type_fitness, input_row_count = _parse_dms_input(
        dms_csv,
        wt_sequence,
    )
    positions = sorted({int(row["position"]) for row in rows})
    wild_type = {int(row["position"]): row["wildtype"] for row in rows}
    wt_compact = "".join(wild_type[position] for position in positions)
    values = [float(row["fitness"]) for row in rows]
    can_fit = min(values) < 0 < max(values)

    if structure_inference not in {"none", "esmfold"}:
        raise ValueError("structure inference must be none or esmfold")
    if structure_text and structure_inference != "none":
        raise ValueError("choose either an uploaded structure or ESMFold inference")

    contact_positions = positions
    contact_pairs: list[list[int]] = []
    resolved_count = 0
    structure_method = "none"
    if structure_text:
        suffix = _normalise_structure_suffix(structure_format)
        contact_map = _contact_map_from_text(
            structure_text,
            suffix=suffix,
            chain_id=chain_id,
        )
        resolved_count = contact_map.n_positions
        contact_positions = list(contact_map.positions)
        contact_pairs = [list(pair) for pair in contact_map.contact_pairs()]
        structure_method = "uploaded"
    elif structure_inference == "esmfold":
        if len(wt_sequence) > 400:
            raise ValueError("ESMFold inference supports sequences up to 400 residues")
        try:
            contact_map = ContactMap.from_sequence(
                wt_sequence,
                threshold=8.0,
                cache_dir=(Path(structure_cache_dir) if structure_cache_dir else None),
            )
        except Exception as exc:
            raise ValueError(f"ESMFold structure inference failed: {exc}") from exc
        resolved_count = contact_map.n_positions
        contact_positions = list(contact_map.positions)
        contact_pairs = [list(pair) for pair in contact_map.contact_pairs()]
        structure_method = "esmfold"

    observed: dict[int, dict[str, float]] = {}
    for row in rows:
        position = int(row["position"])
        observed.setdefault(position, {row["wildtype"]: 0.0})[row["mutant"]] = float(row["fitness"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        complete = _impute_missing_single_sites(observed)
    imputed_count = sum(len(complete[position]) - len(observed[position]) for position in positions)
    depth_capacities = [0] * (min(4, len(positions)) + 1)
    depth_capacities[0] = 1
    for position in positions:
        alternatives = len(observed[position]) - 1
        for depth in range(len(depth_capacities) - 1, 0, -1):
            depth_capacities[depth] += depth_capacities[depth - 1] * alternatives

    source = {
        "version": 2,
        "dmsSchema": schema,
        "wtSequence": wt_sequence,
        "wtCompact": wt_compact,
        "wildTypeFitness": wild_type_fitness,
        "supportedSubstitutions": len(rows),
        "previewCapacity": min(5_000, sum(depth_capacities[1:])),
        "positions": positions,
        "wildType": {str(key): value for key, value in wild_type.items()},
        "dmsRows": rows,
        "observationRows": observation_rows,
        "canFitNonlinearity": can_fit,
        "contactPositions": contact_positions,
        "contactPairs": contact_pairs,
        "contactThreshold": 8.0,
        "hasStructure": structure_method != "none",
        "structureMethod": structure_method,
        "chainId": chain_id,
    }
    summary = {
        "positions": positions,
        "positionCount": len(positions),
        "wtLength": len(wt_sequence),
        "wtCompact": wt_compact,
        "measuredRows": input_row_count,
        "observationCount": len(observation_rows),
        "higherOrderRows": sum(row["depth"] > 1 for row in observation_rows),
        "dmsSchema": schema,
        "wildTypeFitness": wild_type_fitness,
        "supportedSubstitutions": len(rows),
        "previewCapacity": min(5_000, sum(depth_capacities[1:])),
        "imputedEffects": imputed_count,
        "canFitNonlinearity": can_fit,
        "hasStructure": structure_method != "none",
        "structureMethod": structure_method,
        "resolvedResidues": resolved_count,
        "contactCount": len(contact_pairs),
    }
    return source, summary


def _parse_dms_input(
    text: str,
    wt_sequence: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], float, int]:
    try:
        fieldnames = csv.DictReader(io.StringIO(text)).fieldnames
    except csv.Error as exc:
        raise ValueError(f"cannot parse DMS CSV: {exc}") from exc
    fields = set(fieldnames or ())
    if {"variant", "fitness"}.issubset(fields):
        return _parse_variant_csv(text, wt_sequence)
    rows = _parse_dms_csv(text, wt_sequence)
    observations = [
        {"variant": "wt", "mutations": {}, "fitness": 0.0, "depth": 0}
    ]
    observations.extend(
        {
            "variant": f"{row['wildtype']}{row['position']}{row['mutant']}",
            "mutations": {str(row["position"]): row["mutant"]},
            "fitness": float(row["fitness"]),
            "depth": 1,
        }
        for row in rows
    )
    return "singles", rows, observations, 0.0, len(rows)


def _parse_variant_csv(
    text: str,
    wt_sequence: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], float, int]:
    reader = csv.DictReader(io.StringIO(text))
    records: dict[tuple[tuple[int, str], ...], tuple[Mutant, float]] = {}
    for row_number, row in enumerate(reader, start=2):
        raw_label = str(row.get("variant", "")).strip().upper()
        try:
            fitness = float(row["fitness"])
            mutant = Mutant(raw_label)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid variant DMS value on row {row_number}: {exc}") from exc
        if not np.isfinite(fitness):
            raise ValueError(f"DMS fitness on row {row_number} must be finite")
        for position, amino_acid in mutant.items():
            if position < 1 or position > len(wt_sequence):
                raise ValueError(f"DMS position {position} is outside the WT sequence")
            claimed = mutant._wild_type.get(position)
            actual = wt_sequence[position - 1]
            if claimed != actual:
                raise ValueError(
                    f"DMS wild type {claimed}{position} does not match the WT FASTA"
                )
            if amino_acid not in CANONICAL_AAS:
                raise ValueError(f"DMS row {row_number} uses a non-canonical amino acid")
        key = tuple(sorted(mutant.items()))
        if key in records:
            label = Mutant.from_dict(dict(key), mutant._wild_type).label
            raise ValueError(f"DMS has a duplicate measurement for {label}")
        records[key] = (mutant, fitness)
    if not records:
        raise ValueError("DMS CSV has no data rows")
    if () not in records:
        raise ValueError("variant DMS CSV must include a WT row")

    single_keys = {key for key in records if len(key) == 1}
    for key in records:
        missing = [mutation for mutation in key if (mutation,) not in single_keys]
        if missing:
            labels = ", ".join(
                f"{wt_sequence[position - 1]}{position}{amino_acid}"
                for position, amino_acid in missing
            )
            raise ValueError(f"variant DMS is missing constituent singles: {labels}")

    wild_type_fitness = records[()][1]
    rows = []
    observations = []
    for key, (mutant, fitness) in sorted(records.items(), key=lambda item: (len(item[0]), item[0])):
        wild_type = {position: wt_sequence[position - 1] for position, _ in key}
        observations.append(
            {
                "variant": Mutant.from_dict(dict(key), wild_type).label,
                "mutations": {str(position): amino_acid for position, amino_acid in key},
                "fitness": float(fitness),
                "depth": len(key),
            }
        )
        if len(key) == 1:
            position, amino_acid = key[0]
            rows.append(
                {
                    "position": position,
                    "wildtype": wt_sequence[position - 1],
                    "mutant": amino_acid,
                    "fitness": float(fitness - wild_type_fitness),
                }
            )
    if not rows:
        raise ValueError("variant DMS CSV must include at least one single mutant")
    return "variants", rows, observations, float(wild_type_fitness), len(records)


def _parse_fasta(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("WT FASTA is empty")
    if not lines[0].startswith(">"):
        raise ValueError("WT FASTA must start with a header line")
    sequence = "".join(line for line in lines[1:] if not line.startswith(">"))
    if not sequence:
        raise ValueError("WT FASTA has no sequence")
    sequence = sequence.upper()
    invalid = sorted(set(sequence) - CANONICAL_AAS)
    if invalid:
        raise ValueError(f"WT FASTA contains unsupported residues: {','.join(invalid)}")
    return sequence


def _parse_dms_csv(text: str, wt_sequence: str) -> list[dict[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(text))
    except csv.Error as exc:
        raise ValueError(f"cannot parse DMS CSV: {exc}") from exc
    required = {"position", "wildtype", "mutant", "fitness"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("DMS CSV needs position, wildtype, mutant, and fitness columns")

    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    wild_type_by_position: dict[int, str] = {}
    for row_number, row in enumerate(reader, start=2):
        try:
            position = int(row["position"])
            wildtype = str(row["wildtype"]).strip().upper()
            mutant = str(row["mutant"]).strip().upper()
            fitness = float(row["fitness"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid DMS value on row {row_number}") from exc
        if position < 1 or position > len(wt_sequence):
            raise ValueError(f"DMS position {position} is outside the WT sequence")
        if wildtype not in CANONICAL_AAS or mutant not in CANONICAL_AAS:
            raise ValueError(f"DMS row {row_number} uses a non-canonical amino acid")
        if wt_sequence[position - 1] != wildtype:
            raise ValueError(f"DMS wild type {wildtype}{position} does not match the WT FASTA")
        if mutant == wildtype:
            raise ValueError(f"DMS row {row_number} repeats the wild-type amino acid")
        if not np.isfinite(fitness):
            raise ValueError(f"DMS fitness on row {row_number} must be finite")
        previous = wild_type_by_position.setdefault(position, wildtype)
        if previous != wildtype:
            raise ValueError(f"DMS rows disagree on the wild type at position {position}")
        key = (position, mutant)
        if key in by_key:
            raise ValueError(f"DMS has a duplicate measurement for {wildtype}{position}{mutant}")
        by_key[key] = {
            "position": position,
            "wildtype": wildtype,
            "mutant": mutant,
            "fitness": fitness,
        }
    if not by_key:
        raise ValueError("DMS CSV has no data rows")
    return [by_key[key] for key in sorted(by_key)]


def _normalise_structure_suffix(value: str | None) -> str:
    suffix = (value or "pdb").lower().lstrip(".")
    if suffix not in {"pdb", "cif", "mmcif"}:
        raise ValueError("structure format must be pdb, cif, or mmcif")
    return "." + suffix


def _contact_map_from_text(
    text: str,
    suffix: str,
    chain_id: str | None,
) -> ContactMap:
    with tempfile.TemporaryDirectory(prefix="mutascape-structure-") as directory:
        path = Path(directory) / f"structure{suffix}"
        path.write_text(text)
        try:
            return ContactMap.from_pdb(path, threshold=8.0, chain_id=chain_id)
        except Exception as exc:
            raise ValueError(f"cannot parse structure: {exc}") from exc


def custom_model_id(source: dict[str, Any], parameters: BenchmarkParameters) -> str:
    payload = {"source": source, "parameters": parameters.as_json()}
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return f"ms_{digest[:20]}"


def stored_model_id(
    source: dict[str, Any],
    parameters: BenchmarkParameters,
    *,
    model_kind: str,
    prior_id: str | None,
    oracle_seed: int | None,
) -> str:
    """Return a content ID for either stored model family."""
    if model_kind == "deterministic":
        return custom_model_id(source, parameters)
    payload = {
        "source": source,
        "modelKind": model_kind,
        "priorId": prior_id,
        "oracleSeed": oracle_seed,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return f"msp_{digest[:20]}"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def reference_catalogue_entries() -> list[dict[str, Any]]:
    entries = []
    descriptions = {
        "gb1_headline": "Complete four-site GB1 binding landscape from Wu et al. 2016.",
        "trpb_headline": "Complete four-site TrpB activity landscape from Johnston et al. 2024.",
    }
    for model_id, config in REFERENCE_CONFIGS.items():
        entries.append(
            {
                "id": model_id,
                "name": "GB1 reference" if model_id.startswith("gb1") else "TrpB reference",
                "kind": "reference",
                "description": descriptions[model_id],
                "positions": list(config.positions),
                "wtCompact": "".join(config.wt_aa_at_position[p] for p in config.positions),
                "parameters": _reference_parameters(config).as_public_json(),
                "createdAt": None,
            }
        )
    for model_id, base_id in POSTERIOR_REFERENCE_CONFIGS.items():
        config = REFERENCE_CONFIGS[base_id]
        protein = "GB1" if model_id.startswith("gb1") else "TrpB"
        entries.append(
            {
                "id": model_id,
                "name": f"{protein} posterior",
                "kind": "posterior",
                "description": (
                    f"Posterior completion of {protein} from its wild type and single mutants."
                ),
                "positions": list(config.positions),
                "wtCompact": "".join(config.wt_aa_at_position[p] for p in config.positions),
                "parameters": _reference_parameters(config).as_public_json(),
                "posteriorSettings": {
                    "priorId": "combinatorial-v1",
                    "oracleSeed": config.seed,
                },
                "createdAt": None,
            }
        )
    return entries


def reference_detail(model_id: str) -> dict[str, Any] | None:
    if model_id not in REFERENCE_CONFIGS and model_id not in POSTERIOR_REFERENCE_CONFIGS:
        return None
    return next(item for item in reference_catalogue_entries() if item["id"] == model_id)


@lru_cache(maxsize=2)
def reference_posterior_source(model_id: str) -> dict[str, Any]:
    """Return WT and single-mutant observations for a reference model."""
    base_id = POSTERIOR_REFERENCE_CONFIGS.get(model_id, model_id)
    config = REFERENCE_CONFIGS.get(base_id)
    if config is None:
        raise ValueError(f"unknown reference model {model_id}")
    source, _ = parse_custom_source(
        config.dms_csv.read_text(),
        config.wt_fasta.read_text(),
    )
    return source


def _reference_parameters(config: ReferenceConfig) -> BenchmarkParameters:
    return BenchmarkParameters(
        noise_sigma=config.noise_sigma,
        alpha=config.alpha,
        pairwise_sigma=config.sigma,
        combinatorial_upside=config.combinatorial_upside,
        seed=config.seed,
    )


def build_reference_landscape(
    model_id: str,
    parameters: BenchmarkParameters,
) -> Landscape:
    config = REFERENCE_CONFIGS[model_id]
    singles = _single_site_effects_from_rows(config.dms_csv)
    fitted = GlobalEpistasisSigmoid.fit_from_singles(
        singles,
        combinatorial_upside=parameters.combinatorial_upside,
    )
    nonlinearity = GlobalEpistasisSigmoid(
        floor=fitted.floor,
        ceiling=fitted.ceiling,
        scale=fitted.scale if config.scale is None else config.scale,
        floor_hard=config.resolved_floor_hard(),
    )
    return config.build_landscape(
        alpha=parameters.alpha,
        sigma=parameters.pairwise_sigma,
        noise_sigma=parameters.noise_sigma,
        seed=parameters.seed,
        nonlinearity=nonlinearity,
    )


def _single_site_effects_from_rows(path: Path) -> dict[int, dict[str, float]]:
    frame = pd.read_csv(path)
    effects: dict[int, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        position = int(row.position)
        effects.setdefault(position, {str(row.wildtype): 0.0})[str(row.mutant)] = float(row.fitness)
    return effects


def build_custom_landscape(
    source: dict[str, Any],
    parameters: BenchmarkParameters,
) -> Landscape:
    wild_type = {int(key): value for key, value in source["wildType"].items()}
    observed: dict[int, dict[str, float]] = {}
    for row in source["dmsRows"]:
        position = int(row["position"])
        observed.setdefault(position, {row["wildtype"]: 0.0})[row["mutant"]] = float(row["fitness"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        effects = _impute_missing_single_sites(observed)
    contacts = _contact_map_from_source(source, tuple(sorted(wild_type)))
    couplings = RandomEpistasis(
        sigma=parameters.pairwise_sigma,
        seed=parameters.seed,
    ).generate(contacts, effects)
    if source["canFitNonlinearity"]:
        nonlinearity = GlobalEpistasisSigmoid.fit_from_singles(
            observed,
            combinatorial_upside=parameters.combinatorial_upside,
        )
    else:
        nonlinearity = Identity()
    return Landscape(
        wild_type=wild_type,
        single_site_effects=effects,
        contact_map=contacts,
        couplings=couplings,
        alpha=parameters.alpha,
        noise_sigma=parameters.noise_sigma,
        seed=parameters.seed,
        wt_sequence=source["wtSequence"],
        nonlinearity=nonlinearity,
    )


def _contact_map_from_source(
    source: dict[str, Any],
    mutable_positions: tuple[int, ...],
) -> ContactMap:
    available = tuple(
        position
        for position in mutable_positions
        if position in set(int(p) for p in source["contactPositions"])
    )
    if not available:
        available = mutable_positions
    matrix = np.zeros((len(available), len(available)), dtype=bool)
    index = {position: offset for offset, position in enumerate(available)}
    for raw_a, raw_b in source["contactPairs"]:
        a, b = int(raw_a), int(raw_b)
        if a in index and b in index:
            matrix[index[a], index[b]] = True
            matrix[index[b], index[a]] = True
    return ContactMap(
        matrix=matrix,
        positions=available,
        threshold=float(source["contactThreshold"]),
    )


@lru_cache(maxsize=2)
def _reference_artifacts(model_id: str) -> dict[str, Any]:
    config = REFERENCE_CONFIGS[model_id]
    real_frame = config.load_real_dataframe()
    alphabets = observed_alphabets(real_frame, config.positions, config.wt_aa_at_position)
    real_tensor, coverage = fitness_tensor_from_real(
        real_frame,
        config.positions,
        config.wt_aa_at_position,
        alphabets,
    )
    real_walsh = walsh_decompose(real_tensor)
    real_topology = local_optima_from_tensor(real_tensor)
    real_paths = accessible_paths(
        real_tensor,
        n_sources=3000,
        n_rsign_samples=20000,
        seed=config.seed,
    )
    measured = {
        Mutant.from_dict(row.mutations, config.wt_aa_at_position).label: float(row.fitness)
        for row in real_frame.itertuples(index=False)
    }
    measured["wt"] = 0.0
    return {
        "frame": real_frame,
        "alphabets": alphabets,
        "tensor": real_tensor,
        "walsh": real_walsh,
        "topology": real_topology,
        "paths": real_paths,
        "coverage": coverage,
        "measured": measured,
    }


def preview_reference(
    model_id: str,
    parameters: BenchmarkParameters,
    sample_size: int,
    depth_weights: dict[int, float],
) -> dict[str, Any]:
    parameters.validate()
    artifacts = _reference_artifacts(model_id)
    config = REFERENCE_CONFIGS[model_id]
    landscape = build_reference_landscape(model_id, parameters)
    synth_tensor = fitness_tensor_from_landscape(
        landscape,
        config.positions,
        artifacts["alphabets"],
    )
    synth_walsh = walsh_decompose(synth_tensor)
    synth_topology = local_optima_from_tensor(synth_tensor)
    synth_paths = accessible_paths(
        synth_tensor,
        n_sources=3000,
        n_rsign_samples=20000,
        seed=parameters.seed,
    )
    depths = _tensor_depths(
        artifacts["alphabets"],
        config.positions,
        config.wt_aa_at_position,
    )
    sample = _sample_rows(
        landscape,
        sample_size,
        depth_weights,
        parameters.seed,
        measured=artifacts["measured"],
    )
    return {
        "source": reference_detail(model_id),
        "parameters": parameters.as_public_json(),
        "comparisonAvailable": True,
        "profiles": {
            "real": _fitness_profile(artifacts["tensor"], depths),
            "synthetic": _fitness_profile(synth_tensor, depths),
        },
        "walsh": {
            "orders": sorted(artifacts["walsh"]),
            "real": artifacts["walsh"],
            "synthetic": synth_walsh,
        },
        "metrics": [
            _metric("Fitness W1", 0.0, fitness_wasserstein(artifacts["tensor"], synth_tensor)),
            _metric("Walsh L1", 0.0, walsh_l1_distance(artifacts["walsh"], synth_walsh)),
            _metric("Local optima", artifacts["topology"]["n_optima"], synth_topology["n_optima"]),
            _metric(
                "Peak fraction",
                artifacts["topology"]["peak_fraction"],
                synth_topology["peak_fraction"],
            ),
            _metric(
                "Basin entropy",
                artifacts["topology"]["basin_entropy"],
                synth_topology["basin_entropy"],
            ),
            _metric(
                "Accessible fraction",
                artifacts["paths"]["mean_accessible_fraction"],
                synth_paths["mean_accessible_fraction"],
            ),
        ],
        "sample": sample,
        "notices": [
            f"Measured tensor coverage is {artifacts['coverage'] * 100:.1f}%.",
        ],
    }


def preview_custom(
    source: dict[str, Any],
    parameters: BenchmarkParameters,
    sample_size: int,
    depth_weights: dict[int, float],
    model_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    parameters.validate()
    landscape = build_custom_landscape(source, parameters)
    sample = _sample_rows(
        landscape,
        sample_size,
        depth_weights,
        parameters.seed,
    )
    return {
        "source": {
            "id": model_id,
            "name": name or "Custom draft",
            "kind": "custom",
            "positions": source["positions"],
            "wtCompact": source["wtCompact"],
        },
        "parameters": parameters.as_public_json(),
        "comparisonAvailable": False,
        "profiles": {"synthetic": sample["profile"]},
        "walsh": None,
        "metrics": [],
        "sample": sample,
        "notices": [
            "This preview has no measured combinatorial reference.",
            "The chart describes the sampled synthetic oracle only.",
        ],
    }


def _sample_rows(
    landscape: Landscape,
    sample_size: int,
    depth_weights: dict[int, float],
    seed: int,
    measured: dict[str, float] | None = None,
) -> dict[str, Any]:
    if sample_size < 1 or sample_size > 100_000:
        raise ValueError("preview sample size must be in [1, 100000]")
    frames = list(
        landscape.iter_sample_batches(
            n_variants=sample_size,
            depth_weights=depth_weights,
            seed=seed,
            batch_size=4096,
        )
    )
    frame = pd.concat(frames, ignore_index=True)
    graph = _sample_path_graph(landscape, frame)
    rows = []
    for row in frame.head(100).itertuples(index=False):
        rows.append(
            {
                "sequence": row.sequence,
                "variant": row.variant,
                "fitness": float(row.fitness),
                "depth": int(row.n_mutations),
                "measuredFitness": measured.get(row.variant) if measured else None,
            }
        )
    return {
        "requested": sample_size,
        "returnedRows": len(rows),
        "depthWeights": {str(key): value for key, value in depth_weights.items()},
        "profile": _fitness_profile(
            frame["fitness"].to_numpy(),
            frame["n_mutations"].to_numpy(),
        ),
        "rows": rows,
        "graph": graph,
    }


def _sample_path_graph(
    landscape: Landscape,
    frame: pd.DataFrame,
    max_paths: int = 60,
) -> dict[str, Any]:
    node_mutations: dict[str, dict[int, str]] = {"wt": {}}
    sampled_labels = list(dict.fromkeys(str(label) for label in frame.head(max_paths)["variant"]))
    sampled_label_set = set(sampled_labels)
    edges: set[tuple[str, str]] = set()
    for label in sampled_labels:
        target = Mutant(label)
        current: dict[int, str] = {}
        previous = "wt"
        for position in sorted(target):
            current[position] = target[position]
            current_label = Mutant.from_dict(current, landscape.wild_type).label
            node_mutations.setdefault(current_label, dict(current))
            edges.add((previous, current_label))
            previous = current_label

    labels = list(node_mutations)
    fitness = landscape.evaluate_batch([node_mutations[label] for label in labels])
    nodes = [
        {
            "id": label,
            "variant": label,
            "depth": len(node_mutations[label]),
            "fitness": float(value),
            "sampled": label in sampled_label_set,
        }
        for label, value in zip(labels, fitness)
    ]
    return {
        "nodes": nodes,
        "edges": [{"source": source, "target": target} for source, target in sorted(edges)],
    }


def _tensor_depths(
    alphabets: dict[int, list[str]],
    positions: tuple[int, ...],
    wild_type: dict[int, str],
) -> np.ndarray:
    shape = tuple(len(alphabets[position]) for position in positions)
    grids = np.indices(shape)
    depths = np.zeros(shape, dtype=np.int16)
    for axis, position in enumerate(positions):
        depths += grids[axis] != alphabets[position].index(wild_type[position])
    return depths


def _fitness_profile(values: np.ndarray, depths: np.ndarray) -> list[dict[str, Any]]:
    flat_values = np.asarray(values).ravel()
    flat_depths = np.asarray(depths).ravel()
    rows = []
    for depth in sorted(set(int(value) for value in flat_depths)):
        subset = flat_values[(flat_depths == depth) & np.isfinite(flat_values)]
        if subset.size == 0:
            continue
        q05, q25, median, q75, q95 = np.quantile(subset, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            {
                "depth": depth,
                "count": int(subset.size),
                "q05": float(q05),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "q95": float(q95),
            }
        )
    return rows


def _metric(name: str, real: float | int | None, synthetic: float | int) -> dict[str, Any]:
    return {"name": name, "real": real, "synthetic": synthetic}


def preview_cache_key(payload: dict[str, Any]) -> str:
    versioned = {"responseVersion": 2, "request": payload}
    return hashlib.sha256(_canonical_json(versioned).encode()).hexdigest()


class LandscapeRegistry:
    """Process-local cache for published oracle instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._landscapes: dict[str, Landscape] = {}

    def reference(self, model_id: str) -> Landscape:
        with self._lock:
            if model_id not in self._landscapes:
                self._landscapes[model_id] = build_reference_landscape(
                    model_id,
                    _reference_parameters(REFERENCE_CONFIGS[model_id]),
                )
            return self._landscapes[model_id]

    def custom(self, model: dict[str, Any]) -> Landscape:
        model_id = model["id"]
        with self._lock:
            if model_id not in self._landscapes:
                self._landscapes[model_id] = build_custom_landscape(
                    model["source"],
                    BenchmarkParameters(**model["parameters"]),
                )
            return self._landscapes[model_id]


def evaluate_entries(
    landscape: Landscape,
    entries: list[dict[str, str | None]],
) -> list[dict[str, Any]]:
    variants: list[dict[int, str]] = []
    labels: list[str] = []
    positions = list(landscape.positions)
    for entry in entries:
        label = entry.get("mutationLabel")
        sequence = entry.get("compactSequence")
        if bool(label) == bool(sequence):
            raise ValueError("each entry needs one mutationLabel or compactSequence")
        if label:
            mutant = Mutant(str(label))
            _validate_mutant(landscape, mutant)
        else:
            compact = str(sequence).upper()
            if len(compact) != len(positions):
                raise ValueError(f"compactSequence must contain {len(positions)} amino acids")
            invalid = sorted(set(compact) - CANONICAL_AAS)
            if invalid:
                raise ValueError(
                    f"compactSequence contains unsupported residues: {','.join(invalid)}"
                )
            mutation = {
                position: aa
                for position, aa in zip(positions, compact)
                if aa != landscape.wild_type[position]
            }
            mutant = Mutant.from_dict(mutation, landscape.wild_type)
        variants.append(dict(mutant))
        labels.append(Mutant.from_dict(dict(mutant), landscape.wild_type).label)
    fitness = landscape.evaluate_batch(variants)
    return [
        {"index": index, "mutationLabel": label, "fitness": float(value)}
        for index, (label, value) in enumerate(zip(labels, fitness))
    ]


def _validate_mutant(landscape: Landscape, mutant: Mutant) -> None:
    for position, amino_acid in mutant.items():
        if position not in landscape.wild_type:
            raise ValueError(f"position {position} is not mutable in this model")
        claimed_wt = mutant._wild_type.get(position)
        actual_wt = landscape.wild_type[position]
        if claimed_wt != actual_wt:
            raise ValueError(
                f"mutation at {position} claims wild type {claimed_wt}, expected {actual_wt}"
            )
        if amino_acid not in CANONICAL_AAS:
            raise ValueError(f"unsupported amino acid {amino_acid} at position {position}")
