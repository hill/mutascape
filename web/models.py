"""Storage and calculations for the benchmark studio."""

import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import threading
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from mutascape.contacts import ContactMap
from mutascape.epistasis import AMINO_ACIDS, RandomEpistasis
from mutascape.landscape import Landscape, _impute_missing_single_sites
from mutascape.mutant import Mutant
from mutascape.nonlinearity import GlobalEpistasisSigmoid, Identity
from mutascape.priors import DEFAULT_PRIOR_ID
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
CREILOV_POSTERIOR_ID = "creilov_dms_pairwise"
CREILOV_DMS_CSV = Path(__file__).resolve().parents[1] / "data/creilov/creilov.csv"
CREILOV_FASTA = Path(__file__).resolve().parents[1] / "data/creilov/creilov_wt.fasta"
DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / ".mutascape" / "web.sqlite3"


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
    column_map: dict[str, Any] | None = None,
    wild_type_score: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate uploaded source text and return its normal form and summary."""
    if len(dms_csv.encode()) > 20_000_000:
        raise ValueError("DMS CSV must be at most 20 MB")
    if len(wt_fasta.encode()) > 1_000_000:
        raise ValueError("WT FASTA must be at most 1 MB")
    if structure_text is not None and len(structure_text.encode()) > 20_000_000:
        raise ValueError("structure file must be at most 20 MB")

    wt_sequence = _parse_fasta(wt_fasta)
    parsed_input = _parse_dms_input(
        dms_csv,
        wt_sequence,
        column_map=column_map,
        wild_type_score=wild_type_score,
    )
    schema = parsed_input.schema
    rows = parsed_input.rows
    observation_rows = parsed_input.observations
    wild_type_fitness = parsed_input.wild_type_fitness
    input_row_count = parsed_input.input_row_count
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
        "skippedRows": parsed_input.skipped,
        "wildTypeScoreAssumed": parsed_input.wild_type_assumed,
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


def substitution_map(
    source: dict[str, Any],
    max_positions: int = 1_000,
    scope: str = "assayed",
) -> dict[str, Any]:
    """Return a position by amino-acid grid of single-site effects."""
    if scope not in {"assayed", "sequence"}:
        raise ValueError("scope must be assayed or sequence")
    assayed = sorted(int(position) for position in source["positions"])
    sequence = str(source["wtSequence"])
    columns = list(range(1, len(sequence) + 1)) if scope == "sequence" else assayed
    total_positions = len(columns)
    kept = columns[:max_positions]
    index_of = {position: index for index, position in enumerate(kept)}
    assayed_set = set(assayed)
    wild_type = {int(key): value for key, value in source["wildType"].items()}

    observed: dict[int, dict[str, float]] = {}
    for row in source["dmsRows"]:
        position = int(row["position"])
        observed.setdefault(position, {row["wildtype"]: 0.0})[row["mutant"]] = float(row["fitness"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        complete = _impute_missing_single_sites(observed)

    measured: list[list[float]] = []
    imputed: list[list[float]] = []
    wild_type_cells: list[list[int]] = []
    bound = 0.0
    for position in kept:
        column = index_of[position]
        parent = wild_type.get(position) or sequence[position - 1]
        for row, amino_acid in enumerate(AMINO_ACIDS):
            if amino_acid == parent:
                wild_type_cells.append([column, row])
                continue
            if position not in assayed_set:
                continue
            value = float(complete[position][amino_acid])
            bound = max(bound, abs(value))
            target = measured if amino_acid in observed[position] else imputed
            target.append([column, row, value])

    return {
        "positions": kept,
        "aminoAcids": "".join(AMINO_ACIDS),
        "wildTypeCompact": "".join(
            wild_type.get(position) or sequence[position - 1] for position in kept
        ),
        "assayed": [index_of[position] for position in kept if position in assayed_set],
        "measured": measured,
        "imputed": imputed,
        "wildTypeCells": wild_type_cells,
        "measuredCount": len(measured),
        "imputedCount": len(imputed),
        # Fitness is relative to wild type at 0.0, so the neutral colour has to sit on
        # zero. A raw min-to-max ramp would place it wherever the data happened to fall.
        "bound": bound or 1.0,
        "scope": scope,
        "assayedCount": len(assayed),
        "sequenceLength": len(sequence),
        "totalPositions": total_positions,
        "truncated": total_positions > len(kept),
    }


def score_histogram(source: dict[str, Any], bins: int = 40) -> dict[str, Any]:
    """Bin the uploaded fitness values, split by how many substitutions each carries."""
    singles = [float(row["fitness"]) for row in source["dmsRows"]]
    higher = [
        float(row["fitness"]) for row in source.get("observationRows", []) if row["depth"] > 1
    ]
    values = singles + higher
    if not values:
        raise ValueError("source has no fitness values")

    low = min(min(values), 0.0)
    high = max(max(values), 0.0)
    if high - low < 1e-9:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, bins + 1)
    single_counts, _ = np.histogram(singles, bins=edges)
    higher_counts, _ = np.histogram(higher, bins=edges) if higher else (np.zeros(bins, int), None)
    centres = (edges[:-1] + edges[1:]) / 2

    return {
        "centres": [round(float(value), 6) for value in centres],
        "binWidth": float(edges[1] - edges[0]),
        "singles": [int(value) for value in single_counts],
        "higherOrder": [int(value) for value in higher_counts],
        "singleCount": len(singles),
        "higherOrderCount": len(higher),
        # Every fitness in the source is already expressed relative to wild type, so the
        # marker belongs at zero rather than at the raw score the upload carried.
        "wildType": 0.0,
        "median": float(np.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


@dataclass(frozen=True)
class ParsedDms:
    """One DMS upload reduced to the normal form the oracle builder consumes."""

    schema: str
    rows: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    wild_type_fitness: float
    input_row_count: int
    skipped: dict[str, int]
    wild_type_assumed: bool


SCHEMA_ROLES: dict[str, dict[str, tuple[str, ...]]] = {
    "singles": {
        "position": ("position", "pos", "site", "residue", "resi", "aa_pos", "aa_index"),
        "wildtype": ("wildtype", "wild_type", "wt", "wt_aa", "from", "ref", "ref_aa", "parent"),
        "mutant": ("mutant", "mut", "mut_aa", "to", "alt", "alt_aa", "substitution"),
        "fitness": ("fitness", "score", "value", "dms_score", "functional_score", "effect"),
    },
    "variants": {
        "variant": ("variant", "mutant", "mutations", "genotype", "label", "aa_substitutions"),
        "fitness": ("fitness", "score", "value", "dms_score", "functional_score", "effect"),
    },
    "hgvs": {
        "hgvs": ("hgvs_pro", "hgvs_p", "hgvsp", "hgvs", "protein_variant", "protein_change"),
        "score": ("score", "fitness", "value", "functional_score", "dms_score", "effect"),
    },
}
SCHEMA_LABELS = {
    "singles": "One row per substitution",
    "variants": "One row per variant, WT included",
    "hgvs": "HGVS protein notation, MaveDB style",
}
ROLE_LABELS = {
    "position": "Position",
    "wildtype": "Parent residue",
    "mutant": "Substituted residue",
    "fitness": "Fitness",
    "variant": "Variant label",
    "hgvs": "HGVS protein",
    "score": "Score",
}
THREE_LETTER_CODES = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
}  # fmt: skip
STOP_CODES = {"ter", "*", "stop"}
_HGVS_TERM = re.compile(r"^([A-Za-z]{3}|[A-Z*])(\d+)([A-Za-z]{3}|[A-Z*=])$")
_WILD_TYPE_LABELS = {"_wt", "wt", "p.=", "p.(=)", "_sy", "="}


class UnsupportedVariant(ValueError):
    """A row naming a variant the landscape model cannot represent."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def inspect_dms_csv(text: str, sample_size: int = 6) -> dict[str, Any]:
    """Report the headers of an upload and the column mapping we can guess from them."""
    try:
        reader = csv.DictReader(io.StringIO(text))
        headers = list(reader.fieldnames or ())
        sample = [row for _, row in zip(range(sample_size), reader, strict=False)]
        row_count = len(sample) + sum(1 for _ in reader)
    except csv.Error as exc:
        raise ValueError(f"cannot parse DMS CSV: {exc}") from exc
    if not headers:
        raise ValueError("DMS CSV has no header row")

    schemas = []
    for schema, roles in SCHEMA_ROLES.items():
        guesses = {role: _guess_column(headers, hints) for role, hints in roles.items()}
        schemas.append(
            {
                "id": schema,
                "label": SCHEMA_LABELS[schema],
                "complete": all(guesses.values()),
                "roles": [
                    {"id": role, "label": ROLE_LABELS[role], "guess": guesses[role]}
                    for role in roles
                ],
            }
        )
    detected = next((entry for entry in schemas if entry["complete"]), None)
    return {
        "headers": headers,
        "rowCount": row_count,
        "sampleRows": [[str(row.get(header, "")) for header in headers] for row in sample],
        "schemas": schemas,
        "detected": (
            None
            if detected is None
            else {
                "schema": detected["id"],
                "columns": {role["id"]: role["guess"] for role in detected["roles"]},
            }
        ),
    }


def _guess_column(headers: list[str], hints: tuple[str, ...]) -> str | None:
    def normalise(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    normalised = {normalise(header): header for header in reversed(headers)}
    for hint in hints:
        match = normalised.get(normalise(hint))
        if match is not None:
            return match
    for hint in hints:
        for header in headers:
            if normalise(hint) in normalise(header):
                return header
    return None


def _resolve_column_map(headers: list[str], column_map: dict[str, Any] | None) -> tuple[str, dict]:
    if column_map is None:
        fields = set(headers)
        if {"variant", "fitness"}.issubset(fields):
            return "variants", {"variant": "variant", "fitness": "fitness"}
        if {"position", "wildtype", "mutant", "fitness"}.issubset(fields):
            return "singles", {role: role for role in SCHEMA_ROLES["singles"]}
        detected = next(
            (
                entry
                for entry in (
                    {
                        "id": schema,
                        "columns": {
                            role: _guess_column(headers, hints) for role, hints in roles.items()
                        },
                    }
                    for schema, roles in SCHEMA_ROLES.items()
                )
                if all(entry["columns"].values())
            ),
            None,
        )
        if detected is None:
            raise ValueError(
                "cannot recognise the DMS columns, map them by hand: " + ", ".join(headers[:12])
            )
        return detected["id"], detected["columns"]

    schema = str(column_map.get("schema", ""))
    if schema not in SCHEMA_ROLES:
        raise ValueError(f"unknown DMS schema {schema!r}")
    columns = dict(column_map.get("columns") or {})
    resolved = {}
    for role in SCHEMA_ROLES[schema]:
        header = columns.get(role)
        if not header:
            raise ValueError(f"map a column to {ROLE_LABELS[role].lower()}")
        if header not in headers:
            raise ValueError(f"column {header!r} is not in the CSV")
        resolved[role] = header
    return schema, resolved


def _parse_dms_input(
    text: str,
    wt_sequence: str,
    column_map: dict[str, Any] | None = None,
    wild_type_score: float | None = None,
) -> ParsedDms:
    try:
        fieldnames = csv.DictReader(io.StringIO(text)).fieldnames
    except csv.Error as exc:
        raise ValueError(f"cannot parse DMS CSV: {exc}") from exc
    headers = list(fieldnames or ())
    if not headers:
        raise ValueError("DMS CSV has no header row")
    schema, columns = _resolve_column_map(headers, column_map)
    if schema == "singles":
        return _parse_single_rows(text, wt_sequence, columns)
    return _parse_variant_rows(text, wt_sequence, schema, columns, wild_type_score)


def _parse_single_rows(text: str, wt_sequence: str, columns: dict[str, str]) -> ParsedDms:
    rows = _parse_dms_csv(text, wt_sequence, columns)
    observations = [{"variant": "wt", "mutations": {}, "fitness": 0.0, "depth": 0}]
    observations.extend(
        {
            "variant": f"{row['wildtype']}{row['position']}{row['mutant']}",
            "mutations": {str(row["position"]): row["mutant"]},
            "fitness": float(row["fitness"]),
            "depth": 1,
        }
        for row in rows
    )
    return ParsedDms("singles", rows, observations, 0.0, len(rows), {}, False)


def _parse_hgvs_pro(value: str) -> tuple[list[tuple[int, str]], dict[int, str]]:
    """Return substitutions and the parent residues the label claims for them."""
    raw = value.strip()
    if not raw or raw.upper() in {"NA", "NAN", "NONE"}:
        raise UnsupportedVariant("blank")
    if raw.lower() in _WILD_TYPE_LABELS:
        return [], {}
    body = raw[2:] if raw.lower().startswith("p.") else raw
    body = body.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    if not body or body == "=":
        return [], {}
    if any(token in body for token in ("fs", "del", "ins", "dup", "ext", "?")):
        raise UnsupportedVariant("indel or frameshift")

    mutations: list[tuple[int, str]] = []
    parents: dict[int, str] = {}
    for part in re.split(r"[;,]", body):
        term = part.strip()
        if not term:
            continue
        match = _HGVS_TERM.match(term)
        if match is None:
            raise UnsupportedVariant("unreadable")
        old, site, new = match.groups()
        position = int(site)
        parent = _one_letter(old)
        if parent:
            parents[position] = parent
        if new == "=":
            continue
        if new.lower() in STOP_CODES:
            raise UnsupportedVariant("nonsense")
        amino_acid = _one_letter(new)
        if amino_acid not in CANONICAL_AAS:
            raise UnsupportedVariant("non-canonical residue")
        mutations.append((position, amino_acid))
    return mutations, parents


def _one_letter(code: str) -> str:
    if len(code) == 1:
        return code.upper()
    return THREE_LETTER_CODES.get(code.lower(), "")


def _parse_variant_rows(
    text: str,
    wt_sequence: str,
    schema: str,
    columns: dict[str, str],
    wild_type_score: float | None,
) -> ParsedDms:
    label_column = columns["hgvs"] if schema == "hgvs" else columns["variant"]
    score_column = columns["score"] if schema == "hgvs" else columns["fitness"]
    strict = schema == "variants"

    reader = csv.DictReader(io.StringIO(text))
    records: dict[tuple[tuple[int, str], ...], float] = {}
    skipped: dict[str, int] = {}
    input_rows = 0

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row_number, row in enumerate(reader, start=2):
        input_rows += 1
        raw_score = str(row.get(score_column, "")).strip()
        try:
            fitness = float(raw_score)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"invalid fitness on row {row_number}: {raw_score!r}") from exc
            skip("no score")
            continue
        if not np.isfinite(fitness):
            if strict:
                raise ValueError(f"DMS fitness on row {row_number} must be finite")
            skip("no score")
            continue

        label = str(row.get(label_column, "")).strip()
        try:
            mutations, parents = (
                _parse_hgvs_pro(label) if schema == "hgvs" else _mutations_from_label(label)
            )
            key: tuple[tuple[int, str], ...] = tuple(sorted(mutations))
            _check_positions(key, wt_sequence, parents)
        except UnsupportedVariant as exc:
            if strict:
                raise ValueError(f"invalid variant on row {row_number}: {exc.reason}") from exc
            skip(exc.reason)
            continue
        if key in records:
            if strict:
                raise ValueError(f"DMS has a duplicate measurement for {label or 'WT'}")
            skip("duplicate")
            continue
        records[key] = fitness

    if not records:
        raise ValueError("DMS CSV has no usable data rows")

    assumed = False
    if wild_type_score is not None:
        baseline = float(wild_type_score)
    elif () in records:
        baseline = records[()]
    elif strict:
        raise ValueError("variant DMS CSV must include a WT row")
    else:
        # MaveDB score sets rarely carry an explicit wild-type row, and the convention for
        # where wild type sits varies by score set. Zero matches the package convention and
        # the import step shows the score distribution so a wrong baseline is visible.
        baseline = 0.0
        assumed = True
    records.setdefault((), baseline)

    single_keys = {key for key in records if len(key) == 1}
    incomplete = [key for key in records if any((mutation,) not in single_keys for mutation in key)]
    if incomplete and strict:
        first = incomplete[0]
        labels = ", ".join(
            f"{wt_sequence[position - 1]}{position}{amino_acid}" for position, amino_acid in first
        )
        raise ValueError(f"variant DMS is missing constituent singles: {labels}")
    for key in incomplete:
        del records[key]
        skip("missing constituent singles")

    rows = []
    observations = []
    for key, fitness in sorted(records.items(), key=lambda item: (len(item[0]), item[0])):
        wild_type = {position: wt_sequence[position - 1] for position, _ in key}
        observations.append(
            {
                "variant": Mutant.from_dict(dict(key), wild_type).label if key else "wt",
                "mutations": {str(position): amino_acid for position, amino_acid in key},
                "fitness": float(fitness - baseline),
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
                    "fitness": float(fitness - baseline),
                }
            )
    if not rows:
        raise ValueError("DMS CSV must include at least one single mutant")
    return ParsedDms(schema, rows, observations, float(baseline), input_rows, skipped, assumed)


def _mutations_from_label(label: str) -> tuple[list[tuple[int, str]], dict[int, str]]:
    text = label.strip().upper()
    if not text:
        raise UnsupportedVariant("unreadable")
    if text in {"WT", "_WT"}:
        return [], {}
    try:
        mutant = Mutant(text)
    except (TypeError, ValueError) as exc:
        raise UnsupportedVariant("unreadable") from exc
    return list(mutant.items()), dict(mutant._wild_type)


def _check_positions(
    key: tuple[tuple[int, str], ...],
    wt_sequence: str,
    parents: dict[int, str],
) -> None:
    for position, amino_acid in key:
        if position < 1 or position > len(wt_sequence):
            raise UnsupportedVariant("outside the WT sequence")
        if amino_acid not in CANONICAL_AAS:
            raise UnsupportedVariant("non-canonical residue")
    for position, parent in parents.items():
        if position < 1 or position > len(wt_sequence):
            raise UnsupportedVariant("outside the WT sequence")
        if parent != wt_sequence[position - 1]:
            raise UnsupportedVariant(f"wild type {parent}{position} does not match the WT FASTA")


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


def _parse_dms_csv(
    text: str,
    wt_sequence: str,
    columns: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(text))
    except csv.Error as exc:
        raise ValueError(f"cannot parse DMS CSV: {exc}") from exc
    columns = columns or {role: role for role in SCHEMA_ROLES["singles"]}
    if reader.fieldnames is None or not set(columns.values()).issubset(reader.fieldnames):
        raise ValueError("DMS CSV needs position, wildtype, mutant, and fitness columns")

    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    wild_type_by_position: dict[int, str] = {}
    for row_number, row in enumerate(reader, start=2):
        try:
            position = int(row[columns["position"]])
            wildtype = str(row[columns["wildtype"]]).strip().upper()
            mutant = str(row[columns["mutant"]]).strip().upper()
            fitness = float(row[columns["fitness"]])
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
        "gb1_headline": (
            "Every four-site GB1 variant was measured, so the synthetic landscape can be "
            "scored against ground truth. Wu et al. 2016."
        ),
        "trpb_headline": (
            "Every four-site TrpB variant was measured, so the synthetic landscape can be "
            "scored against ground truth. Johnston et al. 2024."
        ),
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
    creilov_source = reference_posterior_source(CREILOV_POSTERIOR_ID)
    entries.append(
        {
            "id": CREILOV_POSTERIOR_ID,
            "name": "CreiLOV DMS-pairwise example",
            "kind": "posterior",
            "description": (
                "Built from CreiLOV wild type and 2,121 measured single mutants across the "
                "protein. The 15-position combinatorial assay remains hidden from the oracle."
            ),
            "positions": list(creilov_source["positions"]),
            "wtCompact": creilov_source["wtCompact"],
            "parameters": BenchmarkParameters(seed=42).as_public_json(),
            "posteriorSettings": {
                "priorId": DEFAULT_PRIOR_ID,
                "oracleSeed": 42,
            },
            "createdAt": None,
        }
    )
    for model_id, base_id in POSTERIOR_REFERENCE_CONFIGS.items():
        config = REFERENCE_CONFIGS[base_id]
        protein = "GB1" if model_id.startswith("gb1") else "TrpB"
        entries.append(
            {
                "id": model_id,
                "name": f"{protein} DMS-pairwise example",
                "kind": "posterior",
                "description": (
                    f"Built from the {protein} wild type and measured single mutants. "
                    "The measured combinations remain hidden so the synthetic oracle can be "
                    "compared with them."
                ),
                "positions": list(config.positions),
                "wtCompact": "".join(config.wt_aa_at_position[p] for p in config.positions),
                "parameters": _reference_parameters(config).as_public_json(),
                "posteriorSettings": {
                    "priorId": DEFAULT_PRIOR_ID,
                    "oracleSeed": config.seed,
                },
                "createdAt": None,
            }
        )
    return entries


def reference_detail(model_id: str) -> dict[str, Any] | None:
    if (
        model_id not in REFERENCE_CONFIGS
        and model_id not in POSTERIOR_REFERENCE_CONFIGS
        and model_id != CREILOV_POSTERIOR_ID
    ):
        return None
    return next(item for item in reference_catalogue_entries() if item["id"] == model_id)


@lru_cache(maxsize=3)
def reference_posterior_source(model_id: str) -> dict[str, Any]:
    """Return WT and single-mutant observations for a reference model."""
    if model_id == CREILOV_POSTERIOR_ID:
        source, _ = parse_custom_source(
            CREILOV_DMS_CSV.read_text(),
            CREILOV_FASTA.read_text(),
        )
        return source
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


class Progress(Protocol):
    """Reports which named stage of a build is running."""

    def __call__(self, stage: str) -> None: ...


def _noop_progress(stage: str) -> None:  # noqa: ARG001
    return None


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
    progress: Progress = _noop_progress,
) -> dict[str, Any]:
    parameters.validate()
    progress("Loading measured data")
    artifacts = _reference_artifacts(model_id)
    config = REFERENCE_CONFIGS[model_id]
    progress("Building the oracle")
    landscape = build_reference_landscape(model_id, parameters)
    progress("Scoring the combinatorial grid")
    synth_tensor = fitness_tensor_from_landscape(
        landscape,
        config.positions,
        artifacts["alphabets"],
    )
    progress("Decomposing interaction orders")
    synth_walsh = walsh_decompose(synth_tensor)
    progress("Measuring landscape topology")
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
    progress("Sampling variants")
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
            f"{artifacts['coverage'] * 100:.1f}% of the combinatorial grid was measured. "
            "Unmeasured cells take the mean of the measured ones before the comparison "
            "metrics run, so the measured and synthetic sides use the same grid shape.",
        ],
    }


def preview_custom(
    source: dict[str, Any],
    parameters: BenchmarkParameters,
    sample_size: int,
    depth_weights: dict[int, float],
    model_id: str | None = None,
    name: str | None = None,
    progress: Progress = _noop_progress,
) -> dict[str, Any]:
    parameters.validate()
    progress("Building the oracle")
    landscape = build_custom_landscape(source, parameters)
    progress("Sampling variants")
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


def _builder_fingerprint() -> str:
    """Digest of the modules that shape a preview response.

    Whole responses are cached, so any edit to how one is built, down to the wording of a
    notice, has to miss every entry written by the previous code.
    """
    digest = hashlib.sha256()
    for name in ("models.py", "posterior.py"):
        try:
            digest.update((Path(__file__).parent / name).read_bytes())
        except OSError:
            digest.update(name.encode())
    return digest.hexdigest()[:16]


_BUILDER_FINGERPRINT = _builder_fingerprint()


def preview_cache_key(payload: dict[str, Any]) -> str:
    versioned = {"responseVersion": _BUILDER_FINGERPRINT, "request": payload}
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
