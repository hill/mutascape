"""Structures for the studio viewer.

Three ways in. Fetch a deposited structure from RCSB by ID, take an uploaded file, or
fold the wild-type sequence with ESMFold. All three end at :func:`describe_structure`,
which aligns the model against the wild-type sequence so a DMS position can be drawn on
the right residue.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

from mutascape.contacts import _predict_esmfold

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/"
PDB_ID_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
MAX_STRUCTURE_BYTES = 20_000_000
ESMFOLD_MAX_RESIDUES = 400

# Below this many overlapping residues an identity score means nothing, and the best
# shift is whatever short run happens to match.
MIN_ALIGNMENT_OVERLAP = 8


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "mutascape" / "structures"


def fetch_rcsb(pdb_id: str, cache_dir: str | Path | None = None) -> tuple[str, str, str]:
    """Download one deposited structure. Returns the text, its format, and the ID.

    Large assemblies are only distributed as mmCIF, so a missing `.pdb` is a normal
    outcome rather than a bad ID.
    """
    identifier = pdb_id.strip().upper()
    if not PDB_ID_PATTERN.match(identifier):
        raise ValueError("a PDB ID is four characters, starting with a digit, for example 1PGA")

    directory = Path(cache_dir) if cache_dir else default_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)

    for suffix in ("pdb", "cif"):
        cached = directory / f"rcsb_{identifier}.{suffix}"
        if cached.exists():
            return cached.read_text(), suffix, identifier

    for suffix in ("pdb", "cif"):
        try:
            with urlopen(f"{RCSB_DOWNLOAD_URL}{identifier}.{suffix}", timeout=30) as response:
                payload = response.read(MAX_STRUCTURE_BYTES + 1)
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise ValueError(f"RCSB returned status {exc.code} for {identifier}") from exc
        except Exception as exc:
            raise ValueError(f"cannot reach RCSB: {exc}") from exc
        if len(payload) > MAX_STRUCTURE_BYTES:
            raise ValueError(f"{identifier} is larger than 20 MB")
        text = payload.decode("utf-8", errors="replace")
        (directory / f"rcsb_{identifier}.{suffix}").write_text(text)
        return text, suffix, identifier

    raise ValueError(f"no entry {identifier} at RCSB")


def fold_with_esmfold(sequence: str, cache_dir: str | Path | None = None) -> str:
    """Fold one sequence through the public ESM Atlas API.

    Shares its cache with :meth:`ContactMap.from_sequence`, so a landscape built with
    ESMFold contacts and the viewer never fold the same sequence twice.
    """
    if len(sequence) > ESMFOLD_MAX_RESIDUES:
        raise ValueError(f"ESMFold supports sequences up to {ESMFOLD_MAX_RESIDUES} residues")

    directory = Path(cache_dir) if cache_dir else default_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / f"esmfold_{hashlib.sha256(sequence.encode()).hexdigest()[:16]}.pdb"

    if not cached.exists():
        try:
            cached.write_text(_predict_esmfold(sequence))
        except Exception as exc:
            raise ValueError(f"ESMFold failed: {exc}") from exc
    return cached.read_text()


def _one_letter(residue_name: str) -> str | None:
    return protein_letters_3to1.get(residue_name.strip().upper())


def _chain_sequences(text: str, suffix: str) -> dict[str, tuple[str, list[int]]]:
    """Return the one-letter sequence and residue numbers of every protein chain."""
    with tempfile.TemporaryDirectory(prefix="mutascape-viewer-") as directory:
        path = Path(directory) / f"structure.{suffix}"
        path.write_text(text)
        parser = MMCIFParser(QUIET=True) if suffix in {"cif", "mmcif"} else PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("protein", str(path))
        except Exception as exc:
            raise ValueError(f"cannot parse structure: {exc}") from exc

    chains: dict[str, tuple[str, list[int]]] = {}
    for model in structure:
        for chain in model:
            letters: list[str] = []
            numbers: list[int] = []
            for residue in chain:
                # Water and ligands share the chain and would otherwise land in the
                # sequence as gaps that shift every downstream residue.
                if residue.id[0] != " ":
                    continue
                letter = _one_letter(residue.get_resname())
                if letter is None:
                    continue
                letters.append(letter)
                numbers.append(int(residue.id[1]))
            if letters:
                chains[chain.id] = ("".join(letters), numbers)
        break
    if not chains:
        raise ValueError("the structure holds no protein chain")
    return chains


def _best_shift(wt_sequence: str, chain_sequence: str) -> tuple[int, float]:
    """Find the ungapped offset that best lines a chain up against the wild type.

    Returns the shift where chain index `j` matches wild-type index `j + shift`, and the
    identity over the overlap. Deposited constructs carry tags, truncations and their own
    numbering, so the two sequences rarely start at the same residue.
    """
    exact = wt_sequence.find(chain_sequence)
    if exact >= 0:
        return exact, 1.0
    exact = chain_sequence.find(wt_sequence)
    if exact >= 0:
        return -exact, 1.0

    best_shift = 0
    best_score = -1.0
    for shift in range(-len(chain_sequence) + 1, len(wt_sequence)):
        start = max(0, -shift)
        stop = min(len(chain_sequence), len(wt_sequence) - shift)
        overlap = stop - start
        if overlap < min(MIN_ALIGNMENT_OVERLAP, len(chain_sequence)):
            continue
        matches = sum(chain_sequence[j] == wt_sequence[j + shift] for j in range(start, stop))
        # Scale by overlap so a three-residue perfect match cannot beat a long good one.
        score = matches / overlap * min(1.0, overlap / len(chain_sequence))
        if score > best_score:
            best_shift, best_score = shift, matches / overlap
    return best_shift, max(best_score, 0.0)


def mean_effects(dms_rows: Iterable[dict[str, Any]]) -> dict[int, float]:
    """Mean measured single-mutant effect at every assayed position."""
    totals: dict[int, list[float]] = {}
    for row in dms_rows:
        totals.setdefault(int(row["position"]), []).append(float(row["fitness"]))
    return {position: sum(values) / len(values) for position, values in totals.items()}


def describe_structure(
    text: str,
    suffix: str,
    wt_sequence: str,
    positions: list[int],
    wild_type: dict[int, str],
    chain_id: str | None = None,
    effects: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Align a structure to the wild-type sequence and map every assayed position onto it."""
    if len(text.encode()) > MAX_STRUCTURE_BYTES:
        raise ValueError("structure must be at most 20 MB")

    chains = _chain_sequences(text, suffix)
    if chain_id and chain_id not in chains:
        raise ValueError(f"chain {chain_id} is not in this structure")
    if not chain_id:
        # The longest chain is the assayed protein far more often than the first one,
        # which in a complex is as likely to be a partner or a nanobody.
        chain_id = max(chains, key=lambda key: len(chains[key][0]))

    chain_sequence, residue_numbers = chains[chain_id]
    shift, identity = _best_shift(wt_sequence, chain_sequence)

    number_of: dict[int, int] = {}
    for index, number in enumerate(residue_numbers):
        wt_index = index + shift
        if 0 <= wt_index < len(wt_sequence):
            number_of[wt_index + 1] = number

    residues: list[dict[str, Any]] = []
    unmatched: list[int] = []
    for position in positions:
        number = number_of.get(position)
        parent = wild_type.get(position) or (
            wt_sequence[position - 1] if 0 < position <= len(wt_sequence) else ""
        )
        if number is None:
            unmatched.append(position)
            continue
        structure_residue = chain_sequence[position - 1 - shift]
        residues.append(
            {
                "position": position,
                "residue": number,
                "parent": parent,
                "structureResidue": structure_residue,
                "matched": structure_residue == parent,
                "effect": (effects or {}).get(position),
            }
        )

    values = [entry["effect"] for entry in residues if isinstance(entry["effect"], (int, float))]
    # Fitness is relative to wild type at 0.0, so the neutral colour has to sit on zero.
    bound = max((abs(value) for value in values), default=0.0) or 1.0

    return {
        "pdb": text,
        "format": "mmcif" if suffix in {"cif", "mmcif"} else "pdb",
        "chains": sorted(chains),
        "chainId": chain_id,
        "chainLength": len(chain_sequence),
        "residueCount": len(residue_numbers),
        "identity": round(identity, 4),
        "bound": bound,
        "residues": residues,
        "unmatched": unmatched,
        "mismatched": [entry["position"] for entry in residues if not entry["matched"]],
    }
