"""Read-only client for the MaveDB score-set API."""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_ROOT = "https://api.mavedb.org/api/v1"
RECORD_URL = "https://www.mavedb.org/score-sets"
MAX_SCORES_BYTES = 20_000_000
TIMEOUT_SECONDS = 45
_URN = re.compile(r"urn:mavedb:[0-9A-Za-z\-]+(?:-[0-9A-Za-z]+)*", re.IGNORECASE)


class MaveDbError(Exception):
    """A MaveDB record that cannot be imported, with a message fit for the interface."""


def parse_urn(value: str) -> str:
    """Accept a bare URN, a MaveDB URL, or a URL with query and fragment."""
    text = (value or "").strip()
    if not text:
        raise MaveDbError("enter a MaveDB score-set URN or URL")
    unquoted = urllib.parse.unquote(text)
    match = _URN.search(unquoted)
    if match is None:
        raise MaveDbError(f"cannot find a MaveDB URN in {text!r}")
    urn = match.group(0)
    # A variant link carries the score-set URN plus a #123 suffix, and the fragment survives
    # unquoting as part of the path. Trailing hash digits are never part of a score-set URN.
    return re.sub(r"#\d+$", "", urn)


def fetch_score_set(urn: str, opener: Any = None) -> dict[str, Any]:
    """Fetch one score set, its target protein sequence, and its scores CSV."""
    urn = parse_urn(urn)
    metadata = _get_json(f"{API_ROOT}/score-sets/{urllib.parse.quote(urn, safe=':')}", opener)
    genes = metadata.get("targetGenes") or []
    if not genes:
        raise MaveDbError(f"{urn} has no target gene, so there is no wild-type sequence")
    if len(genes) > 1:
        raise MaveDbError(f"{urn} has {len(genes)} target genes, which the studio cannot map")

    gene = genes[0]
    target = gene.get("targetSequence") or {}
    sequence = str(target.get("sequence") or "").strip().upper()
    sequence_type = str(target.get("sequenceType") or "").lower()
    if not sequence:
        raise MaveDbError(f"{urn} carries no target sequence")
    if sequence_type != "protein":
        raise MaveDbError(
            f"{urn} stores a {sequence_type or 'unknown'} target sequence. "
            "The studio needs a protein target, so upload the CSV with its own WT FASTA."
        )

    quoted = urllib.parse.quote(urn, safe=":")
    scores_csv = _get_text(f"{API_ROOT}/score-sets/{quoted}/scores", opener)
    label = str(gene.get("name") or metadata.get("title") or urn)
    return {
        "urn": urn,
        "title": str(metadata.get("title") or urn),
        "shortDescription": str(metadata.get("shortDescription") or ""),
        "targetName": label,
        "recordUrl": f"{RECORD_URL}/{urn}",
        "numVariants": int(metadata.get("numVariants") or 0),
        "license": str((metadata.get("license") or {}).get("shortName") or ""),
        "publications": [
            str(entry.get("url") or entry.get("identifier") or "")
            for entry in (metadata.get("primaryPublicationIdentifiers") or [])
        ],
        "wtFasta": f">{label} {urn}\n" + "\n".join(_wrap(sequence)) + "\n",
        "dmsCsv": scores_csv,
    }


def _wrap(sequence: str, width: int = 60) -> list[str]:
    return [sequence[start : start + width] for start in range(0, len(sequence), width)]


def _open(url: str, opener: Any) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": "mutascape"})
    try:
        return (opener or urllib.request.urlopen)(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise MaveDbError("MaveDB has no published record for that URN") from exc
        raise MaveDbError(f"MaveDB returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MaveDbError(f"cannot reach MaveDB: {exc}") from exc


def _get_json(url: str, opener: Any) -> dict[str, Any]:
    with _open(url, opener) as response:
        try:
            payload = json.loads(response.read().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise MaveDbError("MaveDB returned a response we cannot read") from exc
    if not isinstance(payload, dict):
        raise MaveDbError("MaveDB returned an unexpected record shape")
    return payload


def _get_text(url: str, opener: Any) -> str:
    with _open(url, opener) as response:
        raw = response.read(MAX_SCORES_BYTES + 1)
    if len(raw) > MAX_SCORES_BYTES:
        raise MaveDbError("that score set is larger than the 20 MB import limit")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MaveDbError("MaveDB returned a scores file we cannot decode") from exc
