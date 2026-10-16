"""Fetch the pinned external benchmark corpora outside Git."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from mutascape.benchmarks import roots
from mutascape.benchmarks.roots import SSMULA_DIRECTORY, data_root

# SSMuLA ships one archive per Zenodo record rather than a repository snapshot.
SSMULA_ARCHIVE_URL = "https://zenodo.org/api/records/15203754/files/data.tar.gz/content"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=data_root(),
    )
    parser.add_argument(
        "--source",
        choices=["all", *(corpus.directory for corpus in roots.CORPORA)],
        default="all",
    )
    return parser.parse_args()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(path)


def _safe_extract(archive: Path, destination: Path, *, strip_first: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            parts = Path(member.name).parts
            if strip_first:
                parts = parts[1:]
            if not parts:
                continue
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"archive member leaves destination: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _fetch_ssmula(destination: Path) -> None:
    root = destination / SSMULA_DIRECTORY
    marker = root / "source.json"
    if marker.exists():
        return
    archive = destination / "downloads" / "ssmula-data.tar.gz"
    if not archive.exists():
        _download(SSMULA_ARCHIVE_URL, archive)
    checksum = _md5(archive)
    if checksum != roots.SSMULA.md5:
        raise ValueError(f"SSMuLA checksum mismatch: {checksum}")
    _safe_extract(archive, root)
    marker.write_text(
        json.dumps(
            {"url": roots.SSMULA.url, "md5": checksum, "revision": roots.SSMULA.revision},
            indent=2,
        )
        + "\n"
    )


def _fetch_repository(destination: Path, corpus: roots.Corpus) -> None:
    """Unpack one pinned GitHub snapshot, skipping work that already succeeded."""
    root = destination / corpus.directory
    marker = root / "source.json"
    if marker.exists():
        return
    archive = destination / "downloads" / f"{corpus.directory}-{corpus.revision}.tar.gz"
    if not archive.exists():
        _download(corpus.archive_url, archive)
    _safe_extract(archive, root, strip_first=True)
    marker.write_text(json.dumps({"url": corpus.url, "revision": corpus.revision}, indent=2) + "\n")


def main() -> None:
    args = _parse_args()
    if args.source in {"all", SSMULA_DIRECTORY}:
        _fetch_ssmula(args.destination)
    for corpus in (roots.COMBINGYM, roots.GRAPHFLA, roots.NN4DMS):
        if args.source in {"all", corpus.directory}:
            _fetch_repository(args.destination, corpus)


if __name__ == "__main__":
    main()
