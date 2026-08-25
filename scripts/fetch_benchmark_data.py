"""Fetch pinned SSMuLA and CombinGym benchmark data outside Git."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

SSMULA_URL = "https://zenodo.org/api/records/15203754/files/data.tar.gz/content"
SSMULA_MD5 = "d94418e5865949b1e2bf9af49fe9438a"
COMBINGYM_REVISION = "e635a959731d2a14f90269245ccbec4bf2645324"
COMBINGYM_URL = f"https://github.com/sitonglab/CombinGym/archive/{COMBINGYM_REVISION}.tar.gz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(".mutascape/benchmark-data"),
    )
    parser.add_argument(
        "--source",
        choices=["all", "ssmula", "combingym"],
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
    root = destination / "ssmula"
    marker = root / "source.json"
    if marker.exists():
        return
    archive = destination / "downloads" / "ssmula-data.tar.gz"
    if not archive.exists():
        _download(SSMULA_URL, archive)
    checksum = _md5(archive)
    if checksum != SSMULA_MD5:
        raise ValueError(f"SSMuLA checksum mismatch: {checksum}")
    _safe_extract(archive, root)
    marker.write_text(
        json.dumps({"url": SSMULA_URL, "md5": checksum, "record": "15203754"}, indent=2) + "\n"
    )


def _fetch_combingym(destination: Path) -> None:
    root = destination / "combingym"
    marker = root / "source.json"
    if marker.exists():
        return
    archive = destination / "downloads" / f"combingym-{COMBINGYM_REVISION}.tar.gz"
    if not archive.exists():
        _download(COMBINGYM_URL, archive)
    _safe_extract(archive, root, strip_first=True)
    marker.write_text(
        json.dumps({"url": COMBINGYM_URL, "revision": COMBINGYM_REVISION}, indent=2) + "\n"
    )


def main() -> None:
    args = _parse_args()
    if args.source in {"all", "ssmula"}:
        _fetch_ssmula(args.destination)
    if args.source in {"all", "combingym"}:
        _fetch_combingym(args.destination)


if __name__ == "__main__":
    main()
