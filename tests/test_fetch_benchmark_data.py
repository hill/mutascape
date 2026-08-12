import io
import tarfile
from pathlib import Path

import pytest

from scripts.fetch_benchmark_data import _safe_extract


def _archive(path: Path, member_name: str, value: bytes = b"data") -> None:
    with tarfile.open(path, "w:gz") as handle:
        member = tarfile.TarInfo(member_name)
        member.size = len(value)
        handle.addfile(member, io.BytesIO(value))


def test_safe_extract_can_strip_archive_root(tmp_path: Path):
    archive = tmp_path / "source.tar.gz"
    _archive(archive, "project/data/value.txt")

    _safe_extract(archive, tmp_path / "output", strip_first=True)

    assert (tmp_path / "output/data/value.txt").read_text() == "data"


def test_safe_extract_rejects_parent_traversal(tmp_path: Path):
    archive = tmp_path / "source.tar.gz"
    _archive(archive, "../outside.txt")

    with pytest.raises(ValueError, match="leaves destination"):
        _safe_extract(archive, tmp_path / "output")
