"""Canonical locations of the external benchmark corpora.

Every script reads a corpus through these helpers rather than its own literal
path, so a corpus moves in one place. ``MUTASCAPE_DATA_ROOT`` overrides the
default for a checkout that keeps the downloads elsewhere.
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path(".mutascape/benchmark-data")
DATA_ROOT_VARIABLE = "MUTASCAPE_DATA_ROOT"

# Each corpus is fetched by scripts/fetch_benchmark_data.py under this name.
SSMULA_DIRECTORY = "ssmula"
COMBINGYM_DIRECTORY = "combingym"
GRAPHFLA_DIRECTORY = "graphfla"
NN4DMS_DIRECTORY = "nn4dms"


@dataclass(frozen=True)
class Corpus:
    """One pinned external collection, and where it comes from.

    The pin is the provenance claim a dataset carries, and the thing the
    fetcher downloads, so both read it from here.
    """

    directory: str
    url: str
    revision: str = ""
    md5: str = ""

    @property
    def archive_url(self) -> str:
        """Pinned tarball for a repository snapshot."""
        if not self.revision:
            raise ValueError(f"{self.directory} is not pinned to a revision")
        return f"{self.url}/archive/{self.revision}.tar.gz"


SSMULA = Corpus(
    directory=SSMULA_DIRECTORY,
    url="https://zenodo.org/records/15203754",
    revision="4926b8e614d426ca567159988834482c23da13a3",
    md5="d94418e5865949b1e2bf9af49fe9438a",
)
COMBINGYM = Corpus(
    directory=COMBINGYM_DIRECTORY,
    url="https://github.com/sitonglab/CombinGym",
    revision="e635a959731d2a14f90269245ccbec4bf2645324",
)
GRAPHFLA = Corpus(
    directory=GRAPHFLA_DIRECTORY,
    url="https://github.com/COLA-Laboratory/GraphFLA",
    revision="9f86def1a12db76d26af16077ba28cb61ee0af5f",
)
NN4DMS = Corpus(
    directory=NN4DMS_DIRECTORY,
    url="https://github.com/gitter-lab/nn4dms",
    revision="434e354828f306cbb6796033ef1169cbe5207350",
)

CORPORA = (SSMULA, COMBINGYM, GRAPHFLA, NN4DMS)

_SOURCE_FLAGS = {
    SSMULA_DIRECTORY: "ssmula",
    COMBINGYM_DIRECTORY: "combingym",
    GRAPHFLA_DIRECTORY: "graphfla",
    NN4DMS_DIRECTORY: "nn4dms",
}


def data_root() -> Path:
    """Return the directory holding every fetched benchmark corpus."""
    override = os.environ.get(DATA_ROOT_VARIABLE)
    return Path(override) if override else DEFAULT_DATA_ROOT


def corpus(name: str, *, require: bool = True) -> Path:
    """Return one corpus directory, naming the command that creates it."""
    path = data_root() / name
    if require and not path.is_dir():
        flag = _SOURCE_FLAGS.get(name, name)
        raise FileNotFoundError(
            f"benchmark corpus {name!r} is missing from {data_root()}. "
            f"Run `just benchmark-data --source {flag}`, or set "
            f"{DATA_ROOT_VARIABLE} to the directory that holds it."
        )
    return path


def ssmula(*, require: bool = True) -> Path:
    """SSMuLA release (Zenodo record 15203754)."""
    return corpus(SSMULA_DIRECTORY, require=require)


def combingym(*, require: bool = True) -> Path:
    """CombinGym repository snapshot."""
    return corpus(COMBINGYM_DIRECTORY, require=require)


def graphfla(*, require: bool = True) -> Path:
    """GraphFLA repository snapshot, which supplies the RBD and PhoQ landscapes."""
    return corpus(GRAPHFLA_DIRECTORY, require=require)


def nn4dms(*, require: bool = True) -> Path:
    """nn4dms repository snapshot."""
    return corpus(NN4DMS_DIRECTORY, require=require)
