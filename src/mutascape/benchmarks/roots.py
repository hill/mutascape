"""Canonical locations of the external benchmark corpora.

Every script reads a corpus through these helpers rather than its own literal
path, so a corpus moves in one place. ``MUTASCAPE_DATA_ROOT`` overrides the
default for a checkout that keeps the downloads elsewhere.
"""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(".mutascape/benchmark-data")
DATA_ROOT_VARIABLE = "MUTASCAPE_DATA_ROOT"

# Each corpus is fetched by scripts/fetch_benchmark_data.py under this name.
SSMULA_DIRECTORY = "ssmula"
COMBINGYM_DIRECTORY = "combingym"
GRAPHFLA_DIRECTORY = "graphfla"
NN4DMS_DIRECTORY = "nn4dms"

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
