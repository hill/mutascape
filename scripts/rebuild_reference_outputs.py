"""Rebuild the current manuscript from saved results and prepared assay data."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "scripts.rebuild_preprint", *sys.argv[1:]],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


if __name__ == "__main__":
    main()
