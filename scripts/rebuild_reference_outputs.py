"""Rebuild the current manuscript from separate fidelity and search experiments."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if "--legacy-reference" in sys.argv:
        subprocess.run(
            [sys.executable, "-m", "scripts.rebuild_preprint", *sys.argv[1:]], cwd=root, check=True
        )
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=root.parent / "mutascape_manu_v1")
    parser.add_argument(
        "--fidelity-results", type=Path, default=root / "results/manuscript_v2_20261015"
    )
    parser.add_argument(
        "--search-results", type=Path, default=root / "results/calibration_confidence_20261119"
    )
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(args.manuscript / "scripts/build_v2_assets.py"),
            "--results",
            str(args.fidelity_results),
            "--search-results",
            str(args.search_results),
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/analyse_generator_agreement.py",
            "--input",
            str(args.search_results / "search_summary.csv"),
            "--output",
            str(args.search_results / "inference"),
            "--manuscript-dir",
            str(args.manuscript),
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(["typst", "compile", "main.typ", "main.pdf"], cwd=args.manuscript, check=True)


if __name__ == "__main__":
    main()
