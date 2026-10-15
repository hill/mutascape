"""Rebuild the v1 preprint from saved search trials and prepared assay data."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from mutascape.benchmarks import roots
from scripts.build_review_tables import build_control_tables, build_search_table
from scripts.plot_benchmark_validity import plot_budget_curve, plot_landscapes, plot_summary

ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> None:
    subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=ROOT.parent / "mutascape_manu_v1")
    parser.add_argument("--results", type=Path, default=ROOT / "results/preprint_audit_20260923")
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help=(
            "Directory of archived output_validity_* runs from the ICLR preprint. "
            "Those outputs were never tracked and no longer exist on disk; recover "
            "them by rebuilding branch research/benchmark-validity-v1."
        ),
    )
    parser.add_argument(
        "--legacy-reference",
        action="store_true",
        help="Rebuild the archived singles-only reference assets instead of the final model",
    )
    args = parser.parse_args()
    if not args.legacy_reference:
        _run(
            sys.executable,
            "-m",
            "scripts.build_singles_manuscript",
            "--manuscript",
            str(args.manuscript),
        )
        _run(
            "typst", "compile", str(args.manuscript / "main.typ"), str(args.manuscript / "main.pdf")
        )
        return
    figures = args.manuscript / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _run(
        sys.executable,
        "-m",
        "scripts.audit_preprint_results",
        "--archive",
        str(args.archive),
        "--output",
        str(args.results),
    )
    _run(sys.executable, "scripts/replay_c2st_audit.py", "results/creilov_c2st")
    _run(
        sys.executable,
        "scripts/run_creilov_shape_validation.py",
        "--ssmula-root",
        str(roots.ssmula()),
        "--combingym-root",
        str(roots.combingym()),
        "--output",
        str(args.results / "creilov"),
    )
    _run(sys.executable, "scripts/plot_walsh_explainer.py", "--output", str(figures))
    ssmula = [(b, args.archive / f"output_validity_b{b}_primary") for b in (48, 96, 192, 384)]
    graphfla = [(b, args.results / f"graphfla_b{b}") for b in (48, 96, 192, 384)]
    combingym = [(b, args.results / f"combingym_b{b}") for b in (48, 96, 192, 384)]
    plot_summary(dict(ssmula)[96], dict(combingym)[96], dict(graphfla)[96], figures)
    plot_landscapes(dict(ssmula)[96], figures)
    plot_budget_curve(ssmula, combingym, graphfla, figures)
    for extension in ("pdf", "png", "svg"):
        name = f"creilov_shape_validation.{extension}"
        shutil.copyfile(args.results / "creilov" / name, figures / name)

    _run(sys.executable, "-m", "scripts.assess_review_nulls")
    build_search_table(args.manuscript, args.results)
    _run(sys.executable, "-m", "scripts.summarise_review_experiments")
    build_control_tables(args.manuscript, args.results.parent / "review_response_20260923")
    _run(sys.executable, "-m", "scripts.freeze_review_response")
    code = [
        Path(__file__),
        ROOT / "scripts/audit_preprint_results.py",
        ROOT / "scripts/run_creilov_shape_validation.py",
        ROOT / "scripts/plot_walsh_explainer.py",
        ROOT / "scripts/run_search_transfer.py",
        ROOT / "scripts/repair_graphfla_holdout.py",
    ]
    provenance = {
        "audit_code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code
        },
        "software": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "search": "Replayed saved trials. Paths and hashes are in search_provenance.json.",
        "spectral_seed": 20260826,
        "spectral_draws": 30,
        "coupling": 0.1,
    }
    (args.results / "rebuild_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    _run("typst", "compile", str(args.manuscript / "main.typ"), str(args.manuscript / "main.pdf"))


if __name__ == "__main__":
    main()
