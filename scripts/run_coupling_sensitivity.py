"""Score the frozen search-transfer protocol across pair-coupling strengths.

The coupling is the generator's only free parameter. Reporting the headline
metrics against a range of values shows whether a benchmark conclusion depends
on it. Every run uses the confirmation protocol rather than a reduced one,
because reliable-winner accuracy needs at least four measured trials per method
pair and returns an undefined value below that.

This is a sensitivity report on development data. The panels here have already
guided model selection, so a better score at another coupling is not evidence
for changing the default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RUNNER = Path(__file__).with_name("run_search_transfer.py")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--graphfla-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[48, 96])
    parser.add_argument(
        "--couplings",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.4],
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["dms_pairwise_mean", "potts_low_mean", "rmf_high_mean", "additive"],
        help="Arms carried into the aggregate table",
    )
    return parser.parse_args()


def _run(args: argparse.Namespace, budget: int, coupling: float) -> Path:
    output = args.output_root / f"b{budget}_coupling_{coupling:g}"
    if (output / "summary.json").exists():
        print(f"reusing {output}", flush=True)
        return output
    command = [
        sys.executable,
        str(RUNNER),
        "--ssmula-root",
        str(args.ssmula_root),
        "--output",
        str(output),
        "--budget",
        str(budget),
        "--seeds",
        str(args.seeds),
        "--draws",
        str(args.draws),
        "--generators",
        "primary",
        "--dms-pairwise-coupling",
        str(coupling),
        "--jobs",
        str(args.jobs),
    ]
    if args.combingym_root is not None:
        command += ["--combingym-root", str(args.combingym_root)]
    if args.graphfla_root is not None:
        command += ["--graphfla-root", str(args.graphfla_root)]
    print(f"running budget {budget}, coupling {coupling:g}", flush=True)
    subprocess.run(command, check=True)
    return output


def _rows(output: Path, budget: int, coupling: float, arms: list[str]) -> list[dict[str, object]]:
    summary = json.loads((output / "summary.json").read_text())
    transfer = summary["median_transfer"]
    winners = summary["pairwise_winner_accuracy"]
    passes = summary["reliability_pass_rate"]
    rows = []
    for arm in arms:
        if arm not in transfer and arm not in winners:
            continue
        rows.append(
            {
                "budget": budget,
                "coupling": coupling,
                "arm": arm,
                "kendall_tau": transfer.get(arm, {}).get("kendall_tau"),
                "winner_accuracy": winners.get(arm),
                "pass_rate": passes.get(arm),
                "n_landscapes": summary["n_landscapes"],
                "seeds": summary["seeds"],
                "draws": summary["generator_draws"],
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for budget in args.budgets:
        for coupling in args.couplings:
            output = _run(args, budget, coupling)
            rows.extend(_rows(output, budget, coupling, list(args.arms)))

    table = pd.DataFrame(rows)
    destination = args.output_root / "coupling_sensitivity.csv"
    table.to_csv(destination, index=False)
    print(f"\nwrote {destination}\n")

    mutascape = table.loc[table["arm"] == "dms_pairwise_mean"]
    for budget, group in mutascape.groupby("budget"):
        print(f"budget {budget}")
        print(
            group[["coupling", "kendall_tau", "winner_accuracy", "pass_rate"]].to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            )
        )
        print()


if __name__ == "__main__":
    main()
