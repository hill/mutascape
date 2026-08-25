"""Freeze search-transfer outputs for the API, web interface and manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METHODS = (
    ("dms_pairwise_mean", "Mutascape"),
    ("potts_low_mean", "Potts"),
    ("rmf_high_mean", "RMF"),
    ("posterior_mean", "Prior model"),
    ("additive", "Additive"),
)
CURVE_METHODS = METHODS[:3]


def _parse_run(value: str) -> tuple[int, Path]:
    try:
        budget_text, path_text = value.split("=", maxsplit=1)
        budget = int(budget_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected BUDGET=PATH") from error
    return budget, Path(path_text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", action="append", type=_parse_run, required=True)
    parser.add_argument("--external", action="append", type=_parse_run, required=True)
    parser.add_argument("--primary-budget", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/mutascape/data/validation/benchmark-validity-v1.json"),
    )
    return parser.parse_args()


def _load_runs(values: list[tuple[int, Path]]) -> dict[int, Path]:
    runs = dict(values)
    if len(runs) != len(values):
        raise ValueError("each budget must occur once")
    for budget, root in runs.items():
        summary = json.loads((root / "summary.json").read_text())
        if int(summary["budget"]) != budget:
            raise ValueError(f"{root} reports budget {summary['budget']}, expected {budget}")
    return runs


def _arm_metrics(root: Path, arm: str) -> dict[str, float]:
    transfer = pd.read_csv(root / "transfer.csv")
    pairwise = pd.read_csv(root / "pairwise_transfer.csv")
    adjusted = pd.read_csv(root / "reliability_adjusted_transfer.csv")
    arm_transfer = transfer.loc[transfer["arm"] == arm, "kendall_tau"]
    arm_pairs = pairwise.loc[pairwise["arm"] == arm, ["correct_pairs", "scored_pairs"]].sum()
    scored_pairs = int(arm_pairs["scored_pairs"])
    if arm_transfer.empty or scored_pairs == 0:
        raise ValueError(f"{root} has no complete metrics for {arm}")
    return {
        "kendallTau": round(float(arm_transfer.median()), 6),
        "pairwiseAccuracy": round(float(arm_pairs["correct_pairs"] / scored_pairs), 6),
        "reliabilityPassRate": round(
            float(adjusted.loc[adjusted["arm"] == arm, "meets_reliability_floor"].mean()),
            6,
        ),
    }


def _metadata(root: Path) -> dict[str, object]:
    summary = json.loads((root / "summary.json").read_text())
    return {
        "nLandscapes": int(summary["n_landscapes"]),
        "nMethods": int(summary["n_methods"]),
        "budget": int(summary["budget"]),
        "trials": int(summary["seeds"]),
        "draws": int(summary.get("generator_draws", summary["posterior_draws"])),
    }


def main() -> None:
    args = _parse_args()
    development = _load_runs(args.development)
    external = _load_runs(args.external)
    if set(development) != set(external):
        raise ValueError("development and external runs must use the same budgets")
    if args.primary_budget not in development:
        raise ValueError("primary budget is absent")

    development_root = development[args.primary_budget]
    external_root = external[args.primary_budget]
    reliability = pd.read_csv(development_root / "measured_reliability.csv")
    development_metadata = _metadata(development_root)
    external_metadata = _metadata(external_root)
    external_primary = _arm_metrics(external_root, "dms_pairwise_mean")
    budgets = sorted(development)
    output = {
        "id": "benchmark-validity-v1",
        "panel": "SSMuLA development panel",
        **development_metadata,
        "measuredReliabilityFloor": round(float(reliability["split_half_lower"].median()), 6),
        "external": {
            "panel": "GraphFLA protein panel",
            **external_metadata,
            **external_primary,
        },
        "methods": [
            {"id": arm, "label": label, **_arm_metrics(development_root, arm)}
            for arm, label in METHODS
        ],
        "budgetCurve": {
            "budgets": budgets,
            "series": [
                {
                    "id": arm,
                    "label": label,
                    "developmentKendall": [
                        _arm_metrics(development[budget], arm)["kendallTau"] for budget in budgets
                    ],
                    "externalKendall": [
                        _arm_metrics(external[budget], arm)["kendallTau"] for budget in budgets
                    ],
                    "developmentWinners": [
                        _arm_metrics(development[budget], arm)["pairwiseAccuracy"]
                        for budget in budgets
                    ],
                    "externalWinners": [
                        _arm_metrics(external[budget], arm)["pairwiseAccuracy"]
                        for budget in budgets
                    ],
                }
                for arm, label in CURVE_METHODS
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
