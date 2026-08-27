"""Freeze search-transfer outputs for the API, web interface and manuscript."""

import argparse
import json
from pathlib import Path

import pandas as pd

METHODS = (
    ("dms_pairwise_mean", "Mutascape"),
    ("potts_low_mean", "Potts"),
    ("rmf_high_mean", "RMF"),
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
    parser.add_argument(
        "--ssmula",
        "--development",
        dest="ssmula",
        action="append",
        type=_parse_run,
        required=True,
    )
    parser.add_argument("--combingym", action="append", type=_parse_run, required=True)
    parser.add_argument(
        "--graphfla",
        "--external",
        dest="graphfla",
        action="append",
        type=_parse_run,
        required=True,
    )
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
    proteins = pd.read_csv(root / "scores.csv")[["dataset", "protein"]].drop_duplicates()
    transfer = pd.read_csv(root / "transfer.csv")
    pairwise = pd.read_csv(root / "pairwise_transfer.csv")
    adjusted = pd.read_csv(root / "reliability_adjusted_transfer.csv")
    arm_transfer = transfer.loc[transfer["arm"] == arm].merge(proteins, on="dataset")
    arm_pairs = pairwise.loc[pairwise["arm"] == arm].merge(proteins, on="dataset")
    if arm_transfer.empty or arm_pairs.empty:
        raise ValueError(f"{root} has no complete metrics for {arm}")
    protein_transfer = arm_transfer.groupby("protein")["kendall_tau"].median()
    protein_pairs = arm_pairs.groupby("protein")[["correct_pairs", "scored_pairs"]].sum()
    protein_pairs = protein_pairs.loc[protein_pairs["scored_pairs"] > 0]
    if protein_pairs.empty:
        raise ValueError(f"{root} has no reliable method pairs for {arm}")
    protein_adjusted = (
        adjusted.loc[adjusted["arm"] == arm]
        .merge(proteins, on="dataset")
        .groupby("protein")["meets_reliability_floor"]
        .mean()
    )
    return {
        "kendallTau": round(float(protein_transfer.median()), 6),
        "pairwiseAccuracy": round(
            float((protein_pairs["correct_pairs"] / protein_pairs["scored_pairs"]).mean()), 6
        ),
        "reliabilityPassRate": round(float(protein_adjusted.mean()), 6),
    }


def _reliability_floor(root: Path) -> float:
    proteins = pd.read_csv(root / "scores.csv")[["dataset", "protein"]].drop_duplicates()
    reliability = pd.read_csv(root / "measured_reliability.csv").merge(proteins, on="dataset")
    protein_floors = reliability.groupby("protein")["split_half_lower"].median()
    if protein_floors.empty:
        raise ValueError(f"{root} has no measured reliability estimates")
    return round(float(protein_floors.median()), 6)


def _metadata(root: Path) -> dict[str, object]:
    summary = json.loads((root / "summary.json").read_text())
    n_protein_groups = pd.read_csv(root / "scores.csv")["protein"].nunique()
    return {
        "nLandscapes": int(summary["n_landscapes"]),
        "nProteinGroups": int(n_protein_groups),
        "nMethods": int(summary["n_methods"]),
        "budget": int(summary["budget"]),
        "trials": int(summary["seeds"]),
        "draws": int(summary.get("generator_draws", summary["posterior_draws"])),
    }


def main() -> None:
    args = _parse_args()
    ssmula = _load_runs(args.ssmula)
    combingym = _load_runs(args.combingym)
    graphfla = _load_runs(args.graphfla)
    if not (set(ssmula) == set(combingym) == set(graphfla)):
        raise ValueError("SSMuLA, CombinGym, and GraphFLA runs must use the same budgets")
    if args.primary_budget not in ssmula:
        raise ValueError("primary budget is absent")

    ssmula_root = ssmula[args.primary_budget]
    combingym_root = combingym[args.primary_budget]
    graphfla_root = graphfla[args.primary_budget]
    ssmula_metadata = _metadata(ssmula_root)
    combingym_metadata = _metadata(combingym_root)
    graphfla_metadata = _metadata(graphfla_root)
    combingym_primary = _arm_metrics(combingym_root, "dms_pairwise_mean")
    graphfla_primary = _arm_metrics(graphfla_root, "dms_pairwise_mean")
    budgets = sorted(ssmula)
    output = {
        "id": "benchmark-validity-v1",
        "aggregation": "protein-balanced",
        **ssmula_metadata,
        "measuredReliabilityFloor": _reliability_floor(ssmula_root),
        "combingym": {
            **combingym_metadata,
            **combingym_primary,
        },
        "graphfla": {
            **graphfla_metadata,
            **graphfla_primary,
        },
        "methods": [
            {"id": arm, "label": label, **_arm_metrics(ssmula_root, arm)} for arm, label in METHODS
        ],
        "budgetCurve": {
            "budgets": budgets,
            "series": [
                {
                    "id": arm,
                    "label": label,
                    "ssmulaKendall": [
                        _arm_metrics(ssmula[budget], arm)["kendallTau"] for budget in budgets
                    ],
                    "combingymKendall": [
                        _arm_metrics(combingym[budget], arm)["kendallTau"] for budget in budgets
                    ],
                    "graphflaKendall": [
                        _arm_metrics(graphfla[budget], arm)["kendallTau"] for budget in budgets
                    ],
                    "ssmulaWinners": [
                        _arm_metrics(ssmula[budget], arm)["pairwiseAccuracy"] for budget in budgets
                    ],
                    "combingymWinners": [
                        _arm_metrics(combingym[budget], arm)["pairwiseAccuracy"]
                        for budget in budgets
                    ],
                    "graphflaWinners": [
                        _arm_metrics(graphfla[budget], arm)["pairwiseAccuracy"]
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
