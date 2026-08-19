"""Load and compare completion-benchmark runs.

Notebooks present results, scripts produce them. Everything here reads the CSV
and JSON a benchmark run already wrote, so a notebook never recomputes a
number the paper cites.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Metrics where a smaller value is the better one.
LOWER_IS_BETTER = frozenset(
    {"mae", "rmse", "wasserstein", "crps", "c2st_auc", "mmd_squared", "interval_width"}
)

HEADLINE = ("spearman", "ndcg", "mae", "rmse", "wasserstein", "crps", "c2st_auc", "mmd_squared")


def load_run(path: Path | str) -> dict[str, object]:
    """Read one benchmark output directory."""
    path = Path(path)
    run: dict[str, object] = {
        "name": path.name,
        "metrics": pd.read_csv(path / "metrics.csv"),
        "summary": json.loads((path / "summary.json").read_text()),
    }
    calibration = path / "calibration.csv"
    if calibration.exists():
        run["calibration"] = pd.read_csv(calibration)
    return run


def load_runs(paths: dict[str, Path | str]) -> dict[str, dict[str, object]]:
    """Read several runs, keyed by a label you choose."""
    return {label: load_run(path) for label, path in paths.items()}


def median_table(
    runs: dict[str, dict[str, object]],
    *,
    model: str = "posterior",
    metrics: tuple[str, ...] = HEADLINE,
) -> pd.DataFrame:
    """Median of each metric across landscapes, one column per run."""
    columns = {}
    for label, run in runs.items():
        frame = run["metrics"]
        selected = frame.loc[frame["model"] == model]
        columns[label] = selected.groupby("metric")["value"].median()
    table = pd.DataFrame(columns)
    return table.reindex([m for m in metrics if m in table.index])


def paired_comparison(
    runs: dict[str, dict[str, object]],
    baseline: str,
    candidate: str,
    *,
    model: str = "posterior",
    metrics: tuple[str, ...] = HEADLINE,
) -> pd.DataFrame:
    """Compare two runs landscape by landscape.

    Reports the median change and how many landscapes moved the right way.
    Pairing matters because assay scales differ, so an unpaired mean across
    landscapes is dominated by whichever assay has the widest units.
    """
    left = runs[baseline]["metrics"]
    right = runs[candidate]["metrics"]
    rows = []
    for metric in metrics:
        a = left.loc[(left["model"] == model) & (left["metric"] == metric)]
        b = right.loc[(right["model"] == model) & (right["metric"] == metric)]
        merged = a.merge(b, on="dataset", suffixes=("_base", "_cand"))
        if merged.empty:
            continue
        delta = merged["value_cand"] - merged["value_base"]
        improved = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
        rows.append(
            {
                "metric": metric,
                baseline: float(np.median(merged["value_base"])),
                candidate: float(np.median(merged["value_cand"])),
                "median_delta": float(np.median(delta)),
                "improved": int(improved.sum()),
                "n": int(len(merged)),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def versus_baseline(
    run: dict[str, object],
    *,
    model: str = "posterior",
    reference: str = "ridge_additive",
    metrics: tuple[str, ...] = HEADLINE,
) -> pd.DataFrame:
    """Compare a model against a baseline within one run, landscape by landscape."""
    frame = run["metrics"]
    rows = []
    for metric in metrics:
        a = frame.loc[(frame["model"] == reference) & (frame["metric"] == metric)]
        b = frame.loc[(frame["model"] == model) & (frame["metric"] == metric)]
        merged = a.merge(b, on="dataset", suffixes=("_ref", "_model"))
        if merged.empty:
            continue
        wins = (
            merged["value_model"] < merged["value_ref"]
            if metric in LOWER_IS_BETTER
            else merged["value_model"] > merged["value_ref"]
        )
        rows.append(
            {
                "metric": metric,
                reference: float(np.median(merged["value_ref"])),
                model: float(np.median(merged["value_model"])),
                "wins": int(wins.sum()),
                "n": int(len(merged)),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def coverage_table(runs: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Median empirical coverage and interval width per nominal level."""
    frames = []
    for label, run in runs.items():
        calibration = run.get("calibration")
        if calibration is None:
            continue
        grouped = calibration.groupby("nominal")[["coverage", "mean_width"]].median()
        grouped.columns = pd.MultiIndex.from_product([[label], grouped.columns])
        frames.append(grouped)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()
