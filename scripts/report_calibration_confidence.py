"""Audit and report the fixed calibration follow-up after every campaign finishes."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import calibration_confidence as run
from scripts import manuscript_v2 as experiment
from scripts.analyse_generator_agreement import analyse

DESTINATION = Path("docs/experiments/calibration-confidence")


def audit() -> dict:
    expected_methods = set(experiment.METHODS)
    total = 0
    warnings = 0
    fallback = 0
    for name in experiment.EVALUATION:
        for c in range(run.SAMPLES):
            for model in ("measured", "additive", "calibrated", "sigmoid"):
                landscapes = experiment.LANDSCAPES if model in ("calibrated", "sigmoid") else 1
                repeats = (
                    experiment.MEASURED_REPEATS
                    if model == "measured"
                    else experiment.SEARCH_REPEATS
                )
                for landscape in range(landscapes):
                    path = run.OUTPUT / name / "search" / f"{model}_{c}_{landscape}.csv"
                    frame = pd.read_csv(path)
                    assert len(frame) == len(experiment.BUDGETS) * repeats * len(
                        expected_methods
                    ), path
                    assert set(frame.method) == expected_methods, path
                    assert set(frame.budget) == set(experiment.BUDGETS), path
                    assert set(frame.trial) == set(range(repeats)), path
                    assert not frame.duplicated(["budget", "trial", "method"]).any(), path
                    assert frame[["best", "start", "regret"]].notna().all().all(), path
                    assert frame.regret.between(-1e-10, 1 + 1e-10).all(), path
                    assert (frame.best >= frame.start - 1e-10).all(), path
                    total += len(frame)
                    warnings += int(frame.warnings.sum())
                    fallback += int((frame.fallback > 0).sum())
    assert total == 460800, total
    return {"campaigns": total, "warnings": warnings, "campaigns_with_fallback": fallback}


def main() -> None:
    counts = audit()
    run.configure()
    summary = experiment.search_summary()
    experiment.aggregate_search_summary(summary).to_csv(run.OUTPUT / "search_table.csv")
    overall, individual, pairs = analyse(summary, resamples=100000, seed=20261119)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for label, table in [
        ("overall", overall),
        ("per_protein", individual),
        ("paired_samples", pairs),
    ]:
        table.to_csv(DESTINATION / f"{label}.csv", index=False)
    summary.to_csv(DESTINATION / "search_summary.csv", index=False)
    matched = pairs.groupby(["protein", "metric", "budget"]).agg(
        matched_samples=("difference", "count")
    )
    matched.to_csv(DESTINATION / "matched_samples.csv")
    counts["samples_without_eligible_pairs"] = int(
        summary.loc[summary.arm == "calibrated", "eligible_pairs"].eq(0).sum()
    )
    (DESTINATION / "audit.json").write_text(json.dumps(counts, indent=2) + "\n")
    for name in ("manifest.json", "provenance.json"):
        (DESTINATION / name).write_bytes((run.OUTPUT / name).read_bytes())
    order = [*experiment.EVALUATION, "Overall"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, budget in zip(axes, experiment.BUDGETS):
        rows = (
            pd.concat(
                [
                    individual.loc[
                        (individual.metric == "agreement") & (individual.budget == budget)
                    ],
                    overall.loc[
                        (overall.metric == "agreement") & (overall.budget == budget)
                    ].assign(protein="Overall"),
                ]
            )
            .set_index("protein")
            .loc[order]
        )
        mean = rows.difference.to_numpy() * 100
        limits = np.array(
            [mean - rows.ci_low.to_numpy() * 100, rows.ci_high.to_numpy() * 100 - mean]
        )
        ax.errorbar(mean, np.arange(len(order)), xerr=limits, fmt="o", color="#197078", capsize=4)
        ax.axvline(0, color="0.5", linestyle=":", linewidth=1)
        ax.set_yticks(np.arange(len(order)), order)
        ax.set_title(f"{budget} new measurements")
        ax.set_xlabel("Change in pair agreement (percentage points)")
        ax.grid(axis="x", alpha=0.15)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Combination-calibrated minus single-site-only Mutascape\n"
        "40 fresh samples per protein, 95% paired bootstrap intervals",
        fontsize=11,
    )
    fig.savefig(DESTINATION / "agreement.png", dpi=180)
    fig.savefig(DESTINATION / "agreement.pdf")
    plt.close(fig)
    print(json.dumps(counts))
    print(overall.to_string(index=False))
    print(individual.to_string(index=False))


if __name__ == "__main__":
    main()
