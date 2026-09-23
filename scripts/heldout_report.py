"""Summarise the held-out test and say where its primary metric is defined.

Kept apart from ``heldout_test`` so the scoring protocol stays frozen while the
analysis is written.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from scripts.heldout_panel import OUT as PANEL
from scripts.heldout_panel import data
from scripts.heldout_test import OUT as SCORES

CONTROLS = ("shuffled_singles", "structure_only", "fixed_ranking")
DRAWS = 10000
BOOTSTRAP_SEED = 20260923


def plan():
    return json.loads((PANEL / "panel.json").read_text())


def headroom():
    """Fitness a campaign can still gain once it holds wild type and every single.

    Where the best measured variant is wild type or a single mutant, the warm
    start already holds the optimum. Every method then scores zero regret, the
    measured ranking is constant, and Kendall tau is undefined. That depends on
    the measured landscape alone, not on any arm.
    """
    path = SCORES / "headroom.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for name in plan()["panel"]:
        frame, _, _ = data(name)
        values = frame.fitness.to_numpy()
        depth = frame.depth.to_numpy()
        start = values[depth <= 1].max()
        rows.append(
            dict(
                landscape=name,
                held_out=name in plan()["held_out"],
                best_depth=int(depth[values.argmax()]),
                start_regret=float((values.max() - start) / np.ptp(values)),
                above_start=float((values > start).mean()),
            )
        )
    table = pd.DataFrame(rows)
    table.round(6).to_csv(path, index=False)
    return table


def records():
    rows = [json.loads(p.read_text()) for p in sorted(SCORES.glob("*/*.json"))]
    scores = pd.DataFrame([r for r in rows if "arm" in r])
    spaces = {r["landscape"]: r["genotype_space"] for r in plan()["candidates"]}
    scores["genotype_space"] = scores.landscape.map(spaces)
    scores["held_out"] = scores.landscape.isin(plan()["held_out"])
    return scores


def fixed_ranking(names, budget):
    """Each landscape against the mean measured ranking of the others."""
    columns = {}
    for name in names:
        table = pd.read_csv(SCORES / name / "measured.csv", index_col=[0, 1])
        if budget in table.index.get_level_values(0):
            columns[name] = table.loc[budget].regret
    frame = pd.DataFrame(columns)
    return {
        name: float(kendalltau(frame[name], frame.drop(columns=name).mean(axis=1)).statistic)
        for name in frame
    }


def paired_bootstrap(per):
    """Resample genotype spaces, since several landscapes share one."""
    grouped = per.groupby(level="genotype_space").mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, len(grouped), size=(DRAWS, len(grouped)))
    samples = grouped.to_numpy()[picks].mean(axis=1)
    rows = []
    for index, arm in enumerate(grouped.columns):
        row = dict(arm=arm, mean_tau=float(grouped[arm].mean()))
        for control in CONTROLS:
            if control == arm or control not in grouped.columns:
                continue
            other = list(grouped.columns).index(control)
            difference = samples[:, index] - samples[:, other]
            row[f"vs_{control}"] = float(grouped[arm].mean() - grouped[control].mean())
            row[f"vs_{control}_lo"] = float(np.quantile(difference, 0.025))
            row[f"vs_{control}_hi"] = float(np.quantile(difference, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def report():
    space = headroom()
    usable = set(space.loc[space.start_regret > 0, "landscape"])
    scores = records()
    summaries = []
    for budget in plan()["analysis"]["budgets"]:
        column = f"kendall_tau_{budget}"
        subset = scores.loc[
            scores.held_out & scores.landscape.isin(usable) & scores[column].notna()
        ]
        if subset.empty:
            continue
        per = subset.groupby(["genotype_space", "landscape", "arm"])[column].mean().unstack("arm")
        fixed = fixed_ranking(sorted(set(subset.landscape)), budget)
        per["fixed_ranking"] = [fixed[name] for name in per.index.get_level_values("landscape")]
        per = per.dropna(axis=1, how="any")
        per.round(3).to_csv(SCORES / f"per_landscape_{budget}.csv")
        summary = paired_bootstrap(per)
        summary.insert(0, "landscapes", per.shape[0])
        summary.insert(0, "budget", budget)
        summaries.append(summary)
    table = pd.concat(summaries, ignore_index=True)
    table.round(4).to_csv(SCORES / "summary.csv", index=False)
    print(space.round(3).to_string(index=False))
    print()
    print(f"Primary metric defined on {len(usable & set(plan()['held_out']))} held-out landscapes")
    print()
    print(table.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report()


if __name__ == "__main__":
    main()
