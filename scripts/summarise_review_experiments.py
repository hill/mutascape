"""Summarise matched readout trials and the frozen confirmation study."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.validity import (
    fixed_method_ranking,
    pairwise_winner_transfer,
    reliable_method_pairs,
)
from scripts.run_search_transfer import _transfer, _with_ensemble_arms

ROOT = Path("results/review_response_20260923")


def summaries(scores, seed):
    combined = _with_ensemble_arms(scores)
    proteins = scores[["dataset", "protein"]].drop_duplicates()
    pairs = pairwise_winner_transfer(combined, seed=seed).merge(proteins, on="dataset")
    transfer = _transfer(scores).merge(proteins, on="dataset")
    rows = []
    for model in (
        "dms_pairwise_mean",
        "dms_no_readout_mean",
        "potts_low_mean",
        "rmf_high_mean",
        "additive",
    ):
        selected = pairs.loc[pairs.arm == model]
        totals = selected.groupby("protein")[["correct_pairs", "scored_pairs"]].sum()
        ranks = transfer.loc[transfer.arm == model].groupby("protein").kendall_tau.median()
        rows.append(
            dict(
                model=model,
                kendall=float(ranks.median()),
                accuracy=float((totals.correct_pairs / totals.scored_pairs).mean()),
                eligible=int(totals.scored_pairs.sum()),
            )
        )
    return rows


def main():
    provenance = json.loads(
        Path("results/preprint_audit_20260923/search_provenance.json").read_text()
    )
    ablation, confirmation = [], []
    for budget in (48, 96, 192, 384):
        development = []
        for panel in [p for p in provenance if p["budget"] == budget]:
            scores = pd.read_csv(
                Path(panel["directory"]) / "scores.csv", float_precision="round_trip"
            )
            development.append(scores)
            paths = [
                ROOT / "readout_ablation" / f"b{budget}_{name}.csv"
                for name in scores.dataset.unique()
            ]
            missing = [str(p) for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Incomplete readout experiment: {missing}")
            if all(p.exists() for p in paths):
                controls = [pd.read_csv(p, float_precision="round_trip") for p in paths]
                merged = pd.concat([scores, *controls], ignore_index=True)
                for row in summaries(merged, 20260823):
                    ablation.append(dict(budget=budget, collection=panel["collection"], **row))
        path = ROOT / f"confirmation_b{budget}/scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"Incomplete confirmation experiment: {path}")
        scores = pd.read_csv(path, float_precision="round_trip")
        for row in summaries(scores, 20260923):
            confirmation.append(dict(budget=budget, **row))
        fixed = fixed_method_ranking(pd.concat(development), excluded_protein="ParB-Noc")
        pairs = reliable_method_pairs(scores, seed=20260923)
        correct = sum(
            np.sign(fixed[p.first] - fixed[p.second]) == p.sign for p in pairs.itertuples()
        )
        measured = scores.loc[scores.arm == "measured"].groupby("method").regret.mean()
        from scipy.stats import kendalltau

        confirmation.append(
            dict(
                budget=budget,
                model="fixed_ranking",
                kendall=float(kendalltau(measured, fixed.reindex(measured.index)).statistic),
                accuracy=correct / len(pairs),
                eligible=len(pairs),
            )
        )
    pd.DataFrame(ablation).to_csv(ROOT / "readout_summary.csv", index=False)
    pd.DataFrame(confirmation).to_csv(ROOT / "confirmation_summary.csv", index=False)
    print(pd.DataFrame(confirmation).to_string(index=False))


if __name__ == "__main__":
    main()
