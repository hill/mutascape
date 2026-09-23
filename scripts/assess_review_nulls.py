"""Test whether landscape-specific method rankings beat transferable rankings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.validity import fixed_method_ranking, reliable_method_pairs
from scripts.run_search_transfer import _with_ensemble_arms


def main() -> None:
    root = Path("results/review_response_20260923")
    root.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(
        Path("results/preprint_audit_20260923/search_provenance.json").read_text()
    )
    detail, summary, donor_rows, saturation = [], [], [], []
    for budget in (48, 96, 192, 384):
        panels = []
        all_pairs = []
        for source in [p for p in provenance if p["budget"] == budget]:
            df = pd.read_csv(Path(source["directory"]) / "scores.csv", float_precision="round_trip")
            df["collection"] = source["collection"]
            pairs = reliable_method_pairs(df, seed=20260823)
            pairs["collection"] = source["collection"]
            all_pairs.append(pairs)
            panels.append(df)
        scores = pd.concat(panels, ignore_index=True)
        scores["source"] = scores.collection
        combined = _with_ensemble_arms(scores)
        combined["collection"] = combined.dataset.map(dict(zip(scores.dataset, scores.collection)))
        means = combined.groupby(
            ["collection", "dataset", "protein", "arm", "method"], as_index=False
        ).regret.mean()
        measured = means.loc[means.arm == "measured"]
        synthetic = means.loc[means.arm == "dms_pairwise_mean"]
        donors = synthetic.groupby(["protein", "method"]).regret.mean()
        for pairs in all_pairs:
            for dataset, target_pairs in pairs.groupby("dataset"):
                target = measured.loc[measured.dataset == dataset]
                protein = target.protein.iloc[0]
                collection = target.collection.iloc[0]
                null = fixed_method_ranking(measured, excluded_protein=protein)
                predictions = {"Fixed ranking": null}
                for identifier, label in [
                    ("dms_pairwise_mean", "Mutascape"),
                    ("potts_low_mean", "Potts"),
                    ("rmf_high_mean", "RMF"),
                    ("additive", "Additive"),
                ]:
                    predictions[label] = (
                        means.loc[(means.dataset == dataset) & (means.arm == identifier)]
                        .set_index("method")
                        .regret
                    )
                for prediction, values in predictions.items():
                    for pair in target_pairs.itertuples():
                        detail.append(
                            dict(
                                budget=budget,
                                collection=collection,
                                dataset=dataset,
                                protein=protein,
                                prediction=prediction,
                                first=pair.first,
                                second=pair.second,
                                correct=int(
                                    np.sign(values[pair.first] - values[pair.second]) == pair.sign
                                ),
                                involves_random="random" in (pair.first, pair.second),
                            )
                        )
                for donor in donors.index.get_level_values("protein").unique():
                    if donor == protein:
                        continue
                    values = donors.loc[donor]
                    correct = sum(
                        np.sign(values[p.first] - values[p.second]) == p.sign
                        for p in target_pairs.itertuples()
                    )
                    donor_rows.append(
                        dict(
                            budget=budget,
                            collection=collection,
                            dataset=dataset,
                            protein=protein,
                            donor=donor,
                            correct=int(correct),
                            eligible=len(target_pairs),
                        )
                    )
        for (collection, dataset, protein), group in measured.groupby(
            ["collection", "dataset", "protein"]
        ):
            raw = scores.loc[(scores.dataset == dataset) & (scores.arm == "measured")]
            saturation.append(
                dict(
                    budget=budget,
                    collection=collection,
                    dataset=dataset,
                    protein=protein,
                    method_means_distinct=group.regret.nunique(),
                    zero_regret_trial_fraction=float((raw.regret == 0).mean()),
                    method_count=len(group),
                    all_methods_zero=bool((group.regret == 0).all()),
                )
            )
    detail = pd.DataFrame(detail)
    for (budget, collection, prediction), group in detail.groupby(
        ["budget", "collection", "prediction"]
    ):
        for subset, selected in [
            ("all", group),
            ("without_random", group.loc[~group.involves_random]),
            ("random_pairs", group.loc[group.involves_random]),
        ]:
            by_protein = selected.groupby("protein").correct.agg(["sum", "count"])
            summary.append(
                dict(
                    budget=budget,
                    collection=collection,
                    prediction=prediction,
                    subset=subset,
                    accuracy=float((by_protein["sum"] / by_protein["count"]).mean()),
                    eligible=len(selected),
                    proteins=len(by_protein),
                )
            )
    donor_frame = pd.DataFrame(donor_rows)
    for (budget, collection), group in donor_frame.groupby(["budget", "collection"]):
        # One vote per donor protein, then equal weight per target protein.
        target = group.groupby(["protein", "donor"])[["correct", "eligible"]].sum()
        accuracy = (target.correct / target.eligible).groupby("protein").mean().mean()
        summary.append(
            dict(
                budget=budget,
                collection=collection,
                prediction="Other-protein synthetic",
                subset="all",
                accuracy=float(accuracy),
                eligible=int(group.groupby("dataset").eligible.first().sum()),
                proteins=group.protein.nunique(),
            )
        )
    detail.to_csv(root / "null_pair_details.csv", index=False)
    pd.DataFrame(summary).to_csv(root / "null_summary.csv", index=False)
    donor_frame.to_csv(root / "null_donors.csv", index=False)
    pd.DataFrame(saturation).to_csv(root / "saturation.csv", index=False)
    print(pd.DataFrame(summary).query('budget == 48 and subset == "all"').to_string(index=False))


if __name__ == "__main__":
    main()
