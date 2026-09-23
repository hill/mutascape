"""Audit all outcomes of the frozen distribution-calibration comparison."""

import json

import pandas as pd
from scipy.stats import kendalltau

from mutascape.benchmarks.validity import pairwise_winner_transfer
from scripts.confirm_quantile_pilot import OUT, PROTEINS, digest


def main():
    metrics = pd.concat([pd.read_csv(p) for p in OUT.glob("*/p*/metrics.csv")], ignore_index=True)
    assert len(metrics) == 270
    metrics.to_csv(OUT / "metrics.csv", index=False)
    table = metrics.pivot_table(
        index=["protein", "pilot_size", "model"],
        columns=["classifier", "sequence"],
        values="auc",
        aggfunc="median",
    )
    table.columns = [f"{a}_{'sequence' if b else 'original'}" for a, b in table.columns]
    table = table.join(
        metrics.groupby(["protein", "pilot_size", "model"])[["w1", "spearman"]].median()
    )
    spectra = pd.concat(
        [pd.read_csv(p) for p in OUT.glob("*/p*/structure.csv") if p.stat().st_size > 1],
        ignore_index=True,
    )
    assert len(spectra) == 72
    table = table.join(spectra.groupby(["protein", "pilot_size", "model"]).walsh_l1.median())
    table.to_csv(OUT / "comparison.csv")
    print(table.round(4).to_string())
    spectra.to_csv(OUT / "structure.csv", index=False)
    paths = list(OUT.glob("*/p*/search_b*.csv"))
    if len(paths) == 140:
        scores = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        assert len(scores) == 16800
        scores.to_csv(OUT / "search_scores.csv", index=False)
        rows = []
        for (name, size, budget), group in scores.groupby(["protein", "pilot_size", "budget"]):
            combined = group.rename(columns={"model": "arm"}).copy()
            combined["dataset"] = name
            combined["arm"] = [
                f"{m}_{d}" if m != "measured" else m for m, d in zip(combined.arm, combined.draw)
            ]
            for model in ("reference", "quantile"):
                mean = (
                    group.loc[group.model == model]
                    .groupby(["method", "trial"])
                    .regret.mean()
                    .reset_index()
                )
                mean["dataset"] = name
                mean["arm"] = model
                combined = pd.concat([combined, mean], ignore_index=True)
            other = scores.loc[
                (scores.protein != name)
                & (scores.pilot_size == size)
                & (scores.budget == budget)
                & (scores.model == "measured")
            ]
            fixed = other.groupby(["protein", "method"]).regret.mean().groupby("method").mean()
            fixed_rows = pd.DataFrame(
                {
                    "method": fixed.index,
                    "regret": fixed.values,
                    "trial": 0,
                    "dataset": name,
                    "arm": "fixed_other_proteins",
                }
            )
            combined = pd.concat([combined, fixed_rows], ignore_index=True)
            measured = combined.loc[combined.arm == "measured"].groupby("method").regret.mean()
            transfer = pairwise_winner_transfer(combined, seed=2026100100, n_bootstrap=2000)
            for record in transfer.to_dict("records"):
                predicted = (
                    combined.loc[combined.arm == record["arm"]]
                    .groupby("method")
                    .regret.mean()
                    .reindex(measured.index)
                )
                rows.append(
                    dict(
                        protein=name,
                        pilot_size=size,
                        budget=budget,
                        model=record["arm"],
                        correct_pairs=record["correct_pairs"],
                        scored_pairs=record["scored_pairs"],
                        winner_accuracy=record["winner_accuracy"],
                        kendall_tau=float(kendalltau(measured, predicted).statistic),
                    )
                )
        search = pd.DataFrame(rows)
        search.to_csv(OUT / "search_transfer.csv", index=False)
        search = search.loc[search.model.isin(["reference", "quantile", "fixed_other_proteins"])]
        search.to_csv(OUT / "search_comparison.csv", index=False)
        print(search.to_string(index=False))
        print(
            search.groupby(["pilot_size", "model"])[["winner_accuracy", "kendall_tau"]]
            .mean()
            .to_string()
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, checksum in protocol["code_sha256"].items():
        assert digest(path) == checksum, path
        assert digest(OUT / (path.replace("/", "__") + ".txt")) == checksum
    for source in protocol["sources"]:
        assert digest(source["path"]) == source["sha256"]
        assert digest(OUT / source["protein"] / "genotypes.csv") == source["genotype_sha256"]
    for name in PROTEINS:
        pilot = pd.read_csv(OUT / name / "pilot.csv").row
        test = pd.read_csv(OUT / name / "test.csv").row
        assert len(pilot) == 192 and len(test) == 1000 and set(pilot).isdisjoint(test)
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "classifier_outcomes": len(metrics),
                "spectra": len(spectra),
                "search_groups": len(paths),
                "protocol_sha256": digest(OUT / "protocol.json"),
                "outputs": {p.name: digest(p) for p in OUT.glob("*.csv")},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
