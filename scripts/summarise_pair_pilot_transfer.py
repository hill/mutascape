"""Audit and summarise the fixed protein-transfer experiment."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from mutascape.benchmarks.validity import pairwise_winner_transfer
from scripts.test_pair_pilot_transfer import OUT, PROTEINS, digest


def summarise_shape():
    metrics = pd.concat(
        [pd.read_csv(p) for p in sorted(OUT.glob("*/p*/metrics.csv"))], ignore_index=True
    )
    spectra = pd.concat(
        [pd.read_csv(p) for p in sorted(OUT.glob("*/p*/spectra.csv"))], ignore_index=True
    )
    assert len(metrics) == 432 and len(spectra) == 144
    metrics.to_csv(OUT / "metrics.csv", index=False)
    spectra.to_csv(OUT / "spectra.csv", index=False)
    index = ["protein", "pilot_size", "model"]
    summary = metrics.pivot_table(
        index=index, columns=["feature_set", "classifier"], values="auc", aggfunc="median"
    )
    summary.columns = ["_".join(c) + "_auc" for c in summary.columns]
    summary = summary.join(
        metrics.groupby(index)[["scaled_w1", "raw_w1", "spearman", "rmse"]].median()
    )
    summary = summary.join(spectra.groupby(index).walsh_l1.median())
    summary.to_csv(OUT / "comparison.csv")
    differences = []
    for keys, group in metrics.groupby(["protein", "pilot_size", "feature_set", "classifier"]):
        paired = group.pivot(index=["split", "draw"], columns="model", values="auc")
        delta = paired.learned_pairs - paired.pilot_reference
        split_delta = delta.groupby("split").median()
        differences.append(
            dict(
                zip(["protein", "pilot_size", "feature_set", "classifier"], keys),
                median_delta=delta.median(),
                improved=int((delta < 0).sum()),
                total=len(delta),
                split_median_min=split_delta.min(),
                split_median_max=split_delta.max(),
            )
        )
    pd.DataFrame(differences).to_csv(OUT / "paired_classifier_differences.csv", index=False)
    with plt.rc_context({"font.family": "STIX Two Text", "font.size": 10, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True)
        for row, size in enumerate((48, 192)):
            for column, protein in enumerate(PROTEINS):
                axis = axes[row, column]
                for j, (model, colour) in enumerate(
                    (("pilot_reference", "#167db5"), ("learned_pairs", "#299c28"))
                ):
                    for i, feature_set in enumerate(("audit", "with_genotype")):
                        values = (
                            metrics.loc[
                                (metrics.protein == protein)
                                & (metrics.pilot_size == size)
                                & (metrics.model == model)
                                & (metrics.classifier == "histogram")
                                & (metrics.feature_set == feature_set)
                            ]
                            .groupby("split")
                            .auc.median()
                            .to_numpy()
                        )
                        centre = np.median(values)
                        y = i + (j - 0.5) * 0.22
                        axis.errorbar(
                            centre,
                            y,
                            xerr=[[centre - values.min()], [values.max() - centre]],
                            fmt="o",
                            color=colour,
                            capsize=3,
                            lw=0.9,
                            markersize=4,
                        )
                        axis.scatter(values, np.full(3, y), marker="|", s=35, color=colour)
                axis.axvline(0.5, color="0.5", ls="--", lw=0.7)
                axis.set(
                    yticks=[0, 1],
                    yticklabels=["Original features", "With sequence"],
                    xlim=(0.48, 1.01),
                    ylim=(1.5, -0.5),
                    title=f"{protein}, {size} combinations",
                )
                axis.spines[["top", "right"]].set_visible(False)
                if row == 1:
                    axis.set_xlabel("Boosted classifier AUC")
        figure.legend(
            [
                plt.Line2D([], [], color="#167db5", marker="o"),
                plt.Line2D([], [], color="#299c28", marker="o"),
            ],
            ["Current pilot", "Learned pairs"],
            loc="lower center",
            ncol=2,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        figure.savefig(OUT / "classifier_transfer.png", dpi=210)
        figure.savefig(OUT / "classifier_transfer.svg")
        plt.close(figure)
    return summary


def summarise_search():
    paths = sorted(OUT.glob("*/p*/search_b*.csv"))
    assert len(paths) == 252, len(paths)
    scores = pd.concat(
        [pd.read_csv(p).assign(pilot_size=int(p.parent.name.split("_")[0][1:])) for p in paths],
        ignore_index=True,
    )
    assert len(scores) == 30240
    assert np.isfinite(scores[["best", "regret"]].to_numpy()).all()
    assert (
        scores.groupby(["protein", "pilot_size", "split", "budget", "model", "draw"]).size() == 120
    ).all()
    for protein, values in scores.groupby("protein"):
        genotypes = pd.read_csv(OUT / protein / "genotypes.csv")
        available = int((genotypes.depth > 1).sum())
        assert (values.space_size == available - values.pilot_size).all()
    assert not scores.duplicated(
        ["protein", "pilot_size", "split", "budget", "model", "draw", "method", "trial"]
    ).any()
    scores.to_csv(OUT / "search_scores.csv", index=False)
    rows = []
    for (pilot_size, budget, protein, split), group in scores.groupby(
        ["pilot_size", "budget", "protein", "split"]
    ):
        current = group.rename(columns={"model": "arm"}).copy()
        current["dataset"] = f"{protein}_{split}"
        current["arm"] = [
            m if m == "measured" else f"{m}_{d}" for m, d in zip(current.arm, current.draw)
        ]
        for model in ("pilot_reference", "learned_pairs"):
            mean_scores = (
                group.loc[group.model == model]
                .groupby(["method", "trial"])
                .regret.mean()
                .reset_index()
            )
            mean_scores["arm"] = model
            mean_scores["dataset"] = f"{protein}_{split}"
            current = pd.concat([current, mean_scores], ignore_index=True)
        # A comparator estimated from measured searches on the other two proteins.
        other = scores.loc[
            (scores.budget == budget)
            & (scores.pilot_size == pilot_size)
            & (scores.protein != protein)
            & (scores.model == "measured")
        ]
        fixed = other.groupby(["protein", "split", "method"]).regret.mean().groupby("method").mean()
        fixed_rows = pd.DataFrame(
            {
                "method": fixed.index,
                "regret": fixed.values,
                "arm": "fixed_other_proteins",
                "dataset": f"{protein}_{split}",
                "trial": 0,
            }
        )
        current = pd.concat([current, fixed_rows], ignore_index=True)
        transfer = pairwise_winner_transfer(current, seed=2026092700, n_bootstrap=2000)
        real = current.loc[current.arm == "measured"].groupby("method").regret.mean()
        for record in transfer.to_dict("records"):
            predicted = (
                current.loc[current.arm == record["arm"]]
                .groupby("method")
                .regret.mean()
                .reindex(real.index)
            )
            rows.append(
                dict(
                    protein=protein,
                    pilot_size=pilot_size,
                    split=split,
                    budget=budget,
                    model=record["arm"],
                    correct_pairs=record["correct_pairs"],
                    scored_pairs=record["scored_pairs"],
                    winner_accuracy=record["winner_accuracy"],
                    kendall_tau=float(kendalltau(real, predicted).statistic),
                )
            )
    transfer = pd.DataFrame(rows)
    transfer.to_csv(OUT / "search_transfer.csv", index=False)
    summary = (
        transfer.loc[
            transfer.model.isin(["pilot_reference", "learned_pairs", "fixed_other_proteins"])
        ]
        .groupby(["protein", "pilot_size", "budget", "model"])
        .agg(
            correct_pairs=("correct_pairs", "sum"),
            scored_pairs=("scored_pairs", "sum"),
            kendall_tau=("kendall_tau", "median"),
        )
    )
    summary["winner_accuracy"] = summary.correct_pairs / summary.scored_pairs
    summary.to_csv(OUT / "search_comparison.csv")
    return summary


def audit():
    protocol = json.loads((OUT / "protocol.json").read_text())
    counts = []
    for path, checksum in protocol["code_sha256"].items():
        source = OUT / (path.replace("/", "__") + ".txt")
        assert digest(source) == checksum
    amended = json.loads((OUT / "implementation_amendment.json").read_text())
    checksums = dict(protocol["code_sha256"])
    checksums.update(amended["code_sha256"])
    for path, checksum in checksums.items():
        assert digest(path) == checksum
    for name in PROTEINS:
        used = set()
        genotypes = pd.read_csv(OUT / name / "genotypes.csv")
        expected = genotypes.loc[genotypes.depth > 1].sample(n=3 * 2192, random_state=2026092500)
        for split in range(3):
            for role, size in (("pilot", 192), ("test", 2000)):
                frame = pd.read_csv(OUT / name / f"{role}_{split}.csv")
                assert len(frame) == size and frame.row.nunique() == size
                assert used.isdisjoint(frame.row)
                start = split * 2192 + (0 if role == "pilot" else 192)
                np.testing.assert_array_equal(frame.row, expected.index[start : start + size])
                used.update(frame.row)
                counts.append(dict(protein=name, split=split, role=role, rows=size))
        assert all(
            (OUT / name / f"p{s}_s{i}" / "done.json").exists() for s in (48, 192) for i in range(3)
        )
    for dataset in protocol["datasets"]:
        assert digest(dataset["source"]) == dataset["source_sha256"]
        assert digest(OUT / dataset["protein"] / "genotypes.csv") == dataset["genotypes_sha256"]
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "splits": counts,
                "protocol_sha256": digest(OUT / "protocol.json"),
                "amendment_sha256": digest(OUT / "implementation_amendment.json"),
                "outputs": {p.name: digest(p) for p in OUT.glob("*.csv")},
            },
            indent=2,
        )
        + "\n"
    )


def main():
    print(summarise_shape().round(4).to_string())
    if len(list(OUT.glob("*/p*/search_b*.csv"))) == 252:
        print(summarise_search().round(4).to_string())
    audit()


if __name__ == "__main__":
    main()
