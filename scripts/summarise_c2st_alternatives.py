"""Summarise all alternative-model outcomes and audit recorded split identities."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("results/c2st_alternatives_20260923")


def main():
    metrics = pd.read_csv(ROOT / "metrics.csv")
    spectra = pd.read_csv(ROOT / "spectra.csv")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    source = ROOT / "experiment_source.py.txt"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["script_sha256"]
    assert len(metrics) == 135 and len(spectra) == 45
    used = set()
    split_rows = []
    for split in range(3):
        for role, expected in (("pilot", 48), ("test", 2000)):
            data = pd.read_csv(ROOT / f"{role}_{split}.csv")
            assert len(data) == expected and data.variant.nunique() == expected
            assert used.isdisjoint(data.variant)
            used.update(data.variant)
            assert data.mutations.map(ast.literal_eval).map(len).min() >= 2
            split_rows.append(dict(split=split, role=role, count=len(data)))
    table = metrics.pivot_table(
        index="model", columns=["feature_set", "classifier"], values="auc", aggfunc="median"
    )
    table.columns = ["_".join(c) + "_auc" for c in table.columns]
    table = table.join(metrics.groupby("model")[["raw_w1", "raw_w1_real_sd", "spearman"]].median())
    table = table.join(spectra.groupby("model")[["walsh_l1", "cos_ge2"]].median())
    table.to_csv(ROOT / "comparison.csv")
    differences = []
    for (feature_set, classifier), group in metrics.groupby(["feature_set", "classifier"]):
        paired = group.pivot(index=["split", "draw"], columns="model", values="auc")
        for model in ("loo_noise", "raw_quadratic", "learned_pairs"):
            delta = paired[model] - paired.pilot_reference
            differences.append(
                dict(
                    model=model,
                    feature_set=feature_set,
                    classifier=classifier,
                    median_delta=float(delta.median()),
                    improved=int((delta < 0).sum()),
                    comparisons=len(delta),
                )
            )
    pd.DataFrame(differences).to_csv(ROOT / "paired_differences.csv", index=False)
    metadata = {
        "validation": (
            "All 135 classifier outcomes, 45 spectra, and six disjoint genotype sets checked"
        ),
        "script_used": str(source),
        "input_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("results/creilov_c2st/metadata.json"),
                Path("data/creilov/creilov.csv"),
                Path("data/creilov/creilov_wt.fasta"),
            ]
        },
        "results_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.csv")
        },
        "split_counts": split_rows,
    }
    (ROOT / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    names = ["singles_only", "pilot_reference", "loo_noise", "raw_quadratic", "learned_pairs"]
    labels = ["Singles only", "Current pilot", "LOO noise", "Curved fitness", "Learned pairs"]
    with plt.rc_context(
        {
            "font.family": "STIX Two Text",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
        for axis, feature_set, title in zip(
            axes,
            ("audit", "with_genotype"),
            ("Existing classifier features", "Sequence information added"),
        ):
            for i, name in enumerate(names):
                selected = metrics.loc[
                    (metrics.model == name)
                    & (metrics.classifier == "histogram")
                    & (metrics.feature_set == feature_set)
                ]
                values = selected.auc.to_numpy()
                median = float(np.median(values))
                axis.errorbar(
                    median,
                    i,
                    xerr=[[median - values.min()], [values.max() - median]],
                    fmt="o",
                    color="#167db5",
                    ecolor="#737373",
                    capsize=3,
                    markersize=4,
                    linewidth=0.8,
                )
                axis.scatter(values, np.full(len(values), i), marker="|", color="#167db5", s=25)
            axis.axvline(0.5, color="#737373", ls="--", lw=0.7)
            axis.set(
                yticks=range(5),
                yticklabels=labels,
                xlim=(0.48, 1.015),
                xlabel="Classifier AUC",
                title=title,
            )
            axis.invert_yaxis()
            axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(ROOT / "classifier_comparison.png", dpi=220)
        figure.savefig(ROOT / "classifier_comparison.svg")
        plt.close(figure)
    print(table.to_string())


if __name__ == "__main__":
    main()
