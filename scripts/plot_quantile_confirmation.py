"""Plot the complete 192-combination confirmation comparison."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.confirm_quantile_pilot import OUT, PROTEINS


def main():
    metrics = pd.read_csv(OUT / "metrics.csv")
    spectra = pd.read_csv(OUT / "structure.csv")
    with plt.rc_context(
        {
            "font.family": "STIX Two Text",
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
        specs = [
            ("Original classifier features", "auc", False),
            ("Sequence information added", "auc", True),
            ("Interaction-order distance", "walsh_l1", None),
        ]
        for axis, (title, metric, sequence) in zip(axes, specs):
            data = (
                spectra
                if sequence is None
                else metrics.loc[
                    (metrics.classifier == "histogram") & (metrics.sequence == sequence)
                ]
            )
            for row, protein in enumerate(PROTEINS):
                for index, (model, colour) in enumerate(
                    [("reference", "#167db5"), ("quantile", "#279d29")]
                ):
                    values = data.loc[
                        (data.protein == protein)
                        & (data.pilot_size == 192)
                        & (data.model == model),
                        metric,
                    ].to_numpy()
                    if len(values) == 0:
                        continue
                    centre = np.median(values)
                    y = row + (index - 0.5) * 0.22
                    axis.errorbar(
                        centre,
                        y,
                        xerr=[[centre - values.min()], [values.max() - centre]],
                        fmt="o",
                        color=colour,
                        capsize=3,
                        lw=0.8,
                        markersize=4,
                    )
                    axis.scatter(values, np.full(len(values), y), marker="|", color=colour, s=25)
            axis.set(
                yticks=range(len(PROTEINS)),
                yticklabels=PROTEINS,
                ylim=(len(PROTEINS) - 0.5, -0.5),
                title=title,
                xlabel="Classifier AUC" if sequence is not None else "Walsh / ANOVA L1 distance",
            )
            axis.spines[["top", "right"]].set_visible(False)
            if sequence is not None:
                axis.axvline(0.5, color=".5", ls="--", lw=0.7)
                axis.set_xlim(0.48, 1)
            else:
                axis.set_xlim(0, 0.85)
                axis.text(
                    0.42, 3, "Incomplete library", ha="center", va="center", fontsize=9, color=".4"
                )
        figure.legend(
            [
                plt.Line2D([], [], color="#167db5", marker="o"),
                plt.Line2D([], [], color="#279d29", marker="o"),
            ],
            ["Current pilot", "Distribution-calibrated pilot"],
            loc="lower center",
            ncol=2,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0.08, 1, 1))
        figure.savefig(OUT / "confirmation_comparison.png", dpi=220)
        figure.savefig(OUT / "confirmation_comparison.svg")
        plt.close(figure)


if __name__ == "__main__":
    main()
