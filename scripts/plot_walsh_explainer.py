"""Draw an exact two-site Fourier example for the manuscript."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mutascape.validation.fourier import binary_walsh_coefficients, walsh_decompose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Rows encode site A and columns encode site B. Zero denotes wild type.
    fitness = np.array([[2.0, 2.0], [4.0, 8.0]])
    coefficients = binary_walsh_coefficients(fitness) / np.sqrt(fitness.size)
    fractions = walsh_decompose(fitness)
    np.testing.assert_allclose(coefficients, [[4.0, -1.0], [-2.0, 1.0]])
    np.testing.assert_allclose(
        binary_walsh_coefficients(coefficients * np.sqrt(fitness.size)), fitness
    )
    np.testing.assert_allclose([fractions[1], fractions[2]], [5 / 6, 1 / 6])
    np.testing.assert_allclose(np.sum(coefficients.ravel()[1:] ** 2), np.var(fitness))

    plt.rcParams.update(
        {
            "font.family": "STIX Two Text",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "text.color": "#161616",
            "axes.labelcolor": "#161616",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), width_ratios=[1, 1.25, 1])
    blue, red, grey = "#167db5", "#d84a45", "#737373"
    pale_blue, pale_red, pale_grey = "#c1deeb", "#f3cdca", "#e6e6e6"
    ax = axes[0]
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        ax.plot(coords[[i, j], 0], coords[[i, j], 1], color=blue, lw=0.8, zorder=0)
    ax.scatter(coords[:, 0], coords[:, 1], s=15, c="black", zorder=2)
    for (x, y), name, value in zip(coords, ["WT  00", "B  01", "A  10", "AB  11"], fitness.ravel()):
        ax.text(
            x,
            y + (0.12 if y else -0.12),
            f"{name}\n$f={value:g}$",
            ha="center",
            va="bottom" if y else "top",
            linespacing=1.15,
        )
    ax.set(xlim=(-0.43, 1.43), ylim=(-0.57, 1.57))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Genotype fitness", loc="left", fontsize=10, pad=12)

    ax = axes[1]
    values = [coefficients[0, 0], coefficients[1, 0], coefficients[0, 1], coefficients[1, 1]]
    ax.bar(
        range(4),
        values,
        width=0.62,
        color=[pale_grey, pale_blue, pale_blue, pale_red],
        edgecolor=[grey, blue, blue, red],
        linewidth=0.8,
    )
    ax.axhline(0, color="black", lw=0.65)
    for x, value in enumerate(values):
        ax.text(
            x,
            value + (0.17 if value >= 0 else -0.17),
            f"${value:+g}$",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )
    ax.set(
        xticks=range(4),
        xticklabels=["Mean\n00", "A\n10", "B\n01", "AB\n11"],
        xlabel="Coefficient mask",
        ylim=(-3, 5),
        yticks=[],
    )
    ax.set_title("(b) Walsh coefficients", loc="left", fontsize=10, pad=12)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)

    ax = axes[2]
    heights = [100 * fractions[1], 100 * fractions[2]]
    ax.bar(
        [1, 2],
        heights,
        color=[pale_blue, pale_red],
        edgecolor=[blue, red],
        linewidth=0.8,
        width=0.6,
    )
    ax.axhline(0, color="black", lw=0.65)
    for x, value in zip([1, 2], heights):
        ax.text(x, value + 3, f"{value:.1f}%", ha="center", fontsize=10)
    ax.set(
        xticks=[1, 2],
        xticklabels=["1\nAdditive", "2\nPair"],
        ylim=(0, 105),
        yticks=[],
        xlabel="Interaction order",
    )
    ax.set_title("(c) Variance by order", loc="left", fontsize=10, pad=12)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)
    fig.tight_layout(w_pad=2.0)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.output / f"walsh_explainer.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (args.output / "walsh_explainer_checks.json").write_text(
        json.dumps(
            {
                "fitness": fitness.tolist(),
                "coefficients": coefficients.tolist(),
                "variance_fractions": fractions,
                "reconstruction_checked": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
