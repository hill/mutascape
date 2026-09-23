"""Render four scripted alternatives for the lower row of the landscape figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "landscape_bottom_row_alternatives"
BLUE = "#167db5"
LIGHT_BLUE = "#acd4e6"
RED = "#d84a45"
GREY = "#9b9b9b"
INK = "#161616"


def style() -> LinearSegmentedColormap:
    """Set the shared Garnett-inspired style and return the fitness map."""
    plt.rcParams.update(
        {
            "font.family": "STIX Two Text",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    return LinearSegmentedColormap.from_list(
        "fitness_blue", ["#f4f8fb", "#d1e6f0", "#80b8d5", "#247fae"]
    )


def finish(fig: plt.Figure, name: str) -> None:
    """Save one option as PNG and SVG."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(
            OUTPUT / f"{name}.{extension}",
            dpi=260,
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)


def walk_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a two-peak landscape with one local and one global optimum."""
    local = 0.50 * np.exp(-((x + 0.38) ** 2 + (y + 0.28) ** 2) / 0.12)
    global_peak = 0.88 * np.exp(-((x - 0.48) ** 2 + (y - 0.36) ** 2) / 0.10)
    slope = 0.08 * (x + y + 2) / 4
    return 0.06 + slope + local + global_peak


def plot_surface(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, colour) -> None:
    """Draw a surface with the common view."""
    ax.computed_zorder = False
    ax.set_proj_type("ortho")
    ax.view_init(elev=30, azim=-58)
    ax.set_box_aspect((1, 1, 0.67), zoom=1.12)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1.05))
    ax.set_axis_off()
    ax.plot_surface(
        x,
        y,
        z,
        cmap=colour,
        norm=Normalize(0, 1),
        rcount=41,
        ccount=41,
        linewidth=0.22,
        edgecolor="#8ebdd5",
        antialiased=True,
        shade=False,
        zorder=1,
    )


def option_walks(colour) -> None:
    """Show a greedy trap and a walk that crosses a low-fitness region."""
    axis = np.linspace(-1, 1, 161)
    x, y = np.meshgrid(axis, axis)
    z = walk_surface(x, y)
    paths = [
        np.array([[-0.88, -0.72], [-0.72, -0.58], [-0.57, -0.45], [-0.46, -0.35], [-0.38, -0.28]]),
        np.array(
            [
                [-0.88, -0.72],
                [-0.67, -0.55],
                [-0.45, -0.42],
                [-0.18, -0.18],
                [0.08, -0.02],
                [0.29, 0.17],
                [0.48, 0.36],
            ]
        ),
    ]
    titles = ["Greedy walk: trapped", "Exploratory walk: global peak"]
    notes = [
        "Every accepted mutation climbs locally",
        "A neutral or deleterious step opens a route",
    ]
    fig = plt.figure(figsize=(8.3, 2.65))
    for i, path in enumerate(paths):
        ax = fig.add_axes([0.02 + 0.48 * i, 0.13, 0.46, 0.75], projection="3d")
        plot_surface(ax, x, y, z, colour)
        values = walk_surface(path[:, 0], path[:, 1]) + 0.04
        ax.plot(
            path[:, 0],
            path[:, 1],
            values,
            color=RED,
            lw=2.0,
            marker="o",
            ms=3.5,
            zorder=4,
        )
        ax.text(*path[0], values[0] + 0.035, "WT", fontsize=9, ha="center")
        endpoint = "local" if i == 0 else "global"
        ax.text(*path[-1], values[-1] + 0.045, endpoint, fontsize=9, ha="center")
        fig.text(0.25 + 0.48 * i, 0.93, titles[i], ha="center", fontsize=13)
        fig.text(0.25 + 0.48 * i, 0.045, notes[i], ha="center", fontsize=10)
    finish(fig, "option_1_adaptive_walks")


def genotype_positions() -> dict[frozenset[str], tuple[float, float]]:
    """Return a mutation-count layout for the three-locus genotype cube."""
    return {
        frozenset(): (0.5, 0.08),
        frozenset("A"): (0.13, 0.36),
        frozenset("B"): (0.50, 0.36),
        frozenset("C"): (0.87, 0.36),
        frozenset("AB"): (0.13, 0.66),
        frozenset("AC"): (0.50, 0.66),
        frozenset("BC"): (0.87, 0.66),
        frozenset("ABC"): (0.50, 0.94),
    }


def option_combinatorial_paths(colour) -> None:
    """Compare mutation-order access on additive and epistatic genotype graphs."""
    positions = genotype_positions()
    additive = {node: 0.18 + 0.24 * len(node) for node in positions}
    epistatic = {
        frozenset(): 0.20,
        frozenset("A"): 0.40,
        frozenset("B"): 0.15,
        frozenset("C"): 0.12,
        frozenset("AB"): 0.10,
        frozenset("AC"): 0.65,
        frozenset("BC"): 0.08,
        frozenset("ABC"): 0.95,
    }
    titles = ["Additive: every order climbs", "Epistatic: order controls access"]
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.65))
    for ax, fitness, title in zip(axes, [additive, epistatic], titles, strict=True):
        edges = []
        for source in positions:
            for mutation in "ABC":
                target = frozenset((*source, mutation))
                if len(target) == len(source) + 1 and target in positions:
                    edges.append((source, target))
        for source, target in edges:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            accessible = fitness[target] > fitness[source]
            colour_edge = BLUE if accessible else "#c9c9c9"
            ax.annotate(
                "",
                xy=(x1, y1 - 0.025),
                xytext=(x0, y0 + 0.025),
                arrowprops={"arrowstyle": "-", "color": colour_edge, "lw": 1.4},
                zorder=1,
            )
        for node, (xp, yp) in positions.items():
            label = "WT" if not node else "".join(sorted(node))
            ax.scatter(
                xp,
                yp,
                s=310,
                c=[fitness[node]],
                cmap=colour,
                norm=Normalize(0, 1),
                edgecolors=INK,
                linewidths=0.6,
                zorder=3,
            )
            ax.text(xp, yp, label, ha="center", va="center", fontsize=9, zorder=4)
        ax.set_title(title, fontsize=13, pad=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_axis_off()
    fig.text(
        0.5, 0.005, "Nodes are complete genotypes; edges add one mutation", ha="center", fontsize=10
    )
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.86, wspace=0.17)
    finish(fig, "option_2_combinatorial_paths")


def option_pair_epistasis() -> None:
    """Compare positive and negative pairwise epistasis against additivity."""
    labels = ["WT", "A", "B", "AB"]
    single = np.array([0.20, 0.45, 0.50])
    expected = single[1] + single[2] - single[0]
    doubles = [0.94, 0.48]
    titles = ["Positive epistasis", "Negative epistasis"]
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.65), sharey=True)
    for ax, double, title in zip(axes, doubles, titles, strict=True):
        observed = np.r_[single, double]
        ax.vlines(np.arange(4), 0, observed, color=LIGHT_BLUE, lw=5, zorder=1)
        ax.scatter(
            np.arange(4), observed, s=58, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3
        )
        ax.scatter(3, expected, s=65, facecolor="white", edgecolor=GREY, linewidth=1.3, zorder=4)
        ax.plot([2.65, 3.0], [expected, expected], color=GREY, lw=1, ls=(0, (3, 2)))
        ax.annotate(
            r"$\varepsilon > 0$" if double > expected else r"$\varepsilon < 0$",
            xy=(3.04, (double + expected) / 2),
            xytext=(3.25, (double + expected) / 2),
            ha="left",
            va="center",
            fontsize=11,
            color=RED,
            arrowprops={"arrowstyle": "-[", "color": RED, "lw": 1.1},
        )
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xticks(range(4), labels)
        ax.set_xlim(-0.45, 3.65)
        ax.set_ylim(0, 1.03)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Fitness")
    axes[1].text(
        2.48,
        expected,
        "additive\nexpectation",
        ha="right",
        va="center",
        color=GREY,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.005,
        "The double mutant lies above or below the sum of the single effects",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.84, wspace=0.22)
    finish(fig, "option_3_pairwise_epistasis")


def option_global_epistasis(colour) -> None:
    """Show a monotone nonlinear readout of an additive latent trait."""
    axis = np.linspace(-1, 1, 161)
    x, y = np.meshgrid(axis, axis)
    latent = (x + y + 2) / 4
    observed = expit(8 * (latent - 0.5))
    titles = ["Additive latent trait", "Saturating fitness readout"]
    notes = ["Mutation effects add on the latent scale", "Magnitude changes without new peaks"]
    fig = plt.figure(figsize=(8.3, 2.65))
    for i, surface in enumerate([latent, observed]):
        ax = fig.add_axes([0.02 + 0.48 * i, 0.13, 0.46, 0.75], projection="3d")
        plot_surface(ax, x, y, surface, colour)
        fig.text(0.25 + 0.48 * i, 0.93, titles[i], ha="center", fontsize=13)
        fig.text(0.25 + 0.48 * i, 0.045, notes[i], ha="center", fontsize=10)
    finish(fig, "option_4_global_epistasis")


def make_contact_sheet() -> None:
    """Combine the raster previews into one selection sheet."""
    names = [
        "option_1_adaptive_walks",
        "option_2_combinatorial_paths",
        "option_3_pairwise_epistasis",
        "option_4_global_epistasis",
    ]
    labels = [
        "Option 1. Adaptive walks",
        "Option 2. Combinatorial mutation orders",
        "Option 3. Positive and negative pairwise epistasis",
        "Option 4. Global magnitude epistasis",
    ]
    fig = plt.figure(figsize=(8.3, 11.3))
    grid = fig.add_gridspec(8, 1, height_ratios=[0.10, 1] * 4, hspace=0.02)
    for index, (name, label) in enumerate(zip(names, labels, strict=True)):
        label_axis = fig.add_subplot(grid[2 * index])
        label_axis.text(0.01, 0.5, label, ha="left", va="center", fontsize=12, weight="bold")
        label_axis.set_axis_off()
        ax = fig.add_subplot(grid[2 * index + 1])
        image = plt.imread(OUTPUT / f"{name}.png")
        ax.imshow(image)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig(
        OUTPUT / "bottom_row_alternatives.png", dpi=220, bbox_inches="tight", pad_inches=0.08
    )
    plt.close(fig)


def main() -> None:
    """Render all four lower-row candidates and a contact sheet."""
    colour = style()
    option_walks(colour)
    option_combinatorial_paths(colour)
    option_pair_epistasis()
    option_global_epistasis(colour)
    make_contact_sheet()


if __name__ == "__main__":
    main()
