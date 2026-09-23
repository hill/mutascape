"""Plot analytic landscape schematics with shared axes and colour scales.

These surfaces illustrate properties and do not fit measured protein data.
Run with --manuscript-dir to copy the figure into a manuscript's figures folder.
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import maximum_filter
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#167db5"
RED = "#d84a45"


def draw_sequence_walk(fig, colour):
    """Show the same complete three-mutation graph in plan and fitness height."""
    labels = ["WT", "A", "B", "C", "AB", "AC", "BC", "ABC"]
    sets = [set() if label == "WT" else set(label) for label in labels]
    points = np.array(
        [
            [-0.8, -0.6],
            [-0.9, 0.2],
            [-0.15, -0.8],
            [0.4, -0.7],
            [-0.4, 0.75],
            [0.35, 0.5],
            [0.9, -0.15],
            [0.65, 0.9],
        ]
    )
    fitness = np.array([0.10, 0.70, 0.30, 0.20, 0.50, 0.40, 0.60, 0.95])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if len(sets[i] ^ sets[j]) == 1]
    paths = [([0, 1], RED), ([0, 2, 6, 7], BLUE)]
    assert len(edges) == 12
    assert all(fitness[1] > fitness[j if i == 1 else i] for i, j in edges if 1 in (i, j))
    assert all(np.all(np.diff(fitness[path]) > 0) for path, _ in paths)
    ax = fig.add_axes([0.065, 0.045, 0.36, 0.37])
    for i, j in edges:
        ax.plot(*points[[i, j]].T, color="#b4c3ca", lw=0.85, zorder=1)
    for path, stroke in paths:
        for i, j in zip(path[:-1], path[1:]):
            ax.annotate(
                "",
                xy=points[j],
                xytext=points[i],
                arrowprops=dict(arrowstyle="->", color=stroke, lw=1.8, shrinkA=13, shrinkB=13),
                zorder=3,
            )
    ax.scatter(
        *points.T,
        c=fitness,
        cmap=colour,
        vmin=0,
        vmax=1,
        s=360,
        edgecolors="#263c47",
        linewidths=0.8,
        zorder=4,
    )
    for (px, py), value, label in zip(points, fitness, labels):
        ax.text(
            px,
            py,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if value >= 0.65 else "#161616",
            zorder=5,
        )
        ax.text(px, py + 0.20, label, ha="center", fontsize=9)
    ax.set(xlim=(-1.18, 1.18), ylim=(-1.0, 1.3))
    ax.set_axis_off()
    ax = fig.add_axes([0.49, 0.025, 0.45, 0.40], projection="3d")
    ax.computed_zorder = False
    grid = np.linspace(-1.45, 1.45, 150)
    xx, yy = np.meshgrid(grid, grid)

    def walk_hills(query):
        """Form one rising ridge and one separate local hill."""
        distances, heights = [], []
        for i, j in [(0, 2), (2, 6), (6, 7)]:
            direction = points[j] - points[i]
            fraction = np.clip((query - points[i]) @ direction / (direction @ direction), 0, 1)
            nearest = points[i] + fraction[:, None] * direction
            distances.append(np.linalg.norm(query - nearest, axis=1))
            heights.append(fitness[i] + fraction * (fitness[j] - fitness[i]))
        distances, heights = np.array(distances), np.array(heights)
        segment = distances.argmin(axis=0)
        column = np.arange(len(query))
        radius = distances[segment, column] / 0.72
        envelope = np.maximum(1 - radius, 0) ** 4 * (1 + 4 * radius)
        ridge = (heights[segment, column] - 0.035) * envelope
        radius = np.linalg.norm(query - points[1], axis=1) / 0.62
        local = (fitness[1] - 0.035) * np.maximum(1 - radius, 0) ** 4 * (1 + 4 * radius)
        return 0.035 + ridge + local

    assert np.allclose(walk_hills(points[[0, 1, 2, 6, 7]]), fitness[[0, 1, 2, 6, 7]])
    zz = walk_hills(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    assert 0 <= zz.min() < zz.max() <= 1
    summit = np.array([xx.ravel()[zz.argmax()], yy.ravel()[zz.argmax()]])
    assert np.linalg.norm(summit - points[7]) < 0.05
    ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=colour,
        norm=Normalize(0, 1),
        rcount=65,
        ccount=65,
        edgecolor="#a5cbdc",
        linewidth=0.2,
        alpha=0.9,
        shade=False,
        zorder=1,
    )
    for path, stroke in paths:
        ax.plot(
            points[path, 0],
            points[path, 1],
            fitness[path] + 0.025,
            color=stroke,
            marker="o",
            ms=4,
            lw=2,
            zorder=5,
        )
        for i in path:
            ax.text(*points[i], fitness[i] + 0.09, labels[i], fontsize=9, zorder=6)
    ax.set_proj_type("ortho")
    ax.view_init(elev=29, azim=-68)
    ax.set_box_aspect((1, 1, 0.85), zoom=1.1)
    ax.set(xlim=(-1.45, 1.45), ylim=(-1.45, 1.45), zlim=(0, 1.15))
    ax.set_axis_off()
    fig.text(0.25, 0.455, "(c) Combinatorial sequence graph", ha="center", fontsize=13)
    fig.text(0.715, 0.455, "(d) Mutation order shapes the walk", ha="center", fontsize=13)
    fig.text(0.25, 0.015, "Each edge changes one mutation", ha="center", fontsize=10)
    fig.text(0.59, 0.015, "Local optima", color=RED, ha="center", fontsize=10)
    fig.text(0.82, 0.015, "Mutation route to global optima", color=BLUE, ha="center", fontsize=9)


def landscape_surfaces(size: int = 161) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return four fixed surfaces on the same illustrative coordinate grid."""
    axis = np.linspace(-1, 1, size)
    x, y = np.meshgrid(axis, axis)
    # A monotone readout of an additive radial trait gives one isolated peak.
    weak = 0.035 + 0.905 * np.exp(-5.8 * (x**2 + y**2))
    rng = np.random.default_rng(19)
    interacting = np.zeros_like(x)
    for centre_x, centre_y in rng.uniform(-0.88, 0.88, (24, 2)):
        width_x, width_y = rng.uniform(0.075, 0.20, 2)
        height = rng.uniform(0.25, 0.95)
        interacting += height * np.exp(
            -0.5 * (((x - centre_x) / width_x) ** 2 + ((y - centre_y) / width_y) ** 2)
        )
    interacting = 0.035 + 0.905 * interacting / interacting.max()
    radius = (x / 0.88) ** 2 + (y / 0.76) ** 2
    broad = 0.04 + 0.90 * expit(9 * (1 - radius))
    narrow = 0.04 + 0.90 * expit(9 * (1 - 5 * radius))
    return x, y, [weak, interacting, broad, narrow]


def checks(surfaces: list[np.ndarray]) -> dict:
    """Check the geometric claims made by the schematic."""
    weak, interacting, broad, narrow = surfaces
    interaction_residuals = []
    peaks = []
    for surface in surfaces:
        assert np.all(np.isfinite(surface))
        assert 0 <= surface.min() <= surface.max() <= 1
        residual = surface - surface.mean(0) - surface.mean(1)[:, None] + surface.mean()
        interaction_residuals.append(float(np.sqrt(np.mean(residual**2))))
        neighbours = np.ones((3, 3), dtype=bool)
        neighbours[1, 1] = False
        local = surface > maximum_filter(
            surface, footprint=neighbours, mode="constant", cval=-np.inf
        )
        local[[0, -1], :] = False
        local[:, [0, -1]] = False
        peaks.append(int(local.sum()))
    # These schematic coordinates do not estimate protein epistasis strength.
    assert np.max(weak[[0, -1], :]) < 0.04
    coordinate_effects = np.diff(interacting, axis=1)
    assert np.any((coordinate_effects.min(0) < 0) & (coordinate_effects.max(0) > 0))
    assert peaks[0] == 1 and peaks[1] > 1
    active = [float(np.mean(surface > 0.5)) for surface in [broad, narrow]]
    assert active[0] > 3 * active[1]
    # Both functional-region panels apply the same monotone map to an additive radial trait.
    assert peaks[2] == peaks[3] == 1
    return {
        "kind": "analytic schematic, not an empirical fit or a Mutascape sample",
        "interior_peak_counts": peaks,
        "coordinate_interaction_rms": interaction_residuals,
        "illustrative_functional_threshold": 0.5,
        "broad_and_narrow_area_fractions": active,
        "warning": "Grid areas and coordinate interactions are not protein sequence metrics.",
        "checks": "passed",
    }


def render(output: Path) -> None:
    """Save the figure as PNG, SVG and PDF, plus its grid and checks."""
    output.mkdir(parents=True, exist_ok=True)
    x, y, surfaces = landscape_surfaces()
    diagnostics = checks(surfaces)
    plt.rcParams.update(
        {
            "font.family": "STIX Two Text",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "text.color": "#161616",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    colour = LinearSegmentedColormap.from_list(
        "fitness_blue", ["#f4f8fb", "#d1e6f0", "#80b8d5", "#247fae"]
    )
    fig = plt.figure(figsize=(8.3, 5.15))
    titles = [
        "(a) Low epistasis",
        "(b) Strong sign epistasis",
        "(c) Broad functional region",
        "(d) Narrow functional region",
    ]
    notes = [
        "Smooth, single peak",
        "Multiple local peaks",
        "Many variants retain function",
        "Most variants have low fitness",
    ]
    for index, surface in enumerate(surfaces[:2]):
        column, row = index % 2, index // 2
        left, bottom = 0.025 + column * 0.465, 0.535 - row * 0.46
        ax = fig.add_axes([left, bottom, 0.45, 0.39], projection="3d")
        ax.computed_zorder = False
        ax.set_proj_type("ortho")
        ax.view_init(elev=30, azim=-58)
        ax.set_box_aspect((1, 1, 0.67), zoom=1.08)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1.05))
        ax.set_axis_off()
        ax.plot_surface(
            x,
            y,
            surface,
            cmap=colour,
            norm=Normalize(0, 1),
            rcount=41,
            ccount=41,
            linewidth=0.24,
            edgecolor="#8ebdd5",
            antialiased=True,
            shade=False,
            zorder=1,
        )
        if index >= 2:
            # Show the near half of the contour so the surface hides its far side.
            angle = np.deg2rad(-58) + np.linspace(-np.pi / 2, np.pi / 2, 161)
            radius_squared = 1 - np.log((0.5 - 0.04) / (0.94 - 0.5)) / 9
            radius_squared /= 1 if index == 2 else 5
            contour_x = 0.88 * np.sqrt(radius_squared) * np.cos(angle)
            contour_y = 0.76 * np.sqrt(radius_squared) * np.sin(angle)
            ax.plot(contour_x, contour_y, np.full_like(angle, 0.503), color=RED, lw=1.4, zorder=4)
        fig.text(left + 0.225, bottom + 0.402, titles[index], ha="center", fontsize=13)
        fig.text(left + 0.222, bottom - 0.003, notes[index], ha="center", fontsize=11)

    draw_sequence_walk(fig, colour)
    colour_axis = fig.add_axes([0.953, 0.55, 0.012, 0.27])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colour), cax=colour_axis)
    bar.set_ticks([0, 1], labels=["Low", "High"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, pad=4, labelsize=10)
    bar.ax.set_title("Fitness", fontsize=11, pad=9)
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(
            output / f"landscape_properties.{extension}",
            dpi=260,
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)
    np.savez_compressed(
        output / "landscape_properties.npz",
        x=x,
        y=y,
        weak=surfaces[0],
        rugged=surfaces[1],
        broad=surfaces[2],
        narrow=surfaces[3],
    )
    (output / "landscape_properties_checks.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2))


def main() -> None:
    """Render locally and optionally copy the exports into a manuscript."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "landscape_properties")
    parser.add_argument("--manuscript-dir", type=Path)
    args = parser.parse_args()
    render(args.output)
    if args.manuscript_dir:
        figures = args.manuscript_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        for extension in ["svg", "pdf", "png"]:
            source = args.output / f"landscape_properties.{extension}"
            shutil.copy2(source, figures / source.name)


if __name__ == "__main__":
    main()
