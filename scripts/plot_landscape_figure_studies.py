"""Render three draft figures about synthetic landscape properties."""

import json
from dataclasses import replace
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from mutascape.legacy import from_dms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "landscape_figure_studies"
BLUE = "#167db5"
PALE = "#c5dfec"
RED = "#db4943"
INK = "#181818"
GREY = "#64717a"
plt.rcParams.update(
    {
        "font.family": "STIX Two Text",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 15,
        "axes.linewidth": 0.6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save(fig, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def heading(fig, title, subtitle):
    fig.text(0.04, 0.96, title, fontsize=22, va="top")
    fig.text(0.04, 0.897, subtitle, fontsize=12, color=GREY, va="top")


def bare3d(ax, zlim, aspect=(1, 1, 0.67)):
    ax.set_proj_type("ortho")
    ax.view_init(elev=27, azim=-61)
    ax.set_box_aspect(aspect, zoom=1.06)
    ax.set_zlim(*zlim)
    ax.set_axis_off()


def surface_gallery():
    t = np.linspace(-1, 1, 101)
    x, y = np.meshgrid(t, t)
    slope = 0.58 * x + 0.30 * y
    rough = slope + 0.66 * np.sin(4.6 * x + 0.8) * np.cos(4.2 * y - 0.5)
    other = slope + 0.66 * np.sin(4.6 * x - 1.0) * np.cos(4.2 * y + 1.1)
    maps = [slope, rough, 0.65 * np.tanh(2 * rough), 0.65 * np.tanh(2 * other)]
    titles = [
        "(a) Additive slope",
        "(b) Context dependence",
        "(c) Saturation",
        "(d) Another surface",
    ]
    notes = [
        "Effects add along a steady slope.",
        "Interactions introduce peaks and valleys.",
        "A monotone map compresses heights.",
        "Unknown interactions change the shape.",
    ]
    fig = plt.figure(figsize=(14, 5.5))
    heading(fig, "What can change in a fitness landscape?", "A conceptual view of landscape shape")
    colour = LinearSegmentedColormap.from_list("paper_blue", ["#edf5f9", "#b4d6e7", "#559dc5"])
    for i, z in enumerate(maps):
        ax = fig.add_axes([0.015 + i * 0.247, 0.25, 0.248, 0.51], projection="3d")
        bare3d(ax, (-1.45, 1.45))
        ax.plot_surface(
            x,
            y,
            z,
            cmap=colour,
            vmin=-1.45,
            vmax=1.45,
            linewidth=0,
            antialiased=True,
            rcount=80,
            ccount=80,
            shade=False,
        )
        ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color=BLUE, linewidth=0.42, alpha=0.58)
        k = 53
        ax.plot(x[k], y[k], z[k] + 0.014, color=RED, lw=1.6, zorder=20)
        fig.text(0.139 + i * 0.247, 0.755, titles[i], ha="center", fontsize=14)
        fig.text(0.139 + i * 0.247, 0.242, notes[i], ha="center", fontsize=11)
    fig.text(0.045, 0.155, "Height is fitness", fontsize=12)
    fig.add_artist(
        Line2D([0.34, 0.375], [0.164, 0.164], transform=fig.transFigure, color=RED, lw=1.6)
    )
    fig.text(0.387, 0.155, "Same coordinate slice", fontsize=12)
    fig.text(
        0.04,
        0.055,
        "Schematic surfaces. Horizontal coordinates do not encode sequence distance. "
        "Panels (b) and (c) retain the same peak locations.",
        fontsize=11,
        color=GREY,
    )
    save(fig, "01_surface_gallery")


BITS = np.array(list(product([0, 1], repeat=5)))
VARIANTS = [{i + 1: "V" for i, bit in enumerate(row) if bit} for row in BITS]
ANCHORS = BITS.sum(axis=1) <= 1
NEIGHBOURS = np.sum(BITS[:, None] != BITS[None, :], axis=2) == 1
EDGES = [(a, b) for a, b in combinations(range(32), 2) if NEIGHBOURS[a, b]]
XY = (BITS - 0.5) @ np.array([[1, 0.1], [0.4, 0.9], [-0.8, 0.3], [-0.55, -0.85], [0.12, -0.6]])


def model():
    frame = pd.DataFrame(
        {
            "variant": ["WT"] + [f"A{i + 1}V" for i in range(5)],
            "fitness": [0, 0.45, 0.22, -0.32, 0.12, -0.15],
        }
    )
    generator = from_dms(frame, wt_sequence="AAAAA")
    generator = replace(generator, nonlinearity=replace(generator.nonlinearity, floor_hard=None))
    return generator


def peaks(values):
    return np.flatnonzero(np.all((values[:, None] > values[None, :]) | ~NEIGHBOURS, axis=1))


def route(values):
    path = [0]
    while True:
        adjacent = np.flatnonzero(NEIGHBOURS[path[-1]])
        best = adjacent[np.argmax(values[adjacent])]
        if values[best] <= values[path[-1]]:
            return path
        path.append(int(best))


def graph(ax, values, zlim, show_route=True, mark_anchors=True):
    bare3d(ax, zlim, (1, 1, 0.88))
    ax.computed_zorder = False
    for a, b in EDGES:
        ax.plot(*XY[[a, b]].T, values[[a, b]], color=BLUE, alpha=0.32, lw=0.75, zorder=1)
    ax.scatter(
        *XY.T, values, s=20, color=PALE, edgecolor=BLUE, linewidth=0.5, depthshade=False, zorder=3
    )
    if mark_anchors:
        ax.scatter(*XY[ANCHORS].T, values[ANCHORS], s=27, color=INK, depthshade=False, zorder=5)
    high = peaks(values)
    ax.scatter(
        *XY[high].T,
        values[high],
        s=83,
        facecolor="none",
        edgecolor=RED,
        linewidth=1.5,
        depthshade=False,
        zorder=6,
    )
    if show_route:
        p = route(values)
        ax.plot(*XY[p].T, values[p], color=RED, lw=2.1, zorder=4)
    ax.text(*XY[0], values[0] - 0.075 * np.ptp(zlim), "WT", fontsize=10)


def graph_gallery(g):
    couplings = [0, 0.1, 0.8]
    ys = [g.sample_dms_pairwise(0, coupling=c).evaluate_batch(VARIANTS) for c in couplings]
    fig = plt.figure(figsize=(13.8, 6.4))
    heading(
        fig,
        "Interactions change the routes through sequence space",
        "One toy library, 32 genotypes and the same six anchors",
    )
    for i, (c, values) in enumerate(zip(couplings, ys, strict=True)):
        ax = fig.add_axes([0.018 + 0.328 * i, 0.235, 0.313, 0.54], projection="3d")
        graph(ax, values, (-1.35, 0.88))
        titles = ["(a) No pair terms", "(b) Default coupling", "(c) Stronger interactions"]
        fig.text(0.175 + 0.328 * i, 0.80, titles[i], ha="center", fontsize=15)
        fig.text(
            0.175 + 0.328 * i, 0.755, rf"$\lambda = {c:g}$", ha="center", fontsize=13, color=GREY
        )
        n = len(peaks(values))
        fig.text(
            0.175 + 0.328 * i,
            0.22,
            f"{n} local peak{'s' if n != 1 else ''}",
            ha="center",
            fontsize=13,
        )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=INK,
            markeredgecolor=INK,
            label="WT and supplied singles",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=RED,
            label="Local peak",
        ),
        Line2D([], [], color=RED, lw=2, label="Greedy uphill walk"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.04,
        0.051,
        "Height is fitness. Each edge changes one residue. "
        "All panels share a layout and height scale.",
        fontsize=11,
        color=GREY,
    )
    fig.text(
        0.04,
        0.018,
        "Toy inputs, seed 0, smooth readout. "
        "The stronger interaction panel shows sensitivity beyond the manuscript default.",
        fontsize=11,
        color=GREY,
    )
    save(fig, "02_genotype_graphs")
    return ys


def axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#aeb8be")
    ax.tick_params(length=3, width=0.6, labelsize=10)


def mechanism(g):
    y = g.sample_dms_pairwise(0, coupling=0.8).evaluate_batch(VARIANTS)
    latent = np.asarray(g.nonlinearity.inverse(y))
    fig = plt.figure(figsize=(13.8, 8))
    heading(
        fig,
        "How Mutascape changes a landscape",
        "A toy example with fixed single-mutant anchors",
    )
    ax = fig.add_axes([0.025, 0.40, 0.46, 0.40], projection="3d")
    graph(ax, y, (-1.35, 0.88))
    fig.text(0.045, 0.825, "(a) A sampled landscape", fontsize=15)
    fig.text(0.045, 0.395, "Black points fix WT and single mutants.", fontsize=12)
    fig.text(0.045, 0.361, "Edges show allowed steps. Red rings mark local peaks.", fontsize=12)

    ax = fig.add_axes([0.585, 0.535, 0.345, 0.23])
    axes_style(ax)
    # Find a mutation with opposite effects on two genetic backgrounds.
    choices = []
    for site in range(5):
        pairs = [(a, b) for a, b in EDGES if BITS[a, site] != BITS[b, site]]
        plus = [(a, b) for a, b in pairs if y[b] > y[a]]
        minus = [(a, b) for a, b in pairs if y[b] < y[a]]
        if plus and minus:
            p = max(plus, key=lambda e: y[e[1]] - y[e[0]])
            m = min(minus, key=lambda e: y[e[1]] - y[e[0]])
            choices.append((min(y[p[1]] - y[p[0]], y[m[0]] - y[m[1]]), site, p, m))
    _, site, p, m = max(choices)
    for pair, colour, label in [(p, BLUE, "Background 1"), (m, RED, "Background 2")]:
        ax.plot(
            [0, 1], y[list(pair)], color=colour, marker="o", markersize=4.5, lw=1.5, label=label
        )
    ax.set_xticks([0, 1], ["Without mutation", "With mutation"])
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylabel("Fitness")
    ax.legend(
        frameon=False,
        fontsize=10,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        borderaxespad=0,
    )
    fig.text(0.555, 0.825, "(b) The same mutation can help or harm", fontsize=15)
    fig.text(0.585, 0.435, f"A{site + 1}V changes sign between two backgrounds.", fontsize=12)

    ax = fig.add_axes([0.085, 0.11, 0.32, 0.17])
    axes_style(ax)
    z = np.linspace(-6, 6, 300)
    ax.plot(z, g.nonlinearity.forward(z), color=BLUE, lw=1.6)
    ax.axhline(g.nonlinearity.floor, ls=(0, (3, 4)), color=BLUE, lw=0.8, alpha=0.6)
    ax.axhline(g.nonlinearity.ceiling, ls=(0, (3, 4)), color=BLUE, lw=0.8, alpha=0.6)
    ax.scatter(latent, y, s=10, color=INK, zorder=4)
    ax.set_xlabel("Latent score")
    ax.set_ylabel("Fitness")
    ax.set_xticks([-6, 0, 6])
    ax.set_yticks([-1, 0, 1])
    fig.text(0.045, 0.319, "(c) Saturation preserves the order", fontsize=15)

    ax = fig.add_axes([0.585, 0.11, 0.345, 0.17])
    axes_style(ax)
    # A fixed path adds one substitution at each step.
    indices = [0, 16, 24, 28, 30, 31]
    for seed, colour in [(0, BLUE), (19, RED)]:
        values = g.sample_dms_pairwise(seed, coupling=0.8).evaluate_batch(VARIANTS)
        ax.plot(
            range(6),
            values[indices],
            color=colour,
            lw=1.4,
            marker="o",
            markersize=3.5,
            label=f"Seed {seed}",
        )
    ax.scatter([0, 1], y[indices[:2]], color=INK, s=22, zorder=5)
    ax.set_xticks(range(6), ["WT", "1", "2", "3", "4", "5"])
    ax.set_xlabel("Mutations along one fixed path")
    ax.set_ylabel("Fitness")
    ax.legend(
        frameon=False,
        fontsize=10,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        borderaxespad=0,
    )
    fig.text(0.555, 0.319, "(d) Samples agree at the anchors", fontsize=15)
    fig.text(
        0.045,
        0.015,
        r"Toy library, five binary sites. $\lambda = 0.8$ makes the effects visible. "
        r"The manuscript uses $\lambda = 0.1$.",
        fontsize=11,
        color=GREY,
    )
    save(fig, "03_model_explanation")
    assert np.array_equal(peaks(latent), peaks(y))
    return {"sign_epistasis_site": int(site + 1), "background_pairs": [list(p), list(m)]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = model()
    surface_gallery()
    ys = graph_gallery(g)
    metadata = mechanism(g)
    for values in ys:
        np.testing.assert_allclose(values[ANCHORS], ys[0][ANCHORS], atol=1e-12)
        p = route(values)
        assert np.all(np.diff(values[p]) > 0)
        assert p[-1] in peaks(values)
    metadata.update(
        {
            "inputs": [0, 0.45, 0.22, -0.32, 0.12, -0.15],
            "couplings": [0, 0.1, 0.8],
            "seed": 0,
            "peak_counts": [len(peaks(y)) for y in ys],
            "readout": str(g.nonlinearity),
            "anchor_checks": "passed",
            "uphill_path_checks": "passed",
            "monotone_peak_check": "passed",
        }
    )
    (OUT / "checks.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame(
        {
            "genotype": ["".join(map(str, row)) for row in BITS],
            "anchor": ANCHORS,
            **{f"coupling_{c}": y for c, y in zip([0, 0.1, 0.8], ys, strict=True)},
        }
    ).to_csv(OUT / "toy_scores.csv", index=False)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
