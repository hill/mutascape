"""Build structural-prior contact map figures for the paper.

Produces:
  - notebooks/validation/output_structural/gfp_contact_map.{pdf,png}

Run:
    uv run python scripts/build_contact_map_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mutascape.validation.figures import save_fig

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data/gfp"
OUTPUT_DIR = REPO_ROOT / "notebooks/validation/output_structural"


def fig_gfp_contact_map() -> plt.Figure:
    data = np.load(DATA_DIR / "gfp_distances.npz")
    distances = data["distances"]
    positions = data["positions"]
    threshold = 8.0
    contacts = distances <= threshold
    np.fill_diagonal(contacts, False)

    n = len(positions)
    n_contacts = int(np.triu(contacts, k=1).sum())
    n_possible = n * (n - 1) // 2
    density = n_contacts / n_possible
    degree = contacts.sum(axis=1)

    rows, cols = np.where(np.triu(contacts, k=1))
    palette = sns.color_palette("deep")

    fig, (ax, ax_info) = plt.subplots(
        1,
        2,
        figsize=(7.4, 5.2),
        gridspec_kw={"width_ratios": (5.4, 1.35)},
        constrained_layout=True,
    )

    ax.scatter(cols, rows, s=3, marker="s", color=palette[0], alpha=0.75, linewidths=0)
    ax.scatter(rows, cols, s=3, marker="s", color=palette[0], alpha=0.75, linewidths=0)
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="0.75", linewidth=0.8)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Residue position")
    ax.set_title("GFP structural contact prior (PDB 2WUR, 8 A C-beta threshold)")

    tick_idx = np.linspace(0, n - 1, 8, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(int(positions[i])) for i in tick_idx], rotation=0)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(int(positions[i])) for i in tick_idx])

    ax_info.hist(degree, bins=np.arange(degree.max() + 2) - 0.5, color=palette[1], alpha=0.85)
    ax_info.set_xlabel("Contact degree")
    ax_info.set_ylabel("Residues")
    ax_info.set_title("Degree")
    ax_info.spines[["top", "right"]].set_visible(False)
    ax_info.text(
        0.02,
        0.98,
        (
            f"{n} resolved DMS positions\n"
            f"{n_contacts:,} structural contacts\n"
            f"{n_possible:,} possible pairs\n"
            f"{100 * density:.1f}% contact density"
        ),
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        linespacing=1.35,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.35"),
    )

    return fig


def main() -> None:
    save_fig(fig_gfp_contact_map(), "gfp_contact_map", OUTPUT_DIR)


if __name__ == "__main__":
    main()
