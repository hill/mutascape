"""End-to-end Mutascape walkthrough on CreiLOV.

Loads the CreiLOV DMS dataset, predicts a structural contact map directly from
the wild-type sequence with ESMFold, builds a synthetic combinatorial fitness
landscape, queries it in a few different ways, and renders the contact map.

Run:
    uv run python examples/creilov.py

The first run calls the ESM Atlas API to fold the sequence (about 30 seconds).
The predicted PDB is cached under ~/.cache/mutascape/structures/ so subsequent
runs are instant.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mutascape import (
    ContactMap,
    Landscape,
    Mutant,
    RandomEpistasis,
    SignConsistentEpistasis,
    GlobalEpistasisSigmoid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "creilov"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_wt_sequence(fasta_path: Path) -> str:
    lines = fasta_path.read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def pick_contacting_dms_pair(contacts: ContactMap, landscape: Landscape) -> tuple[int, int] | None:
    """First pair of DMS-covered positions that appear in the ESMFold contact graph."""
    dms_positions = set(landscape.single_site_effects.keys())
    for i, j in contacts.contact_pairs():
        if i in dms_positions and j in dms_positions:
            return i, j
    return None


def pick_mutation_at(landscape: Landscape, position: int) -> str:
    """Pick the single-site mutation with the largest fitness effect at a position."""
    effects = landscape.single_site_effects[position]
    best_aa, _ = max(effects.items(), key=lambda kv: abs(kv[1]))
    wt_aa = landscape.wild_type[position]
    return f"{wt_aa}{position}{best_aa}"


def render_contact_map(contacts: ContactMap, out_path: Path) -> None:
    matrix = contacts.distances <= contacts.threshold
    np.fill_diagonal(matrix, False)
    n = contacts.n_positions
    rows, cols = np.where(np.triu(matrix, k=1))
    degree = matrix.sum(axis=1)
    positions = contacts.positions

    fig, (ax, ax_info) = plt.subplots(
        1,
        2,
        figsize=(7.4, 5.0),
        gridspec_kw={"width_ratios": (5.4, 1.35)},
        constrained_layout=True,
    )

    ax.scatter(cols, rows, s=4, marker="s", color="#1F77B4", alpha=0.8, linewidths=0)
    ax.scatter(rows, cols, s=4, marker="s", color="#1F77B4", alpha=0.8, linewidths=0)
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="0.78", linewidth=0.7)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Residue position")
    ax.set_title(f"CreiLOV ESMFold contact map ({contacts.threshold:.0f} A)")

    tick_idx = np.linspace(0, n - 1, 7, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(int(positions[i])) for i in tick_idx])
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(int(positions[i])) for i in tick_idx])

    ax_info.hist(degree, bins=np.arange(degree.max() + 2) - 0.5, color="#FF8B3D", alpha=0.85)
    ax_info.set_xlabel("Contact degree")
    ax_info.set_ylabel("Residues")
    ax_info.set_title("Degree")
    ax_info.spines[["top", "right"]].set_visible(False)
    n_possible = n * (n - 1) // 2
    ax_info.text(
        0.02,
        0.98,
        (
            f"{n} positions\n"
            f"{contacts.n_contacts:,} contacts\n"
            f"{n_possible:,} possible pairs\n"
            f"{100 * contacts.density:.1f}% density"
        ),
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    wt_seq = load_wt_sequence(DATA_DIR / "creilov_wt.fasta")
    wt_dict = {i + 1: aa for i, aa in enumerate(wt_seq)}
    print(f"loaded CreiLOV WT sequence, length {len(wt_seq)}")

    print("predicting structure with ESMFold (cached after first call)...")
    contacts = ContactMap.from_sequence(wt_seq, threshold=8.0)
    print(
        f"contact map: {contacts.n_positions} positions, "
        f"{contacts.n_contacts:,} contacts, density {100 * contacts.density:.1f}%"
    )

    landscape = Landscape.from_dms(
        dms_path=DATA_DIR / "creilov.csv",
        wt_sequence=wt_seq,
        contact_map=contacts,
        epistasis=RandomEpistasis(sigma=0.3, seed=42),
        alpha=1.0,
        seed=42,
    )

    print("\nfitness queries")
    print(f"  WT (reference)             : {landscape['']:+.4f}")
    print(f"  T7S (single)               : {landscape['T7S']:+.4f}")
    print(f"  R60D (single)              : {landscape['R60D']:+.4f}")
    print(f"  T7S + R60D (double)        : {landscape[Mutant('T7S') + Mutant('R60D')]:+.4f}")
    triple = Mutant("T7S") + Mutant("R60D") + Mutant("V103I")
    print(f"  {triple} (triple)  : {landscape[triple]:+.4f}")
    print(f"    depth                    : {triple.depth}")
    print(f"    positions                : {dict(triple)}")

    pair = pick_contacting_dms_pair(contacts, landscape)
    if pair is not None:
        i, j = pair
        m_i, m_j = pick_mutation_at(landscape, i), pick_mutation_at(landscape, j)
        double = Mutant(m_i, wild_type=wt_dict) + Mutant(m_j, wild_type=wt_dict)
        actual = landscape[double]
        additive = landscape.additive_fitness(dict(double))
        print(f"\nepistatic deviation on a contacting pair ({i}, {j})")
        print(f"  {m_i} + {m_j} actual    : {actual:+.4f}")
        print(f"  {m_i} + {m_j} additive  : {additive:+.4f}")
        print(f"  pairwise epistatic delta : {actual - additive:+.4f}")

    print("\nalpha sweep at depth 3 (sample stats)")
    for alpha in (0.0, 1.0, 2.5):
        ls = Landscape.from_dms(
            dms_path=DATA_DIR / "creilov.csv",
            wt_sequence=wt_seq,
            contact_map=contacts,
            epistasis=RandomEpistasis(sigma=0.3, seed=42),
            alpha=alpha,
            seed=42,
        )
        stats = ls.stats(n_samples=2000, n_mutations=3)
        print(
            f"  alpha={alpha:.1f}  additive_corr={stats['additive_correlation']:.3f}  "
            f"sign_epi_frac={stats['fraction_sign_epistasis']:.3f}"
        )

    print("\nepistasis-model swap at alpha=1.0 (depth 3 sample stats)")
    for label, model in (
        ("Random         ", RandomEpistasis(sigma=0.3, seed=42)),
        ("SignConsistent ", SignConsistentEpistasis(sigma=0.3, seed=42)),
    ):
        ls = Landscape.from_dms(
            dms_path=DATA_DIR / "creilov.csv",
            wt_sequence=wt_seq,
            contact_map=contacts,
            epistasis=model,
            alpha=1.0,
            seed=42,
        )
        stats = ls.stats(n_samples=2000, n_mutations=3)
        print(
            f"  {label} additive_corr={stats['additive_correlation']:.3f}  "
            f"sign_epi_frac={stats['fraction_sign_epistasis']:.3f}"
        )

    print("\nstreaming first 5 sampled triple mutants")
    for variant in landscape.generate(n_mutations=3, n=5, seed=7):
        m = Mutant(variant.mutations, wild_type=wt_dict)
        print(f"  {str(m):20s}  fitness={variant.fitness:+.4f}")

    print("\ncombining with a global-epistasis function and noise")
    nonlin = GlobalEpistasisSigmoid.fit_from_singles(
        landscape.single_site_effects,
        combinatorial_upside=2.0,
    )
    realistic = Landscape.from_dms(
        dms_path=DATA_DIR / "creilov.csv",
        wt_sequence=wt_seq,
        contact_map=contacts,
        epistasis=RandomEpistasis(sigma=0.3, seed=42),
        alpha=1.0,
        nonlinearity=nonlin,
        noise_sigma=0.15,
        seed=42,
    )
    print(f"  T7S latent       : {landscape['T7S']:+.4f}")
    print(f"  T7S sigmoid+noise: {realistic['T7S']:+.4f}")

    fig_path = OUT_DIR / "creilov_contact_map.png"
    render_contact_map(contacts, fig_path)
    print(f"\nrendered contact map to {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
