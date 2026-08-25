"""Rebuild the frozen publication outputs from raw prepared data.

This is the non-notebook smoke script for the paper pipeline. It regenerates:
  - GB1 noise sweep CSV
  - TrpB noise sweep CSV
  - cross-protein headline figure set
  - structural-prior contact map figure

Run:
    uv run python scripts/rebuild_reference_outputs.py
"""

from build_contact_map_figure import main as build_contact_map_figure_main
from build_noise_figures import main as build_noise_figures_main
from run_noise_sweep import run_sweep


def main() -> None:
    print("rebuilding GB1 headline sweep...")
    run_sweep("gb1")
    print("\nrebuilding TrpB headline sweep...")
    run_sweep("trpb")
    print("\nrebuilding headline figures...")
    build_noise_figures_main()
    print("\nrebuilding structural-prior figure...")
    build_contact_map_figure_main()


if __name__ == "__main__":
    main()
