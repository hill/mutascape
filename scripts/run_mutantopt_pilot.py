"""Run a small mutantopt benchmark pilot on frozen mutascape reference configs.

The pilot exports compact sequence-only datasets over the mutable positions
of a complete combinatorial landscape, then benchmarks a small method set
against a `noise_sigma` sweep. This keeps the first MLDE/ALDE comparison
cheap and reproducible while preserving the landscape topology that matters
for benchmarking.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "benchmark_pilot"


def _bootstrap_sys_path() -> None:
    for path in (REPO_ROOT / "src", WORKSPACE_ROOT / "mutantopt" / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _parse_args() -> argparse.Namespace:
    _bootstrap_sys_path()
    from mutascape.validation.reference_configs import REFERENCE_CONFIGS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="gb1_headline",
        choices=sorted(REFERENCE_CONFIGS),
        help="Frozen reference config to benchmark.",
    )
    parser.add_argument(
        "--noise-sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.5],
        help="Noise levels to evaluate.",
    )
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=6)
    parser.add_argument("--initial-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activity-mode",
        choices=["scaled", "raw"],
        default="scaled",
        help="Activity values passed to mutantopt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory.",
    )
    return parser.parse_args()


def _compact_wt(config: Any) -> str:
    return "".join(config.wt_aa_at_position[pos] for pos in config.positions)


def export_compact_dataset(config: Any, noise_sigma: float, out_dir: Path) -> tuple[Path, Path]:
    from mutascape.validation.fourier import observed_alphabets

    out_dir.mkdir(parents=True, exist_ok=True)
    real_df = config.load_real_dataframe()
    alphabets = observed_alphabets(real_df, config.positions, config.wt_aa_at_position)
    landscape = config.build_landscape(noise_sigma=noise_sigma)

    combos = list(product(*(alphabets[pos] for pos in config.positions)))
    mutations = [
        {
            pos: aa
            for pos, aa in zip(config.positions, combo)
            if aa != config.wt_aa_at_position[pos]
        }
        for combo in combos
    ]
    fitness = landscape.evaluate_batch(mutations)

    dataset_csv = out_dir / "landscape.csv"
    pd.DataFrame({
        "sequence": ["".join(combo) for combo in combos],
        "fitness": fitness.astype(float),
    }).to_csv(dataset_csv, index=False)

    wt_fasta = out_dir / "wt.fasta"
    wt_fasta.write_text(f">WT\n{_compact_wt(config)}\n")

    metadata = {
        "source_config": config.name,
        "noise_sigma": noise_sigma,
        "positions": list(config.positions),
        "wt_compact": _compact_wt(config),
        "alphabets": {str(k): v for k, v in alphabets.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return dataset_csv, wt_fasta


def run_pilot(config_name: str, noise_sigma: float, args: argparse.Namespace) -> pd.DataFrame:
    _bootstrap_sys_path()
    from mutantopt.core import BenchmarkingExperiment, ExperimentConfig
    from mutantopt.data import DMSDataset
    from mutantopt.encoding import OneHotEncoding
    from mutantopt.methods.alde import ALDEMethod
    from mutantopt.methods.evolvepro import EvolveProMethod
    from mutantopt.methods.random import RandomMethod

    from mutascape.validation.reference_configs import get_reference_config

    config = get_reference_config(config_name)
    slug = f"{config_name}_noise_{noise_sigma:.2f}".replace(".", "p")
    dataset_dir = args.output_dir / slug / "dataset"
    results_dir = args.output_dir / slug / "results"
    dataset_csv, wt_fasta = export_compact_dataset(config, noise_sigma, dataset_dir)

    dataset = DMSDataset(
        dms_path=dataset_csv,
        wt_path=wt_fasta,
        name=slug,
        variant_column="sequence",
        activity_columns=["fitness"],
        variant_format="sequence",
    )

    methods = [
        RandomMethod(name="Random"),
        EvolveProMethod(name="EvolvePro-RF", model_type="randomforest"),
        ALDEMethod(name="ALDE-UCB", acquisition="ucb"),
    ]
    experiment = BenchmarkingExperiment(
        dataset=dataset,
        methods=methods,
        encodings=[OneHotEncoding()],
        config=ExperimentConfig(
            n_runs=args.n_runs,
            n_iterations=args.n_iterations,
            initial_size=args.initial_size,
            batch_size=args.batch_size,
            random_seed=args.seed,
            activity_mode=args.activity_mode,
            output_dir=results_dir,
        ),
    )
    result = experiment.run()

    summary = result.to_dataframe()
    final = summary.loc[summary.groupby(["method", "encoding", "run"])["iteration"].idxmax()]
    aggregate = (
        final.groupby(["method", "encoding"])["best_value"]
        .agg(final_best_mean="mean", final_best_std="std")
        .reset_index()
    )
    aggregate["config"] = config_name
    aggregate["noise_sigma"] = noise_sigma
    aggregate["dataset_dir"] = str(dataset_dir)
    aggregate["results_dir"] = str(results_dir)
    aggregate.to_csv(results_dir / "pilot_final_summary.csv", index=False)
    return aggregate


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for noise_sigma in args.noise_sigmas:
        print(f"running mutantopt pilot for {args.config} at noise_sigma={noise_sigma}")
        frames.append(run_pilot(args.config, noise_sigma, args))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output_dir / f"{args.config}_pilot_summary.csv", index=False)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
