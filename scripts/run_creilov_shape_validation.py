"""Compare current Mutascape draws with the measured CreiLOV landscape.

The target contributes only its wild type and single mutants. The prior uses
the other SSMuLA and CombinGym proteins. Each comparison uses the same measured
variants. The script reports fitness-distribution and Walsh-spectrum distances.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.generators import dms_pairwise_readout_tensor
from mutascape.benchmarks.splits import mutation_order_split
from mutascape.benchmarks.topology import measured_tensor, supported_cube
from mutascape.completion import LandscapePrior, PosteriorLandscape
from mutascape.validation.fourier import (
    binary_cube_spectral_metrics,
    binary_cube_spectrum,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path, required=True)
    parser.add_argument("--combingym-root", type=Path, required=True)
    parser.add_argument("--dms-csv", type=Path, default=Path("data/creilov/creilov.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--coupling", type=float, default=0.1)
    return parser.parse_args()


def _binary_indices(mask: np.ndarray, cube) -> np.ndarray:
    coordinates = np.argwhere(mask)
    wild_type = np.asarray(
        [cube.alphabets[position].index(cube.wild_type[position]) for position in cube.positions]
    )
    mutated = coordinates != wild_type
    weights = 1 << np.arange(len(cube.positions), dtype=np.int64)
    return np.asarray(mutated @ weights, dtype=np.int64)


def _distribution_distances(real: np.ndarray, synthetic: np.ndarray) -> tuple[float, float, float]:
    wasserstein = float(stats.wasserstein_distance(real, synthetic))
    scale = float(np.std(real))
    normalised = wasserstein / scale if scale > 1e-12 else float("nan")
    synthetic_scale = float(np.std(synthetic))
    if scale > 1e-12 and synthetic_scale > 1e-12:
        standardised = float(
            stats.wasserstein_distance(
                (real - np.mean(real)) / scale,
                (synthetic - np.mean(synthetic)) / synthetic_scale,
            )
        )
    else:
        standardised = float("nan")
    return wasserstein, normalised, standardised


def _row(
    label: str,
    draw: int,
    real: np.ndarray,
    synthetic: np.ndarray,
    cell_indices: np.ndarray,
    n_positions: int,
) -> dict[str, object]:
    wasserstein, normalised, standardised = _distribution_distances(real, synthetic)
    full = binary_cube_spectral_metrics(real, synthetic, cell_indices, n_positions, max_order=None)
    first_two = binary_cube_spectral_metrics(
        real, synthetic, cell_indices, n_positions, max_order=2
    )
    return {
        "model": label,
        "draw": draw,
        "fitness_w1": wasserstein,
        "fitness_w1_real_sd": normalised,
        "fitness_shape_w1": standardised,
        "walsh_l1": full["walsh_l1"],
        "order_w1": full["order_w1"],
        "walsh_l1_orders_1_2": first_two["walsh_l1"],
        "order_w1_orders_1_2": first_two["order_w1"],
    }


def _standardise(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    if scale <= 1e-12:
        return values - np.mean(values)
    return (values - np.mean(values)) / scale


def _histogram_table(real: np.ndarray, baseline: np.ndarray, draws: np.ndarray) -> pd.DataFrame:
    lower = float(min(real.min(), baseline.min(), draws.min()))
    upper = float(max(real.max(), baseline.max(), draws.max()))
    edges = np.linspace(lower, upper, 81)
    centres = (edges[:-1] + edges[1:]) / 2
    real_density, _ = np.histogram(real, bins=edges, density=True)
    baseline_density, _ = np.histogram(baseline, bins=edges, density=True)
    draw_density = np.vstack(
        [np.histogram(values, bins=edges, density=True)[0] for values in draws]
    )
    return pd.DataFrame(
        {
            "fitness": centres,
            "measured": real_density,
            "no_pairs": baseline_density,
            "mutascape_median": np.median(draw_density, axis=0),
            "mutascape_lower": np.quantile(draw_density, 0.05, axis=0),
            "mutascape_upper": np.quantile(draw_density, 0.95, axis=0),
        }
    )


def _spectrum_table(
    real: np.ndarray,
    baseline: np.ndarray,
    draws: np.ndarray,
    cell_indices: np.ndarray,
    n_positions: int,
) -> pd.DataFrame:
    _, real_spectrum = binary_cube_spectrum(real, cell_indices, n_positions)
    _, baseline_spectrum = binary_cube_spectrum(baseline, cell_indices, n_positions)
    draw_spectra = [binary_cube_spectrum(values, cell_indices, n_positions)[1] for values in draws]
    rows = []
    for order in range(1, n_positions + 1):
        values = np.asarray([spectrum[order] for spectrum in draw_spectra])
        rows.append(
            {
                "order": order,
                "measured": real_spectrum[order],
                "no_pairs": baseline_spectrum[order],
                "mutascape_median": float(np.median(values)),
                "mutascape_lower": float(np.quantile(values, 0.05)),
                "mutascape_upper": float(np.quantile(values, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _plot(histogram: pd.DataFrame, spectrum: pd.DataFrame, output: Path) -> None:
    colour = "#6d5bd0"
    measured = "#17151c"
    additive = "#8b8793"
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    axes[0].plot(histogram["fitness"], histogram["measured"], color=measured, label="Measured")
    axes[0].plot(
        histogram["fitness"],
        histogram["mutascape_median"],
        color=colour,
        label="Mutascape",
    )
    axes[0].fill_between(
        histogram["fitness"],
        histogram["mutascape_lower"],
        histogram["mutascape_upper"],
        color=colour,
        alpha=0.18,
        linewidth=0,
    )
    axes[0].plot(
        histogram["fitness"],
        histogram["no_pairs"],
        color=additive,
        linestyle="--",
        label="No pair interactions",
    )
    axes[0].set(xlabel="Standardised fitness", ylabel="Density", title="Fitness distribution")
    axes[0].legend(frameon=False)

    orders = spectrum["order"].to_numpy()
    lower = spectrum["mutascape_median"] - spectrum["mutascape_lower"]
    upper = spectrum["mutascape_upper"] - spectrum["mutascape_median"]
    axes[1].bar(orders - 0.2, spectrum["measured"], width=0.4, color=measured, label="Measured")
    axes[1].bar(
        orders + 0.2,
        spectrum["mutascape_median"],
        width=0.4,
        color=colour,
        label="Mutascape",
        yerr=np.vstack([lower, upper]),
        error_kw={"elinewidth": 0.8, "capsize": 1.5},
    )
    axes[1].scatter(
        orders,
        spectrum["no_pairs"],
        marker="_",
        color=additive,
        label="No pair interactions",
    )
    axes[1].set(
        xlabel="Interaction order",
        ylabel="Fitness variance fraction",
        title="Walsh spectrum",
        xticks=orders,
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    figure.savefig(output / "creilov_shape_validation.pdf")
    figure.savefig(output / "creilov_shape_validation.png", dpi=180)


def _full_dms_observations(path: Path, target, library_singles: pd.DataFrame) -> pd.DataFrame:
    dms = pd.read_csv(path)
    required = {"position", "wildtype", "mutant", "fitness"}
    if not required.issubset(dms.columns):
        raise ValueError(f"{path} needs position, wildtype, mutant, and fitness columns")

    rows = [{"mutations": {}, "fitness": 0.0, "depth": 0}]
    seen: set[tuple[int, str]] = set()
    for position, wild_type, mutant, fitness in dms[
        ["position", "wildtype", "mutant", "fitness"]
    ].itertuples(index=False, name=None):
        position = int(position)
        if target.wild_type_sequence[position - 1] != str(wild_type):
            raise ValueError(f"DMS wild type does not match the CreiLOV sequence at {position}")
        mutation = (position, str(mutant))
        if mutation in seen:
            raise ValueError(f"DMS contains a duplicate single at {position}{mutant}")
        seen.add(mutation)
        rows.append({"mutations": {position: str(mutant)}, "fitness": float(fitness), "depth": 1})

    target_wt = float(library_singles.loc[library_singles["depth"] == 0, "fitness"].iloc[0])
    for mutations, fitness in library_singles.loc[
        library_singles["depth"] == 1, ["mutations", "fitness"]
    ].itertuples(index=False, name=None):
        mutation = next(iter(mutations.items()))
        if mutation not in seen:
            rows.append(
                {
                    "mutations": dict(mutations),
                    "fitness": float(fitness) - target_wt,
                    "depth": 1,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    if args.draws < 1:
        raise SystemExit("draws must be positive")
    if args.coupling < 0:
        raise SystemExit("coupling must be non-negative")

    ssmula = load_ssmula_landscapes(args.ssmula_root, strict=False)
    combingym = load_combingym_landscapes(args.combingym_root, strict=False)
    target = next(dataset for dataset in combingym if dataset.name == "CreiLOV_fluorescence")
    training = [dataset for dataset in [*ssmula, *combingym] if dataset.protein != target.protein]
    prior = LandscapePrior.fit(training, max_order=3)

    split = mutation_order_split(target.variants, max_train_depth=1)
    cube = supported_cube(target, split.train)
    measured, coverage = measured_tensor(target, cube)
    mask = np.isfinite(measured)
    real_values = measured[mask]
    cell_indices = _binary_indices(mask, cube)
    observations = _full_dms_observations(args.dms_csv, target, split.train)
    posterior = PosteriorLandscape.condition(target.wild_type_sequence, observations, prior)
    baseline_values = dms_pairwise_readout_tensor(
        cube,
        posterior,
        coupling=0.0,
        seed=args.seed,
    )[mask]

    tensors = []
    rows = [
        _row(
            "no_pairs",
            -1,
            real_values,
            baseline_values,
            cell_indices,
            len(cube.positions),
        )
    ]
    for draw in range(args.draws):
        tensor = dms_pairwise_readout_tensor(
            cube,
            posterior,
            coupling=args.coupling,
            seed=args.seed + draw,
        )
        values = tensor[mask]
        tensors.append(values)
        rows.append(
            _row(
                "mutascape",
                draw,
                real_values,
                values,
                cell_indices,
                len(cube.positions),
            )
        )
        print(f"draw {draw + 1}/{args.draws}", flush=True)

    draws = np.vstack(tensors)
    metrics = pd.DataFrame(rows)
    raw_histogram = _histogram_table(real_values, baseline_values, draws)
    histogram = _histogram_table(
        _standardise(real_values),
        _standardise(baseline_values),
        np.vstack([_standardise(values) for values in draws]),
    )
    spectrum = _spectrum_table(
        real_values, baseline_values, draws, cell_indices, len(cube.positions)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "creilov_shape_metrics.csv", index=False)
    raw_histogram.to_csv(args.output / "creilov_fitness_histogram_raw.csv", index=False)
    histogram.to_csv(args.output / "creilov_fitness_histogram_standardised.csv", index=False)
    spectrum.to_csv(args.output / "creilov_walsh_spectrum.csv", index=False)
    _plot(histogram, spectrum, args.output)

    mutascape = metrics.loc[metrics["model"] == "mutascape"].drop(columns=["model", "draw"])
    summary = {
        "dataset": target.name,
        "finite_variants": int(mask.sum()),
        "dms_anchors": len(observations) - 1,
        "supported_domain": cube.size,
        "coverage": coverage,
        "draws": args.draws,
        "coupling": args.coupling,
        "metrics": {
            column: {
                "median": float(mutascape[column].median()),
                "lower": float(mutascape[column].quantile(0.05)),
                "upper": float(mutascape[column].quantile(0.95)),
            }
            for column in mutascape
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(mutascape.agg(["median", "min", "max"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
