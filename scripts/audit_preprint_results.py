"""Recalculate manuscript summaries and audit complete landscape spectra."""

import argparse
import hashlib
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.generators import dms_pairwise_readout_tensor
from mutascape.benchmarks.splits import mutation_order_split
from mutascape.benchmarks.topology import measured_tensor, supported_cube
from mutascape.benchmarks.validity import (
    measured_rank_reliability,
    pairwise_winner_transfer,
    reliability_adjusted_transfer,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape
from mutascape.validation.fourier import (
    aggregate_binary_cube,
    binary_walsh_coefficients,
    index_orders,
    walsh_decompose,
    walsh_l1_distance,
)
from scripts.freeze_benchmark_validity import METHODS, _arm_metrics
from scripts.run_creilov_shape_validation import _binary_indices, _full_dms_observations
from scripts.run_search_transfer import _transfer, _with_ensemble_arms


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(actual: pd.DataFrame, saved: pd.DataFrame, keys: list[str]) -> None:
    actual = actual.sort_values(keys).reset_index(drop=True)
    saved = saved.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual[saved.columns], saved, check_dtype=False, atol=1e-12, rtol=1e-10
    )


def audit_search(archive: Path, output: Path) -> None:
    rows, counts, provenance, variation = [], [], [], []
    frozen = json.loads(
        Path("src/mutascape/data/validation/benchmark-validity-v1.json").read_text()
    )
    for collection in ("ssmula", "graphfla", "combingym"):
        for index, budget in enumerate((48, 96, 192, 384)):
            if collection in {"combingym", "graphfla"}:
                root = output / f"{collection}_b{budget}"
            else:
                root = archive / f"output_validity_b{budget}_primary"
            scores = pd.read_csv(root / "scores.csv", float_precision="round_trip")
            transfer = _transfer(scores)
            combined = _with_ensemble_arms(scores)
            pairs = pairwise_winner_transfer(combined, seed=20260823)
            reliability = measured_rank_reliability(combined, seed=20260823)
            adjusted = reliability_adjusted_transfer(transfer, reliability)
            for name, actual, keys in [
                ("transfer", transfer, ["dataset", "arm"]),
                ("pairwise_transfer", pairs, ["dataset", "arm"]),
                ("measured_reliability", reliability, ["dataset"]),
                ("reliability_adjusted_transfer", adjusted, ["dataset", "arm"]),
            ]:
                _compare(actual, pd.read_csv(root / f"{name}.csv"), keys)
            print(f"Replayed {collection} at {budget} queries", flush=True)
            protein_map = scores[["dataset", "protein"]].drop_duplicates()
            for identifier, label in METHODS:
                metrics = _arm_metrics(root, identifier)
                selected = pairs.loc[pairs.arm == identifier].merge(protein_map, on="dataset")
                rank_values = transfer.loc[transfer.arm == identifier, "kendall_tau"]
                rows.append(
                    {
                        "collection": collection,
                        "budget": budget,
                        "model": label,
                        **metrics,
                        "eligible_pairs": int(selected.scored_pairs.sum()),
                        "matching_pairs": int(selected.correct_pairs.sum()),
                        "defined_rank_correlations": int(rank_values.notna().sum()),
                        "landscapes": int(scores.dataset.nunique()),
                    }
                )
                for record in selected.to_dict("records"):
                    counts.append(
                        {"collection": collection, "budget": budget, "model": label, **record}
                    )
                if label != "Additive":
                    curve = next(
                        s for s in frozen["budgetCurve"]["series"] if s["id"] == identifier
                    )
                    np.testing.assert_allclose(
                        metrics["kendallTau"], curve[f"{collection}Kendall"][index], atol=1e-6
                    )
                    np.testing.assert_allclose(
                        metrics["pairwiseAccuracy"], curve[f"{collection}Winners"][index], atol=1e-6
                    )
                    prefix = identifier.removesuffix("mean")
                    for draw in range(3):
                        variation.append(
                            {
                                "collection": collection,
                                "budget": budget,
                                "model": label,
                                "draw": draw,
                                **_arm_metrics(root, f"{prefix}{draw}"),
                            }
                        )
            provenance.append(
                {
                    "collection": collection,
                    "budget": budget,
                    "directory": str(root),
                    "scores_sha256": _hash(root / "scores.csv"),
                    "summary_sha256": _hash(root / "summary.json"),
                    "bootstrap_seed_verified_by_replay": 20260823,
                }
            )
    pd.DataFrame(rows).to_csv(output / "search_summary.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "eligible_pairs.csv", index=False)
    pd.DataFrame(variation).to_csv(output / "draw_sensitivity.csv", index=False)
    (output / "search_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def _spectrum(tensor: np.ndarray) -> dict[int, float]:
    if all(size == 2 for size in tensor.shape):
        coefficients = binary_walsh_coefficients(tensor).ravel()
        order = index_orders(tensor.ndim)
        energy = coefficients**2
        return {
            k: float(energy[order == k].sum() / energy[order > 0].sum())
            for k in range(1, tensor.ndim + 1)
        }
    return walsh_decompose(tensor)


def audit_spectra(output: Path) -> None:
    datasets = [
        *load_ssmula_landscapes(Path(".mutascape/benchmark-data/ssmula"), strict=False),
        *load_combingym_landscapes(Path(".mutascape/benchmark-data/combingym"), strict=False),
    ]
    inventory = []
    for dataset in datasets:
        split = mutation_order_split(dataset.variants, max_train_depth=1)
        cube = supported_cube(dataset, split.train)
        tensor, coverage = measured_tensor(dataset, cube)
        inventory.append(
            {
                "dataset": dataset.name,
                "protein": dataset.protein,
                "source": dataset.spec.source,
                "measured_variants": len(dataset.variants),
                "positions": len(cube.positions),
                "supported_domain": cube.size,
                "finite_candidates": int(np.isfinite(tensor).sum()),
                "coverage": coverage,
            }
        )
        if dataset.name == "CreiLOV_fluorescence":
            mask = np.isfinite(tensor)
            indices = _binary_indices(mask, cube)
            counts = np.bincount(indices, minlength=2 ** len(cube.positions))
            projected = aggregate_binary_cube(tensor[mask], indices, len(cube.positions))
            projection = {
                "positions": len(cube.positions),
                "patterns": int(counts.size),
                "observed_patterns": int(np.count_nonzero(counts)),
                "missing_patterns": int((counts == 0).sum()),
                "variants_per_observed_pattern_min": int(counts[counts > 0].min()),
                "variants_per_pattern_max": int(counts.max()),
                "pattern_coverage": float(np.isfinite(projected).mean()),
            }
            (output / "creilov_projection.json").write_text(json.dumps(projection, indent=2) + "\n")
            # Select the largest complete subcube using coverage alone, before reading fitness.
            fixed = None
            for count in range(1, len(cube.positions)):
                for axes in combinations(range(len(cube.positions)), count):
                    for states in product((0, 1), repeat=count):
                        key = [slice(None)] * len(cube.positions)
                        for axis, state in zip(axes, states):
                            key[axis] = state
                        if np.isfinite(projected[tuple(key)]).all():
                            fixed = (axes, states, tuple(key))
                            break
                    if fixed is not None:
                        break
                if fixed is not None:
                    break
            if fixed is None:
                raise ValueError("No complete CreiLOV subcube")
            axes, states, key = fixed
            projection["complete_subcube_patterns"] = int(projected[key].size)
            projection["fixed_positions"] = [list(reversed(cube.positions))[axis] for axis in axes]
            projection["fixed_mutation_states"] = list(states)
            (output / "creilov_projection.json").write_text(json.dumps(projection, indent=2) + "\n")
            prior = LandscapePrior.fit(
                [d for d in datasets if d.protein != dataset.protein], max_order=3
            )
            observations = _full_dms_observations(
                Path("data/creilov/creilov.csv"), dataset, split.train
            )
            posterior = PosteriorLandscape.condition(
                dataset.wild_type_sequence, observations, prior
            )
            real_spectrum = _spectrum(projected[key])
            sensitivity = []
            for label, coupling, draws in [("no_pairs", 0.0, 1), ("mutascape", 0.1, 30)]:
                for draw in range(draws):
                    values = dms_pairwise_readout_tensor(
                        cube, posterior, coupling=coupling, seed=20260826 + draw
                    )[mask]
                    binary = aggregate_binary_cube(values, indices, len(cube.positions))
                    sensitivity.append(
                        {
                            "model": label,
                            "draw": draw,
                            "walsh_l1": walsh_l1_distance(real_spectrum, _spectrum(binary[key])),
                        }
                    )
            pd.DataFrame(sensitivity).to_csv(output / "creilov_complete_subcube.csv", index=False)
    pd.DataFrame(inventory).to_csv(output / "dataset_inventory.csv", index=False)

    rows, spectra = [], []
    # These are descriptive checks on existing development data, not confirmation proteins.
    targets = [
        dataset
        for dataset in datasets
        if dataset.name == "DHFR"
        or (dataset.protein == "eqFP611" and "red" in dataset.name.lower())
    ]
    if len(targets) != 2:
        raise ValueError(f"Expected DHFR and eqFP611 red, found {[d.name for d in targets]}")
    for dataset in targets:
        train = mutation_order_split(dataset.variants, max_train_depth=1).train
        cube = supported_cube(dataset, train)
        measured, coverage = measured_tensor(dataset, cube)
        if coverage != 1.0:
            raise ValueError(f"{dataset.name} is incomplete")
        prior = LandscapePrior.fit(
            [d for d in datasets if d.protein != dataset.protein], max_order=3
        )
        posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, train, prior)
        real_spectrum = _spectrum(measured)
        for label, coupling, draws in [("no_pairs", 0.0, 1), ("mutascape", 0.1, 30)]:
            for draw in range(draws):
                values = dms_pairwise_readout_tensor(
                    cube, posterior, coupling=coupling, seed=20260826 + draw
                )
                spectrum = _spectrum(values)
                rows.append(
                    {
                        "dataset": dataset.name,
                        "model": label,
                        "draw": draw,
                        "coverage": coverage,
                        "walsh_l1": walsh_l1_distance(real_spectrum, spectrum),
                    }
                )
                for order, value in spectrum.items():
                    spectra.append(
                        {
                            "dataset": dataset.name,
                            "model": label,
                            "draw": draw,
                            "order": order,
                            "measured": real_spectrum[order],
                            "synthetic": value,
                        }
                    )
        print(f"Compared complete spectra for {dataset.name}", flush=True)
    pd.DataFrame(rows).to_csv(output / "complete_spectrum_metrics.csv", index=False)
    pd.DataFrame(spectra).to_csv(output / "complete_spectra.csv", index=False)


def audit_imputation(output: Path) -> None:
    root = Path("results/creilov_imputation")
    predictions = pd.read_csv(root / "held_out_predictions.csv")
    rows = []
    for (split, method), group in predictions.groupby(["split", "method"]):
        error = group.prediction - group.measured
        rows.append(
            {
                "split": split,
                "method": method,
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "spearman": float(spearmanr(group.measured, group.prediction).statistic),
            }
        )
    actual = pd.DataFrame(rows)
    _compare(actual, pd.read_csv(root / "metrics.csv"), ["split", "method"])
    actual.to_csv(output / "imputation_replay.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(".mutascape/worktrees/posterior-landscapes-v1/notebooks/iclr"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/preprint_audit_20260923"))
    parser.add_argument("--spectra-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.spectra_only:
        audit_search(args.archive, args.output)
        audit_imputation(args.output)
    audit_spectra(args.output)


if __name__ == "__main__":
    main()
