"""Describe interaction coverage, local peaks, and held-out upper-tail agreement."""

import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from mutascape.benchmarks.topology import supported_cube
from mutascape.validation.fourier import (
    binary_walsh_coefficients,
    index_orders,
    walsh_signed_effects,
)
from scripts.compare_c2st_alternatives import basis
from scripts.test_pair_pilot_transfer import DRAW_SEEDS, OUT, PROTEINS, load_protein


def strict_peaks(tensor):
    peaks = np.ones(tensor.shape, dtype=bool)
    for axis in range(tensor.ndim):
        highest = np.max(tensor, axis=axis, keepdims=True)
        second = np.take(np.partition(tensor, -2, axis=axis), [-2], axis=axis)
        neighbour_maximum = np.where(tensor == highest, second, highest)
        peaks &= tensor > neighbour_maximum
    return int(peaks.sum())


def signed(tensor):
    if all(size == 2 for size in tensor.shape):
        return binary_walsh_coefficients(tensor).ravel(), index_orders(tensor.ndim)
    return walsh_signed_effects(tensor)


def main():
    rows, coverage, distribution, coefficients_summary = [], [], [], []
    for name in PROTEINS:
        dataset, _ = load_protein(name)
        frame = dataset.variants.reset_index(drop=True)
        singles = frame.loc[frame.depth <= 1]
        cube = supported_cube(dataset, singles)
        shape = tuple(len(cube.alphabets[p]) for p in cube.positions)
        codes = np.array(
            [
                [cube.alphabets[p].index(m.get(p, cube.wild_type[p])) for p in cube.positions]
                for m in frame.mutations
            ]
        )
        indices = np.ravel_multi_index(codes.T, shape)
        inverse = np.empty(len(frame), dtype=int)
        inverse[indices] = np.arange(len(frame))
        real = frame.fitness.to_numpy()
        tensor = real[inverse].reshape(shape)
        real_coefficients, orders = signed(tensor)
        real_peaks = strict_peaks(tensor)
        # Fixed uniformly sampled directed mutation edges, shared by all models.
        rng = np.random.default_rng(2026092800)
        left_rows = rng.integers(len(frame), size=10000)
        neighbour_codes = codes[left_rows].copy()
        axes = rng.integers(len(shape), size=len(left_rows))
        for i, axis in enumerate(axes):
            old = neighbour_codes[i, axis]
            replacement = rng.integers(shape[axis] - 1)
            neighbour_codes[i, axis] = replacement + (replacement >= old)
        right_rows = inverse[np.ravel_multi_index(neighbour_codes.T, shape)]
        real_delta = real[right_rows] - real[left_rows]
        keys = sorted(k for m in singles.mutations for k in m.items())
        distribution.append(
            dict(
                protein=name,
                measured_peaks=real_peaks,
                modal_value=float(frame.fitness.mode().iloc[0]),
                modal_fraction=float(frame.fitness.value_counts().max() / len(frame)),
                q01=float(np.quantile(real, 0.01)),
                q50=float(np.quantile(real, 0.5)),
                q99=float(np.quantile(real, 0.99)),
            )
        )
        for split in range(3):
            pilot_ids = pd.read_csv(OUT / name / f"pilot_{split}.csv").row.to_numpy()
            test_ids = pd.read_csv(OUT / name / f"test_{split}.csv").row.to_numpy()
            count = max(1, len(test_ids) // 100)
            top_real = set(np.argsort(real[test_ids])[-count:])
            for size in (48, 192):
                _, pair_design, pairs = basis(frame.iloc[pilot_ids[:size]].mutations, keys)
                coverage.append(
                    dict(
                        protein=name,
                        split=split,
                        pilot_size=size,
                        possible_pairs=len(pairs),
                        observed_pairs=int((pair_design.sum(axis=0) > 0).sum()),
                        design_rank=int(np.linalg.matrix_rank(pair_design.astype(float))),
                    )
                )
                directory = OUT / name / f"p{size}_s{split}"
                precision = json.loads((directory / "done.json").read_text())
                fitted = np.load(directory / "pair_fit.npz")
                coefficients_summary.append(
                    dict(
                        protein=name,
                        split=split,
                        pilot_size=size,
                        prior_sd=precision["coefficient_precision"] ** -0.5,
                        posterior_mean_sd=float(fitted["mean"].std()),
                        posterior_average_sd=float(
                            np.sqrt((fitted["root"] ** 2).sum(axis=1)).mean()
                        ),
                    )
                )
                model_draws = [
                    (m, d) for m in ("pilot_reference", "learned_pairs") for d in DRAW_SEEDS
                ]
                model_draws += [(m, DRAW_SEEDS[0]) for m in ("learned_mean", "additive")]
                for model, draw in model_draws:
                    predicted = np.load(directory / f"{model}_{draw}.npy")
                    predicted_tensor = predicted[inverse].reshape(shape)
                    coefficients, _ = signed(predicted_tensor)
                    x, y = real_coefficients[orders >= 2], coefficients[orders >= 2]
                    denominator = np.linalg.norm(x) * np.linalg.norm(y)
                    cosine = float(x @ y / denominator) if denominator > 1e-12 else np.nan
                    pred_delta = predicted[right_rows] - predicted[left_rows]
                    top_prediction = set(np.argsort(predicted[test_ids])[-count:])
                    rows.append(
                        dict(
                            protein=name,
                            split=split,
                            pilot_size=size,
                            model=model,
                            draw=draw,
                            peaks=strict_peaks(predicted_tensor),
                            measured_peaks=real_peaks,
                            higher_order_cosine=cosine,
                            edge_correlation=float(pearsonr(real_delta, pred_delta).statistic),
                            top_one_percent_recall=len(top_real & top_prediction) / count,
                        )
                    )
        print(f"Diagnosed {name}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "diagnostics.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "pair_coverage.csv", index=False)
    pd.DataFrame(distribution).to_csv(OUT / "measured_distributions.csv", index=False)
    pd.DataFrame(coefficients_summary).to_csv(OUT / "coefficient_diagnostics.csv", index=False)
    (OUT / "diagnostic_protocol.json").write_text(
        json.dumps(
            {
                "status": "Post-result diagnostics. No model settings changed.",
                "edge_seed": 2026092800,
                "edge_count": 10000,
                "tail": "Top one per cent within each 2000-genotype test set",
                "spectra_and_peaks": (
                    "Full-library descriptive checks, including calibration genotypes"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
