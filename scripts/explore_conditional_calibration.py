"""Compare tail preservation and conditional distribution calibration during development."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import ndtr
from sklearn.ensemble import RandomForestRegressor

from mutascape.calibrated import QuantileCalibratedLandscape, empirical_quantiles
from mutascape.validation.fourier import walsh_l1_distance
from scripts.audit_preprint_results import _spectrum
from scripts.compare_c2st_alternatives import basis, field
from scripts.explore_copula_landscapes import cube_codes, score, summary_features
from scripts.test_pair_pilot_transfer import OUT as PREVIOUS
from scripts.test_pair_pilot_transfer import PROTEINS, load_protein

OUT = Path("results/conditional_development_20260923")


def upper_tail(reference, calibrated, pilot_fitness, eligible, count=5):
    cutoff = 1 - count / len(pilot_fitness)
    threshold = float(np.quantile(reference[eligible], cutoff))
    boundary = float(empirical_quantiles(np.asarray(pilot_fitness), np.array([cutoff]))[0])
    maximum = max(float(reference[eligible].max()), boundary)
    selected = eligible & (reference > threshold)
    result = calibrated.copy()
    if reference[eligible].max() > threshold:
        result[selected] = boundary + (reference[selected] - threshold) * (maximum - boundary) / (
            reference[eligible].max() - threshold
        )
    return result


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    ordered = values[order]
    unique, start, counts = np.unique(ordered, return_index=True, return_counts=True)
    grouped = np.add.reduceat(weights[:, order], start, axis=1)
    cumulative = np.cumsum(grouped, axis=1)
    previous = cumulative - grouped
    knots = []
    ordinates = []
    for j, count in enumerate(counts):
        if count > 1:
            knots.extend([previous[:, j], cumulative[:, j]])
            ordinates.extend([unique[j], unique[j]])
        else:
            knots.append(previous[:, j] + grouped[:, j] / 2)
            ordinates.append(unique[j])
    knots = np.column_stack(knots)
    ordinates = np.asarray(ordinates)
    right = np.sum(knots < probabilities[:, None], axis=1).clip(1, len(ordinates) - 1)
    left = right - 1
    rows = np.arange(len(weights))
    fraction = (probabilities - knots[rows, left]) / np.maximum(
        knots[rows, right] - knots[rows, left], 1e-14
    )
    # Keep values inside the observed conditional range for this exploratory model.
    fraction = fraction.clip(0, 1)
    return ordinates[left] + fraction * (ordinates[right] - ordinates[left])


def conditional_draws(frame, singles, pilot_ids, sequence, seed):
    design = summary_features(frame, singles)
    y = frame.fitness.to_numpy()[pilot_ids]
    forest = RandomForestRegressor(
        n_estimators=64, min_samples_leaf=8, max_features=1.0, random_state=seed, n_jobs=1
    ).fit(design[pilot_ids], y)
    leaves = forest.apply(design)
    training = leaves[pilot_ids]
    tables = []
    for j, tree in enumerate(forest.estimators_):
        table = np.zeros((tree.tree_.node_count, len(pilot_ids)))
        for leaf in np.unique(training[:, j]):
            members = np.flatnonzero(training[:, j] == leaf)
            table[leaf, members] = 1 / len(members)
        tables.append(table)
    keys = sorted(k for m in singles.mutations for k in m.items())
    _, pairs_design, pairs = basis(frame.mutations, keys)
    normal = (
        pairs_design
        @ field(keys, pairs, sequence, seed)
        / np.sqrt(np.maximum(pairs_design.sum(axis=1), 1))
    )
    independent = np.random.default_rng(seed + 317).standard_normal(len(frame))
    probabilities = {
        "forest_pairs": ndtr(normal),
        "forest_mixed": ndtr(np.sqrt(0.75) * normal + 0.5 * independent),
        "forest_independent": ndtr(independent),
    }
    outputs = {key: np.empty(len(frame)) for key in probabilities}
    for start in range(0, len(frame), 2048):
        stop = min(start + 2048, len(frame))
        weights = sum(table[leaves[start:stop, j]] for j, table in enumerate(tables)) / len(tables)
        weights = 0.95 * weights + 0.05 / len(y)
        for key, u in probabilities.items():
            outputs[key][start:stop] = weighted_quantile(y, weights, u[start:stop])
    return outputs


def run(name):
    dataset, _ = load_protein(name)
    frame = dataset.variants.reset_index(drop=True)
    singles = frame.loc[frame.depth <= 1]
    codes, shape = cube_codes(dataset)
    flat = np.ravel_multi_index(codes.T, shape)
    inverse = np.argsort(flat)
    real_spectrum = _spectrum(frame.fitness.to_numpy()[inverse].reshape(shape))
    pilots = pd.read_csv(PREVIOUS / name / "pilot_0.csv").row.to_numpy()[:192]
    tests = pd.read_csv(PREVIOUS / name / "test_0.csv").row.to_numpy()
    seed = 2026092901
    g = QuantileCalibratedLandscape.fit(dataset.wild_type_sequence, singles, frame.iloc[pilots])
    d = g.draw(seed, reference_size=None)
    reference = d.base_values([tuple(sorted(m.items())) for m in frame.mutations])
    quantile = d.evaluate_batch(frame.mutations.tolist())
    outputs = {
        "reference": reference,
        "quantile": quantile,
        "tail_quantile": upper_tail(
            reference, quantile, frame.fitness.to_numpy()[pilots], (frame.depth > 1).to_numpy()
        ),
    }
    outputs.update(conditional_draws(frame, singles, pilots, dataset.wild_type_sequence, seed))
    rows = []
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    for model, values in outputs.items():
        values[singles.index] = singles.fitness
        np.save(folder / f"{model}.npy", values)
        metrics = score(frame, singles, tests, values, codes, dict(protein=name, model=model))
        spectrum = _spectrum(values[inverse].reshape(shape))
        for row in metrics:
            row["walsh_l1"] = walsh_l1_distance(real_spectrum, spectrum)
        rows += metrics
        pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
        print(
            name,
            model,
            [(r["auc"], r["w1"], r["spearman"], r["walsh_l1"]) for r in metrics],
            flush=True,
        )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "status": (
                    "Further development after initial quantile confirmation. "
                    "Uses original transfer split zero. No confirmation claims from these scores."
                ),
                "tail_count": 5,
                "forest_trees": 64,
                "minimum_leaf": 8,
                "global_distribution_weight": 0.05,
            },
            indent=2,
        )
        + "\n"
    )
    from scripts.explore_conditional_calibration import run as worker

    Parallel(n_jobs=3)(delayed(worker)(name) for name in PROTEINS)
