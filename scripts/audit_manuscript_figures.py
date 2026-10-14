"""Check manuscript search summaries and CreiLOV plots against saved data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape import from_dms
from mutascape.benchmarks.validity import reliable_method_pairs
from mutascape.calibrated import QuantileCalibratedLandscape
from scripts.build_singles_manuscript import (
    PILOT,
    REPEAT_SEARCH,
    RESULTS,
    ROOT,
    SEEDS,
    repeat_search_results,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum(values, cubes):
    """Compute parity coefficients independently of the figure's Hadamard routine."""
    masks = np.arange(8)
    parity = np.array([[(-1) ** int(x & s).bit_count() for s in masks] for x in masks])
    coefficients = values[cubes] @ parity / 8
    energy = (coefficients[:, 1:] ** 2).sum(axis=0)
    orders = np.array([int(s).bit_count() for s in masks[1:]])
    return np.array([energy[orders == k].sum() for k in (1, 2, 3)]) / energy.sum()


def audit(manuscript):
    manifest = json.loads((REPEAT_SEARCH / "analysis_manifest.json").read_text())
    for path, expected in manifest["trial_hashes"].items():
        assert digest(ROOT / path) == expected, path
    assert digest(ROOT / "scripts/analyse_search_selection.py") == manifest["analysis_sha256"]
    assert digest(REPEAT_SEARCH / "manifest.json") == manifest["run_manifest_sha256"]
    summary, intervals = repeat_search_results()
    for filename, expected in (
        ("singles_search_comparisons.csv", summary),
        ("singles_search_intervals.csv", intervals),
    ):
        pd.testing.assert_frame_equal(pd.read_csv(manuscript / "figures" / filename), expected)

    original = pd.read_csv(RESULTS / "T7/warm_measured_0.csv")
    original_pairs = reliable_method_pairs(
        original.query("budget == 48").rename(columns={"model": "arm"}).assign(dataset="T7"),
        seed=2026092300,
        n_bootstrap=2000,
    )
    assert original_pairs.empty

    frame = pd.read_pickle(RESULTS / "CreiLOV/frame.pkl")
    design = np.load(RESULTS / "CreiLOV/design.npz")
    meta = json.loads((RESULTS / "CreiLOV/metadata.json").read_text())
    assert digest(ROOT / meta["source"]) == meta["source_sha256"]
    published = json.loads((manuscript / "figures/creilov_shape_data.json").read_text())
    test, cubes = design["test"], design["cubes"]
    requested = np.unique(np.r_[test, cubes.ravel(), np.flatnonzero(frame.depth <= 1)])
    variants = frame.iloc[requested].mutations.tolist()
    singles = frame.loc[frame.depth <= 1]
    generator = from_dms(singles[["mutations", "fitness"]], wt_sequence=meta["sequence"])
    calibrated = QuantileCalibratedLandscape.from_dict(
        json.loads((PILOT / "CreiLOV/model.json").read_text())
    )
    calibration_rows = pd.read_csv(PILOT / "CreiLOV/pilot.csv")["row"].to_numpy()
    assert not np.intersect1d(calibration_rows, requested).size
    bins = np.array(published["bin_edges"])
    width = np.diff(bins)
    real = frame.fitness.to_numpy()
    measured_counts = np.histogram(real[test], bins=bins)[0]
    assert measured_counts.sum() == len(test) == 1000
    np.testing.assert_allclose(measured_counts / len(test) / width, published["measured_histogram"])
    np.testing.assert_allclose(spectrum(real, cubes), published["measured_spectrum"])
    records = []
    for index, seed in enumerate(SEEDS):
        for label, oracle, path in (
            (
                "Mutascape",
                generator.draw(seed=seed),
                RESULTS / "CreiLOV" / f"fitness_sigmoid_{seed}.npy",
            ),
            (
                "Combination-calibrated Mutascape",
                calibrated.draw(seed=seed),
                PILOT / "CreiLOV" / f"evaluation_{seed}.npy",
            ),
        ):
            saved = np.load(path)
            predicted = oracle.evaluate_batch(variants)
            error = float(np.max(np.abs(predicted - saved[requested])))
            np.testing.assert_allclose(predicted, saved[requested], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(
                oracle.evaluate_batch(singles.mutations.tolist()), singles.fitness
            )
            counts = np.histogram(saved[test], bins=bins)[0]
            assert counts.sum() == len(test)
            density = counts / len(test) / width
            assert np.isclose(np.sum(density * width), 1)
            np.testing.assert_allclose(density, published["histograms"][label][index])
            np.testing.assert_allclose(spectrum(saved, cubes), published["spectra"][label][index])
            records.append(
                dict(
                    model=label,
                    seed=seed,
                    max_prediction_error=error,
                    final_bin_fraction=float(counts[-1] / len(test)),
                    peak_density=float(density.max()),
                )
            )
    return dict(
        checked_trial_files=len(manifest["trial_hashes"]),
        original_t7_48_eligible_pairs=len(original_pairs),
        search_comparisons=summary.to_dict(orient="records"),
        creilov=dict(
            lower=generator.draw(seed=SEEDS[0]).readout.floor,
            upper=generator.draw(seed=SEEDS[0]).readout.ceiling,
            bin_width=float(width[-1]),
            final_bin=bins[-2:].tolist(),
            measured_final_bin_fraction=float(measured_counts[-1] / len(test)),
            checked_genotypes=len(requested),
            samples=records,
        ),
        sources={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                REPEAT_SEARCH / "analysis_manifest.json",
                REPEAT_SEARCH / "per_protein.csv",
                REPEAT_SEARCH / "pair_differences.csv",
                RESULTS / "CreiLOV/design.npz",
                PILOT / "CreiLOV/model.json",
                Path(__file__),
            )
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=ROOT.parent / "mutascape_manu_v1")
    args = parser.parse_args()
    report = audit(args.manuscript)
    path = args.manuscript / "figures/figure_audit_20260930.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {report['checked_trial_files']} search files and all plotted CreiLOV samples.")
    print(path)


if __name__ == "__main__":
    main()
