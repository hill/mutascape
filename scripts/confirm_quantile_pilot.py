"""Confirm distribution calibration on fresh genotypes and a further protein."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mutascape.benchmarks import roots
from mutascape.benchmarks.datasets import SSMULA_MANIFEST, load_landscape
from mutascape.calibrated import QuantileCalibratedLandscape, empirical_quantiles
from mutascape.validation.fourier import binary_cube_spectral_metrics, walsh_l1_distance
from scripts.audit_preprint_results import _spectrum
from scripts.compare_c2st_alternatives import load_data
from scripts.diagnose_pair_pilot_transfer import strict_peaks
from scripts.explore_copula_landscapes import cube_codes, score
from scripts.test_pair_pilot_transfer import OUT as PREVIOUS
from scripts.test_pair_pilot_transfer import load_protein

OUT = Path("results/quantile_confirmation_20260923")
PROTEINS = ("DHFR", "eqFP611", "ParB-Noc", "ParD2", "CreiLOV")
SEEDS = (2026093001, 2026093002, 2026093003)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    if name == "CreiLOV":
        frame, singles, _, sequence, metadata = load_data()
        positions = sorted({p for m in singles.mutations for p in m})
        alphabets = {
            p: sorted(
                {sequence[p - 1]} | {aa for m in singles.mutations for q, aa in m.items() if p == q}
            )
            for p in positions
        }
        codes = np.array(
            [
                [alphabets[p].index(m.get(p, sequence[p - 1])) for p in positions]
                for m in frame.mutations
            ],
            dtype=np.int16,
        )
        return (
            frame,
            singles,
            sequence,
            codes,
            tuple(len(alphabets[p]) for p in positions),
            Path(metadata["raw_path"]),
        )
    if name == "ParD2":
        spec = next(s for s in SSMULA_MANIFEST if s.name == name)
        source = roots.ssmula() / spec.relative_path
        dataset = load_landscape(str(roots.ssmula()), spec)
    else:
        dataset, source = load_protein(name)
    frame = dataset.variants.reset_index(drop=True)
    codes, shape = cube_codes(dataset)
    return frame, frame.loc[frame.depth <= 1], dataset.wild_type_sequence, codes, shape, source


def freeze():
    OUT.mkdir(exist_ok=True)
    protocol = {
        "status": "Confirmation of a model selected on the previous split-zero development data.",
        "proteins": PROTEINS,
        "calibration_sizes": [48, 192],
        "test_size": 1000,
        "selection_seed": 2026093000,
        "draw_seeds": SEEDS,
        "models": ["reference", "quantile", "marginal_only"],
        "selection": (
            "Exclude all six previous transfer genotype sets. Reserve one new "
            "calibration set and one new test set per protein. ParD2 did not select "
            "the model."
        ),
        "development_repairs": (
            "Linear quantile tails avoid artificial extreme plateaux. Repeated "
            "calibration values retain their full empirical probability mass."
        ),
        "primary": (
            "Compare paired classifier scores on the three previously failing "
            "proteins. Report both classifier feature sets and all other outcomes."
        ),
        "decision": (
            "Distribution improvement needs lower boosted AUC on both feature sets "
            "for each of the three transfer proteins at 192 combinations. Benchmark "
            "adoption also needs non-decreasing macro winner agreement and Kendall "
            "correlation, with at least half of protein-budget comparisons not "
            "declining."
        ),
        "spectra": (
            "Full categorical spectra for complete libraries. CreiLOV uses the "
            "existing binary projection. ParD2 has missing genotypes and receives "
            "no filled-spectrum claim."
        ),
        "search": {
            "query_budgets": [48, 96],
            "calibration_sizes": [48, 192],
            "seeds": 10,
            "methods": "All 12 existing methods",
            "models": ["measured", "reference", "quantile"],
            "domain": "Measured combinations excluding the calibration variants",
            "cost": "Calibration combinations plus search queries, beyond measured WT and singles",
        },
        "sources": [],
        "code_sha256": {},
    }
    for name in PROTEINS:
        frame, singles, _, _, _, source = load(name)
        excluded = set()
        if name in ("DHFR", "eqFP611", "ParB-Noc"):
            for p in (PREVIOUS / name).glob("*.csv"):
                if p.name.startswith(("pilot_", "test_")):
                    excluded.update(pd.read_csv(p).row)
        elif name == "CreiLOV":
            variants = set()
            files = list(Path("results").glob("creilov*/samples_*.csv")) + list(
                Path("results").glob("creilov*/pilot_*.csv")
            )
            files += list(Path("results/c2st_alternatives_20260923").glob("pilot_*.csv")) + list(
                Path("results/c2st_alternatives_20260923").glob("test_*.csv")
            )
            for p in files:
                data = pd.read_csv(p)
                if "variant" in data:
                    variants.update(data.variant)
            excluded.update(frame.index[frame.variant.isin(variants)])
        pool = frame.loc[(frame.depth > 1) & ~frame.index.isin(excluded)]
        selected = pool.sample(n=1192, random_state=2026093000)
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        pd.DataFrame({"row": selected.index[:192]}).to_csv(folder / "pilot.csv", index=False)
        pd.DataFrame({"row": selected.index[192:]}).to_csv(folder / "test.csv", index=False)
        frame.to_csv(folder / "genotypes.csv", index=False)
        protocol["sources"].append(
            dict(
                protein=name,
                path=str(source),
                sha256=digest(source),
                rows=len(frame),
                excluded_rows=len(excluded),
                remaining_pool=len(pool),
                genotype_sha256=digest(folder / "genotypes.csv"),
            )
        )
    for p in [
        Path(__file__),
        Path("src/mutascape/calibrated.py"),
        Path("scripts/explore_copula_landscapes.py"),
        Path("scripts/run_quantile_search.py"),
        Path("src/mutascape/benchmarks/methods.py"),
    ]:
        protocol["code_sha256"][str(p)] = digest(p)
        (OUT / (str(p).replace("/", "__") + ".txt")).write_bytes(p.read_bytes())
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def run(name, size):
    folder = OUT / name / f"p{size}"
    if (folder / "done.json").exists():
        return
    folder.mkdir(exist_ok=True)
    frame, singles, sequence, codes, shape, _ = load(name)
    pilots = pd.read_csv(OUT / name / "pilot.csv").row.to_numpy()[:size]
    tests = pd.read_csv(OUT / name / "test.csv").row.to_numpy()
    model = QuantileCalibratedLandscape.fit(sequence, singles, frame.iloc[pilots])
    (folder / "model.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
    eligible = (frame.depth > 1).to_numpy()
    flat = np.ravel_multi_index(codes.T, shape)
    complete = len(frame) == int(np.prod(shape))
    real_tensor = frame.fitness.to_numpy()[np.argsort(flat)].reshape(shape) if complete else None
    real_spectrum = _spectrum(real_tensor) if complete else None
    rows, structure = [], []
    mutations = frame.mutations.tolist()
    for seed in SEEDS:
        draw = model.draw(seed, reference_size=None)
        reference = draw.base_values([tuple(sorted(m.items())) for m in mutations])
        quantile = draw.evaluate_batch(mutations)
        rng = np.random.default_rng(seed + 177)
        marginal = empirical_quantiles(model.pilot_fitness, rng.random(int(np.prod(shape)))[flat])
        predictions = {"reference": reference, "quantile": quantile, "marginal_only": marginal}
        for label, values in predictions.items():
            values[singles.index] = singles.fitness
            assert np.isfinite(values).all()
            if label == "quantile":
                np.testing.assert_allclose(
                    values[tests[:10]],
                    draw.evaluate_batch(frame.iloc[tests[:10]].mutations.tolist()),
                    atol=1e-10,
                )
                ordered = np.argsort(reference[eligible])
                assert np.diff(values[eligible][ordered]).min() >= -1e-9
            np.save(folder / f"{label}_{seed}.npy", values)
            rows += score(
                frame,
                singles,
                tests,
                values,
                codes,
                dict(protein=name, pilot_size=size, model=label, draw=seed),
            )
            if complete:
                tensor = values[np.argsort(flat)].reshape(shape)
                structure.append(
                    dict(
                        protein=name,
                        pilot_size=size,
                        model=label,
                        draw=seed,
                        walsh_l1=walsh_l1_distance(real_spectrum, _spectrum(tensor)),
                        peaks=strict_peaks(tensor),
                        measured_peaks=strict_peaks(real_tensor),
                    )
                )
            elif name == "CreiLOV":
                positions = sorted({p for m in singles.mutations for p in m})
                binary = np.array(
                    [sum(1 << i for i, p in enumerate(positions) if p in m) for m in mutations]
                )
                structure.append(
                    dict(
                        protein=name,
                        pilot_size=size,
                        model=label,
                        draw=seed,
                        **binary_cube_spectral_metrics(
                            frame.fitness.to_numpy(), values, binary, len(positions)
                        ),
                    )
                )
            pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
            pd.DataFrame(structure).to_csv(folder / "structure.csv", index=False)
            print(f"Scored {name} pilot {size} {label} {seed}", flush=True)
    (folder / "done.json").write_text(
        json.dumps(
            {
                "complete_library": complete,
                "pilot_size": size,
                "test_size": len(tests),
                "median_reference_w1": float(
                    np.median([r["w1"] for r in rows if r["model"] == "reference"])
                ),
            }
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.freeze:
        if (OUT / "protocol.json").exists():
            raise ValueError("Confirmation protocol already exists")
        freeze()
    protocol = json.loads((OUT / "protocol.json").read_text())
    for p, checksum in protocol["code_sha256"].items():
        assert digest(p) == checksum, p
    Parallel(n_jobs=args.jobs)(delayed(run)(name, size) for size in (48, 192) for name in PROTEINS)


if __name__ == "__main__":
    main()
