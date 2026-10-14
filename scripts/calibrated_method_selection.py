"""Run method-selection experiments across independent calibration samples."""

import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from scipy.stats import wasserstein_distance

from mutascape.benchmarks.library import BudgetedOracle, SearchSpace
from mutascape.benchmarks.methods import METHODS, SEARCH_PROTOCOL_VERSION, make_features
from mutascape.calibrated import QuantileCalibratedLandscape
from mutascape.model import DMSLandscape

OUT = Path("results/calibrated_method_selection_20261014")
SOURCE = Path("results/singles_only_restart_20260923")
PROTOCOL = Path("docs/calibrated_method_selection_protocol_2026-10-14.md")
EVALUATION = ("GB1", "TrpB4", "T7", "TEV")
TRAINING = ("DHFR", "eqFP611", "ParB-Noc", "ParD2", "CreiLOV")
PROTEINS = (*EVALUATION, *TRAINING)
MODELS = ("calibrated", "singles", "additive", "measured")
BUDGETS = (48, 96)
N_CALIBRATION = 96
N_SAMPLES = 3
N_REPEATS = 6


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")


def seed_for(kind: int, protein: str, calibration: int = 0, block: int = 0) -> int:
    """Keep random streams separate across roles and experimental units."""
    index = PROTEINS.index(protein)
    return int(
        np.random.SeedSequence([20261014, kind, index, calibration, block]).generate_state(1)[0]
    )


def load_data(protein: str) -> tuple[pd.DataFrame, dict, dict]:
    folder = SOURCE / protein
    frame = pd.read_pickle(folder / "frame.pkl")
    design = dict(np.load(folder / "design.npz"))
    order = np.lexsort(design["codes"].T[::-1])
    frame = frame.iloc[order].reset_index(drop=True)
    design = {"codes": design["codes"][order], "shape": design["shape"], "source_rows": order}
    return frame, design, json.loads((folder / "metadata.json").read_text())


def sample_calibration(depth: np.ndarray, seed: int, size: int = 96) -> np.ndarray:
    """Select only multi-mutant identities without using fitness."""
    return np.sort(
        np.random.default_rng(seed).choice(np.flatnonzero(depth > 1), size, replace=False)
    )


class KnownOracle(BudgetedOracle):
    """Keep existing observations free and count each new observation once."""

    def __init__(self, values: np.ndarray, budget: int, known: np.ndarray):
        super().__init__(values, budget)
        if len(set(known)) != len(known):
            raise ValueError("Known observations contain duplicate identities")
        self._known_count = len(known)
        self._rows = list(map(int, known))
        self._observed = list(map(float, values[known]))
        self._seen = dict(zip(self._rows, self._observed))

    @property
    def spent(self) -> int:
        return len(self._rows) - self._known_count


def condition_values(generated: np.ndarray, measured: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Use experimental values for observations already available to a campaign."""
    values = generated.copy()
    values[known] = measured[known]
    return values


def freeze() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        return verify()
    paths = [Path(__file__).relative_to(Path.cwd()), PROTOCOL]
    paths.extend(Path("src/mutascape").rglob("*.py"))
    paths.extend(
        SOURCE / p / f for p in PROTEINS for f in ("frame.pkl", "design.npz", "metadata.json")
    )
    manifest = dict(
        evaluation=EVALUATION,
        training=TRAINING,
        calibration_sets=10,
        training_calibration_sets=3,
        calibration_size=N_CALIBRATION,
        samples=N_SAMPLES,
        repeats=N_REPEATS,
        measured_repeats=18,
        methods=list(METHODS),
        models=MODELS,
        budgets=BUDGETS,
        protocol=SEARCH_PROTOCOL_VERSION,
        seed_rule="SeedSequence([20261014, role, protein_index, calibration_index, block_index])",
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            scipy=scipy.__version__,
            sklearn=sklearn.__version__,
        ),
        source_sha256={str(p): digest(p) for p in paths},
    )
    save_json(OUT / "manifest.json", manifest)
    return manifest


def verify() -> dict:
    manifest = json.loads((OUT / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen experiment input changed at {path}")
    return manifest


def prepare_protein(protein: str) -> None:
    frame, design, metadata = load_data(protein)
    folder = OUT / protein
    folder.mkdir(exist_ok=True)
    np.savez(folder / "design.npz", **design, singles=np.flatnonzero(frame.depth.to_numpy() <= 1))
    np.save(folder / "measured.npy", frame.fitness.to_numpy())
    save_json(folder / "metadata.json", metadata)


def prepare_sample(protein: str, calibration: int) -> None:
    folder = OUT / protein / str(calibration)
    if (folder / "prepared.json").exists():
        return
    folder.mkdir(exist_ok=True)
    started = time.monotonic()
    frame, design, metadata = load_data(protein)
    measured = frame.fitness.to_numpy()
    singles = np.flatnonzero(frame.depth.to_numpy() <= 1)
    rows = sample_calibration(frame.depth.to_numpy(), seed_for(1, protein, calibration))
    known = np.sort(np.r_[singles, rows])
    np.save(folder / "known.npy", known)
    np.save(folder / "calibration.npy", rows)
    frame.iloc[rows].assign(row=rows).to_csv(folder / "calibration.csv", index=False)
    diagnostics = []
    if protein in EVALUATION:
        model = QuantileCalibratedLandscape.fit(
            metadata["sequence"], frame.iloc[singles], frame.iloc[rows]
        )
        save_json(folder / "fit.json", model.to_dict())
        variants = frame.mutations.tolist()
        unseen = np.setdiff1d(np.arange(len(frame)), known)
        for label in ("calibrated", "singles", "additive"):
            for sample in range(1 if label == "additive" else N_SAMPLES):
                seed = seed_for(2, protein, calibration, sample)
                fitted = (
                    model.draw(seed)
                    if label == "calibrated"
                    else DMSLandscape.from_singles(
                        metadata["sequence"],
                        frame.iloc[singles],
                        mode="linear" if label == "additive" else "sigmoid",
                        strength=0.0 if label == "additive" else 0.3,
                        seed=seed,
                    )
                )
                generated = np.concatenate(
                    [
                        fitted.evaluate_batch(variants[j : j + 8192])
                        for j in range(0, len(frame), 8192)
                    ]
                )
                np.testing.assert_array_equal(generated[singles], measured[singles])
                assert np.isfinite(generated).all()
                values = condition_values(generated, measured, known)
                np.save(folder / f"{label}_{sample}.npy", values)
                diagnostics.append(
                    dict(
                        model=label,
                        sample=sample,
                        landscape_seed=seed,
                        initial_best=float(measured[known].max()),
                        synthetic_max=float(values.max()),
                        synthetic_headroom=float(values.max() - values[known].max()),
                        experimental_headroom=float(measured.max() - measured[known].max()),
                        calibration_rmse=float(
                            np.sqrt(np.mean((generated[rows] - measured[rows]) ** 2))
                        ),
                        w1_scaled=float(
                            wasserstein_distance(measured[unseen], generated[unseen])
                            / np.std(measured[unseen])
                        ),
                    )
                )
        pd.DataFrame(diagnostics).to_csv(folder / "diagnostics.csv", index=False)
    checksums = {str(p): digest(p) for p in folder.glob("*.npy")}
    save_json(
        folder / "prepared.json",
        dict(
            protein=protein,
            calibration=calibration,
            calibration_seed=seed_for(1, protein, calibration),
            known=len(known),
            genotypes=len(frame),
            source_sha256=checksums,
            seconds=time.monotonic() - started,
        ),
    )
    print(
        f"Prepared {protein} calibration {calibration} in {time.monotonic() - started:.1f}s",
        flush=True,
    )


def make_space(
    codes: np.ndarray, shape: np.ndarray, known: np.ndarray, values: np.ndarray
) -> SearchSpace:
    return SearchSpace(
        codes,
        tuple(shape),
        np.arange(len(codes)),
        int(known[np.argmax(values[known])]),
        {tuple(c): i for i, c in enumerate(codes)},
    )


def run(protein: str, calibration: int, model: str, block: int, smoke: bool = False) -> None:
    direct = calibration < 0
    folder = OUT / protein / ("direct" if direct else str(calibration))
    folder.mkdir(exist_ok=True)
    target = folder / f"trials_{model}_{block}{'_smoke' if smoke else ''}.csv"
    if target.exists():
        return
    started = time.monotonic()
    design = np.load(OUT / protein / "design.npz")
    known = design["singles"] if direct else np.load(folder / "known.npy")
    measured = np.load(OUT / protein / "measured.npy")
    sample = 0 if model == "additive" else block
    values = measured if model == "measured" else np.load(folder / f"{model}_{sample}.npy")
    np.testing.assert_array_equal(values[known], measured[known])
    space = make_space(design["codes"], design["shape"], known, values)
    features = make_features(space)
    records, queries = [], []
    spread = max(float(np.ptp(values)), 1e-12)
    for budget in BUDGETS:
        further = budget + N_CALIBRATION if direct else budget
        for trial in range(1 if smoke else N_REPEATS):
            role = 4 if direct else (3 if model == "measured" else 5)
            seed = seed_for(role, protein, max(calibration, 0), block * N_REPEATS + trial)
            for method, function in METHODS.items():
                oracle = KnownOracle(values, further, known)
                tick = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    function(space, oracle, features, seed)
                elapsed = time.monotonic() - tick
                new = oracle.rows[len(known) :]
                assert oracle.spent == further
                assert len(np.unique(new)) == further
                assert not np.intersect1d(new, known).size
                assert oracle.best() == values[np.r_[known, new]].max()
                queries.append(np.pad(new, (0, 192 - further), constant_values=-1))
                records.append(
                    dict(
                        protein=protein,
                        calibration=calibration,
                        model=model,
                        block=block,
                        trial=trial,
                        sample=sample,
                        budget=budget,
                        further=further,
                        total_budget=budget + N_CALIBRATION,
                        method=method,
                        search_seed=seed,
                        best=oracle.best(),
                        initial_best=float(values[known].max()),
                        regret=float((values.max() - oracle.best()) / spread),
                        spent=oracle.spent,
                        fallback_queries=oracle.fallback_queries,
                        warnings=len(caught),
                        warning_messages=" | ".join(sorted({str(w.message) for w in caught})),
                        elapsed_seconds=elapsed,
                    )
                )
    np.savez_compressed(target.with_suffix(".npz"), queries=np.asarray(queries))
    temp = target.with_suffix(".tmp")
    pd.DataFrame(records).to_csv(temp, index=False)
    temp.replace(target)
    print(
        f"Completed {protein} calibration {calibration} {model} block {block} "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )


def tasks() -> list[tuple]:
    result = []
    for protein in PROTEINS:
        for calibration in range(10 if protein in EVALUATION else 3):
            for model in MODELS if protein in EVALUATION else ("measured",):
                for block in range(3 if protein in EVALUATION else 1):
                    result.append((protein, calibration, model, block))
        result.extend((protein, -1, "measured", b) for b in range(3))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "run", "status"])
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "prepare":
        freeze()
        for protein in PROTEINS:
            prepare_protein(protein)
        jobs = [(p, c) for p in PROTEINS for c in range(10 if p in EVALUATION else 3)]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            Parallel(n_jobs=args.jobs)(delayed(prepare_sample)(*j) for j in jobs)
    elif args.stage == "smoke":
        verify()
        run("GB1", 0, "measured", 0, smoke=True)
    elif args.stage == "run":
        verify()
        for p in OUT.glob("*/*/prepared.json"):
            for path, expected in json.loads(p.read_text())["source_sha256"].items():
                assert digest(Path(path)) == expected, path
        with parallel_config(backend="loky", inner_max_num_threads=1):
            Parallel(n_jobs=args.jobs)(delayed(run)(*j) for j in tasks())
    else:
        completed = [p for p in OUT.glob("*/*/trials_*.csv") if "smoke" not in p.name]
        print(json.dumps(dict(completed=len(completed), expected=len(tasks()))))


if __name__ == "__main__":
    main()
