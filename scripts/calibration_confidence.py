"""Run fresh paired search campaigns with the frozen manuscript settings."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config

from scripts import manuscript_v2 as experiment

SOURCE = Path("results/manuscript_v2_20261015")
OUTPUT = Path("results/calibration_confidence_20261119")
SAMPLES = 40
ROOT_SEED = 2026111901


def fresh_seed(role: int, protein: str, *parts: int) -> int:
    return int(
        np.random.SeedSequence(
            [ROOT_SEED, role, experiment.PROTEINS.index(protein), *parts]
        ).generate_state(1)[0]
    )


def landscape_values(name: str, model: str, c: int, landscape: int) -> np.ndarray:
    return np.load(OUTPUT / name / "landscapes" / f"{model}_{c}_{landscape}.npy")


def configure() -> None:
    experiment.OUT = OUTPUT
    experiment.seed_for = fresh_seed  # ty: ignore[invalid-assignment]
    experiment.landscape_values = landscape_values  # ty: ignore[invalid-assignment]


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "selected.json", OUTPUT / "selected.json")
    manifest = {"seed": ROOT_SEED, "samples": SAMPLES, "source_files": {}}
    for name in experiment.PROTEINS:
        destination = OUTPUT / name
        destination.mkdir(exist_ok=True)
        for filename in ("frame.pkl", "metadata.json"):
            source = SOURCE / name / filename
            shutil.copy2(source, destination / filename)
            manifest["source_files"][str(source)] = experiment.digest(source)
        source = SOURCE / name / "design.npz"
        manifest["source_files"][str(source)] = experiment.digest(source)
        with np.load(source) as loaded:
            design = dict(loaded)
        if name in experiment.EVALUATION:
            import pandas as pd

            frame = pd.read_pickle(destination / "frame.pkl")
            eligible = np.setdiff1d(
                np.flatnonzero(frame.depth.to_numpy() > 1), design["cubes"].ravel()
            )
            design["calibration"] = np.array(
                [
                    np.sort(
                        np.random.default_rng(fresh_seed(2, name, c)).choice(
                            eligible, experiment.CALIBRATION_SIZE, replace=False
                        )
                    )
                    for c in range(SAMPLES)
                ]
            )
        else:
            shutil.copytree(SOURCE / name / "search", destination / "search", dirs_exist_ok=True)
        np.savez(destination / "design.npz", **design)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    provenance = OUTPUT / "provenance.json"
    if not provenance.exists():
        files = [
            Path(__file__),
            Path(experiment.__file__),
            Path("uv.lock"),
            Path("src/mutascape/calibrated.py"),
            Path("src/mutascape/model.py"),
            Path("src/mutascape/benchmarks/methods.py"),
            Path("src/mutascape/benchmarks/validity.py"),
            SOURCE / "selected.json",
        ]
        files += [
            p
            for name in experiment.MODEL_SELECTION
            for p in (SOURCE / name / "search").glob("*.csv")
        ]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        experiment.save_json(
            provenance,
            {
                "commit": commit,
                "sha256": {str(p): experiment.digest(p) for p in files},
                "seed": ROOT_SEED,
            },
        )


def task(name: str, c: int, model: str, landscape: int) -> None:
    configure()
    started = time.monotonic()
    if model != "measured":
        folder = OUTPUT / name / "landscapes"
        folder.mkdir(exist_ok=True)
        path = folder / f"{model}_{c}_{landscape}.npy"
        if not path.exists():
            frame, _, _ = experiment.data(name)
            seed = fresh_seed(5, name, c, landscape)
            if model == "calibrated":
                fitted = experiment.calibrated_model(name, c)
                experiment.save_json(folder / f"fit_{c}.json", fitted.to_dict())
                fitted = fitted.draw(seed)
            else:
                fitted = experiment.singles_model(
                    name, experiment.singles_configs(name)[model], seed, False
                )
            np.save(path, experiment.evaluate_all(fitted, frame))
    experiment.search_task(name, c, model, landscape)
    print(f"Elapsed {name} {c} {model} {landscape} {time.monotonic() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "summarise"])
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    configure()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        tasks = [
            (name, c, model, landscape)
            for c in range(SAMPLES)
            for name in experiment.EVALUATION
            for model in ("measured", "additive", "calibrated", "sigmoid")
            for landscape in range(
                experiment.LANDSCAPES if model in ("calibrated", "sigmoid") else 1
            )
        ]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            Parallel(n_jobs=args.jobs)(delayed(task)(*item) for item in tasks)
    else:
        summary = experiment.search_summary()
        experiment.aggregate_search_summary(summary).to_csv(OUTPUT / "search_table.csv")


if __name__ == "__main__":
    main()
