"""Rerun every manuscript experiment from raw data with the current generators.

Stages run in order and write only under ``OUT``. No stage reads an earlier
results folder. Each stage skips work whose output already exists, so an
interrupted run resumes where it stopped.

    uv run python -m scripts.manuscript_v2 prepare
    uv run python -m scripts.manuscript_v2 select
    uv run python -m scripts.manuscript_v2 fidelity --jobs 8
    uv run python -m scripts.manuscript_v2 search --jobs 8
    uv run python -m scripts.manuscript_v2 reference --jobs 8
    uv run python -m scripts.manuscript_v2 summarise
"""

import argparse
import ast
import hashlib
import json
import platform
import re
import subprocess
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from joblib import Parallel, delayed, parallel_config
from scipy.linalg import hadamard
from scipy.stats import kendalltau, wasserstein_distance

from mutascape.benchmarks import roots
from mutascape.benchmarks.datasets import (
    COMBINGYM_MANIFEST,
    SSMULA_MANIFEST,
    _graphfla_spec,
    load_landscape,
)
from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.benchmarks.library import BudgetedOracle, SearchSpace
from mutascape.benchmarks.methods import (
    ACQUISITION_POOL,
    METHODS,
    SEARCH_PROTOCOL_VERSION,
    make_features,
)
from mutascape.benchmarks.validity import reliable_method_pairs
from mutascape.calibrated import DEFAULT_REFERENCE_SIZE, QuantileCalibratedLandscape
from mutascape.contacts import ContactMap
from mutascape.model import DMSLandscape

OUT = Path("results/manuscript_v2_20261015")
SSMULA = roots.ssmula()
COMBINGYM = roots.combingym()
GRAPHFLA = roots.graphfla()
CREILOV_RAW = COMBINGYM / "Data/DMS/Raw/CreiLOV_raw.xlsx"

MODEL_SELECTION = ("DHFR", "eqFP611", "ParB-Noc", "ParD2", "CreiLOV")
EVALUATION = ("GB1", "TrpB4", "RBD-S309", "PhoQ")
PROTEINS = (*MODEL_SELECTION, *EVALUATION)

CALIBRATION_SIZE = 96
CALIBRATION_SETS = 10
# C2ST dominates the run time, so it uses the first calibration sets only.
C2ST_SETS = 3
TRAINING_SETS = 3
LANDSCAPES = 3
SEARCH_REPEATS = 10
# Method pairs are judged reliable from experimental repeats, which need more power.
MEASURED_REPEATS = 50
BUDGETS = (48, 96)
SELECTION_SAMPLE = 1000
CUBES = 256
BOOTSTRAP = 2000

CONFIGS = {
    f"{mode}_m{margin:g}_a{strength:g}": dict(mode=mode, margin=margin, strength=strength)
    for mode in ("linear", "sigmoid", "pool")
    for margin in ((0.05, 0.5) if mode == "sigmoid" else (0.05,))
    for strength in (0.0, 0.1, 0.3, 1.0)
}
CLASSIFIER_SEED = 2026101500


def seed_for(role: int, protein: str, *parts: int) -> int:
    """Separate random streams by role, protein and experimental unit."""
    key = [20261015, role, PROTEINS.index(protein), *parts]
    return int(np.random.SeedSequence(key).generate_state(1)[0])


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")


def source(name: str):
    """Return the raw data root and dataset spec for one protein."""
    ssmula = {"DHFR": "DHFR", "ParD2": "ParD2", "GB1": "GB1", "TrpB4": "TrpB4"}
    if name in ssmula:
        return SSMULA, next(s for s in SSMULA_MANIFEST if s.name == ssmula[name])
    if name == "eqFP611":
        return COMBINGYM, next(s for s in COMBINGYM_MANIFEST if s.name == "eqFP611_red")
    if name == "ParB-Noc":
        return GRAPHFLA, _graphfla_spec("Jalal2020_NBS.csv", "ParB-Noc", "RTAG")
    if name == "PhoQ":
        return GRAPHFLA, _graphfla_spec("Podgornaia2015_PhoQ.csv", "PhoQ", "AVST")
    if name == "RBD-S309":
        # Presence-absence table of 15 Omicron BA.1 substitutions, ancestral Wuhan-Hu-1 as zero.
        return GRAPHFLA, _graphfla_spec("Moulana2023_S309.csv", "SARS-CoV-2-RBD", "0" * 15)
    raise ValueError(name)


HGVS = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")


def parse_hgvs(label: str) -> dict[int, str]:
    """Read ``p.Thr7Ser;p.Arg5Asp`` style labels, with ``wt`` for wild type."""
    if label == "wt":
        return {}
    found = [HGVS.fullmatch(part.strip()) for part in re.split(r"[;,]", label)]
    if not all(found):
        raise ValueError(f"cannot parse {label!r}")
    return {int(m.group(2)): seq1(m.group(3)) for m in found}


def load_creilov() -> tuple[pd.DataFrame, str, Path]:
    """Log fluorescence relative to WT, from the CombinGym raw workbook."""
    raw = pd.read_excel(CREILOV_RAW).rename(columns={"Unnamed: 0": "variant"})
    raw["mutations"] = raw.variant.map(parse_hgvs)
    raw["depth"] = raw.mutations.map(len)
    raw["fitness"] = raw.mean_log - float(raw.loc[raw.depth == 0, "mean_log"].iloc[0])
    raw = raw.loc[np.isfinite(raw.fitness), ["mutations", "depth", "fitness"]]
    sequence = "".join(Path("data/creilov/creilov_wt.fasta").read_text().splitlines()[1:])
    return raw.reset_index(drop=True), sequence, CREILOV_RAW


def load_raw(name: str):
    if name == "CreiLOV":
        frame, sequence, path = load_creilov()
        return frame, sequence, path, None
    root, spec = source(name)
    dataset = load_landscape(root, spec, require_complete=False)
    frame = dataset.variants.reset_index(drop=True)[["mutations", "depth", "fitness"]]
    return frame, dataset.wild_type_sequence, root / spec.relative_path, spec


def supported(frame: pd.DataFrame, sequence: str):
    """Keep genotypes whose substitutions all have a measured single, in code order."""
    singles = frame.loc[frame.depth == 1]
    allowed = {item for m in singles.mutations for item in m.items()}
    frame = frame.loc[frame.mutations.map(lambda m: set(m.items()) <= allowed)]
    positions = sorted({p for p, _ in allowed})
    alphabets = {
        p: sorted({sequence[p - 1]} | {a for q, a in allowed if q == p}) for p in positions
    }
    codes = np.array(
        [
            [alphabets[p].index(m.get(p, sequence[p - 1])) for p in positions]
            for m in frame.mutations
        ],
        dtype=np.int16,
    )
    order = np.lexsort(codes.T[::-1])
    frame = frame.iloc[order].reset_index(drop=True)
    if (frame.depth == 0).sum() != 1:
        raise ValueError("expected exactly one WT row")
    return frame, codes[order], tuple(len(alphabets[p]) for p in positions)


def contact_pairs(name: str, spec) -> list | None:
    """Site pairs in structural contact, where an SSMuLA structure matches the WT."""
    if spec is None or spec.source != "ssmula":
        return None
    stem = "TrpB" if name == "TrpB4" else name
    path = SSMULA / f"data/{stem}/{stem}.pdb"
    if not path.exists():
        return None
    structure = PDBParser(QUIET=True).get_structure(name, path)[0]
    for chain in structure:
        letters = "".join(
            seq1(chain[(" ", p, " ")].resname) if (" ", p, " ") in chain else "?"
            for p in spec.positions
        )
        if letters != spec.wild_type:
            continue
        contact = ContactMap.from_pdb(path, chain_id=chain.id, positions=spec.positions)
        if set(contact.positions) != set(spec.positions):
            continue
        mapping = {p: i + 1 for i, p in enumerate(spec.positions)}
        return sorted({tuple(sorted((mapping[p], mapping[q]))) for p, q in contact.contact_pairs()})
    return None


def sample_cubes(codes: np.ndarray, shape: tuple, seed: int) -> np.ndarray:
    """Complete three-site subcubes of measured genotypes."""
    rng = np.random.default_rng(seed)
    lookup = {tuple(c): i for i, c in enumerate(codes)}
    bits = np.array(list(product((0, 1), repeat=3)))
    cubes, seen = [], set()
    for _ in range(200_000):
        base = codes[rng.integers(len(codes))].copy()
        sites = rng.choice(codes.shape[1], 3, replace=False)
        corners = np.tile(base, (8, 1))
        for j, site in enumerate(sites):
            corners[:, site] = rng.choice(shape[site], 2, replace=False)[bits[:, j]]
        rows = [lookup.get(tuple(c)) for c in corners]
        if any(r is None for r in rows) or tuple(sorted(rows)) in seen:
            continue
        seen.add(tuple(sorted(rows)))
        cubes.append(rows)
        if len(cubes) == CUBES:
            break
    if len(cubes) < CUBES:
        raise ValueError("too few complete three-site subcubes")
    return np.array(cubes, dtype=np.int64)


def walsh_spectrum(values: np.ndarray, cubes: np.ndarray) -> np.ndarray:
    """Pool coefficient energy over subcubes, then split it by interaction order."""
    energy = ((values[cubes] @ hadamard(8) / 8)[:, 1:]) ** 2
    orders = np.array([i.bit_count() for i in range(1, 8)])
    totals = np.array([energy[:, orders == k].sum() for k in (1, 2, 3)])
    return totals / max(totals.sum(), 1e-30)


def classifier_settings() -> dict:
    """Read the C2ST classifier's parameters from the package rather than restating them."""
    source = Path("src/mutascape/benchmarks/evaluation.py")
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HistGradientBoostingClassifier"
        ):
            return {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg != "random_state"
            }
    raise ValueError("histogram classifier not found")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name in PROTEINS:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        raw, sequence, path, spec = load_raw(name)
        frame, codes, shape = supported(raw, sequence)
        multi = np.flatnonzero(frame.depth.to_numpy() > 1)
        cubes = sample_cubes(codes, shape, seed_for(1, name))
        eligible = np.setdiff1d(multi, cubes.ravel())
        calibration = np.array(
            [
                np.sort(
                    np.random.default_rng(seed_for(2, name, c)).choice(
                        eligible, CALIBRATION_SIZE, replace=False
                    )
                )
                for c in range(CALIBRATION_SETS)
            ]
        )
        selection_rows = np.sort(
            np.random.default_rng(seed_for(3, name)).choice(
                multi, min(SELECTION_SAMPLE, len(multi)), replace=False
            )
        )
        frame.to_pickle(folder / "frame.pkl")
        np.savez(
            folder / "design.npz",
            codes=codes,
            shape=np.array(shape),
            cubes=cubes,
            calibration=calibration,
            selection=selection_rows,
        )
        pairs = contact_pairs(name, spec)
        save_json(
            folder / "metadata.json",
            dict(
                sequence=sequence,
                source=str(path),
                source_sha256=digest(path),
                contacts=pairs,
            ),
        )
        f = frame.fitness.to_numpy()
        inventory.append(
            dict(
                protein=name,
                use="model selection" if name in MODEL_SELECTION else "evaluation",
                sites=len(shape),
                singles=int((frame.depth == 1).sum()),
                genotypes=len(frame),
                raw_genotypes=len(raw),
                multi_mutants=len(multi),
                best_single=float(f[frame.depth <= 1].max()),
                best=float(f.max()),
                at_minimum=float(np.mean(np.isclose(f, f.min()))),
                contacts=None if pairs is None else len(pairs),
            )
        )
        print(f"Prepared {name}: {len(frame):,} genotypes, {len(shape)} sites", flush=True)
    pd.DataFrame(inventory).to_csv(OUT / "inventory.csv", index=False)
    save_json(
        OUT / "manifest.json",
        dict(
            git_head=git_head(),
            model_selection=MODEL_SELECTION,
            evaluation=EVALUATION,
            calibration_size=CALIBRATION_SIZE,
            calibration_sets=CALIBRATION_SETS,
            c2st_sets=C2ST_SETS,
            training_sets=TRAINING_SETS,
            landscapes=LANDSCAPES,
            search_repeats=SEARCH_REPEATS,
            measured_repeats=MEASURED_REPEATS,
            budgets=BUDGETS,
            cubes=CUBES,
            selection_sample=SELECTION_SAMPLE,
            classifier=classifier_settings(),
            acquisition_pool=ACQUISITION_POOL,
            reference_size=DEFAULT_REFERENCE_SIZE,
            configs=CONFIGS,
            search_protocol=SEARCH_PROTOCOL_VERSION,
            methods=list(METHODS),
            versions=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                pandas=pd.__version__,
                scipy=scipy.__version__,
                sklearn=sklearn.__version__,
            ),
            source_sha256={
                str(p): digest(p)
                for p in [
                    Path(__file__),
                    Path("src/mutascape/calibrated.py"),
                    Path("src/mutascape/model.py"),
                    Path("src/mutascape/benchmarks/methods.py"),
                    Path("src/mutascape/benchmarks/evaluation.py"),
                ]
            },
        ),
    )


def data(name: str):
    folder = OUT / name
    return (
        pd.read_pickle(folder / "frame.pkl"),
        dict(np.load(folder / "design.npz")),
        json.loads((folder / "metadata.json").read_text()),
    )


def evaluate_all(model, frame: pd.DataFrame) -> np.ndarray:
    variants = frame.mutations.tolist()
    values = np.concatenate(
        [model.evaluate_batch(variants[j : j + 8192]) for j in range(0, len(variants), 8192)]
    )
    singles = frame.depth.to_numpy() <= 1
    np.testing.assert_array_equal(values[singles], frame.fitness.to_numpy()[singles])
    if not np.isfinite(values).all():
        raise ValueError("generated fitness is not finite")
    return values


def singles_model(name: str, config: dict, seed: int, contacts: bool = False) -> DMSLandscape:
    frame, _, meta = data(name)
    mask = {tuple(p) for p in meta["contacts"]} if contacts else None
    return DMSLandscape.from_singles(
        meta["sequence"], frame.loc[frame.depth <= 1], seed=seed, contacts=mask, **config
    )


def screen(name: str) -> None:
    target = OUT / name / "screen.csv"
    if target.exists():
        return
    frame, design, _ = data(name)
    real = frame.fitness.to_numpy()
    rows = design["selection"]
    measured = walsh_spectrum(real, design["cubes"])
    requested = np.unique(np.r_[rows, design["cubes"].ravel(), np.flatnonzero(frame.depth <= 1)])
    subset = frame.iloc[requested]
    records = []
    for label, config in CONFIGS.items():
        for landscape in range(LANDSCAPES):
            values = np.full(len(frame), np.nan)
            values[requested] = evaluate_all(
                singles_model(name, config, seed_for(4, name, landscape)), subset
            )
            records.append(
                dict(
                    protein=name,
                    config=label,
                    landscape=landscape,
                    w1=wasserstein_distance(real[rows], values[rows]) / real[rows].std(),
                    walsh=float(np.abs(measured - walsh_spectrum(values, design["cubes"])).sum()),
                )
            )
    pd.DataFrame(records).to_csv(target, index=False)
    print(f"Screened {name}", flush=True)


def select(jobs: int) -> None:
    Parallel(n_jobs=jobs)(delayed(screen)(name) for name in MODEL_SELECTION)
    scores = pd.concat([pd.read_csv(OUT / n / "screen.csv") for n in MODEL_SELECTION])
    medians = scores.groupby(["protein", "config"])[["w1", "walsh"]].median()
    ranks = medians.groupby(level="protein").rank(pct=True).mean(axis=1)
    ranks = ranks.groupby(level="config").mean().sort_values()
    chosen = {
        mode: str(ranks[[CONFIGS[k]["mode"] == mode for k in ranks.index]].idxmin())
        for mode in ("linear", "sigmoid", "pool")
    }
    ranks.to_csv(OUT / "selection_ranks.csv", header=["mean_rank"])
    save_json(OUT / "selected.json", chosen)
    print(json.dumps(chosen), flush=True)


def singles_configs(name: str) -> dict:
    chosen = json.loads((OUT / "selected.json").read_text())
    configs = {mode: CONFIGS[label] for mode, label in chosen.items()}
    configs["additive"] = dict(mode="linear", strength=0.0, margin=0.05)
    _, _, meta = data(name)
    if meta["contacts"]:
        configs["contacts"] = configs["sigmoid"]
    return configs


def calibrated_model(name: str, c: int) -> QuantileCalibratedLandscape:
    frame, design, meta = data(name)
    rows = design["calibration"][c]
    return QuantileCalibratedLandscape.fit(
        meta["sequence"], frame.loc[frame.depth <= 1], frame.iloc[rows]
    )


def generated(name: str, model: str, c: int, landscape: int) -> np.ndarray:
    """Generate and cache fitness for every genotype in the protein's search space."""
    folder = OUT / name / "landscapes"
    folder.mkdir(exist_ok=True)
    tag = f"{model}_{c}_{landscape}" if model == "calibrated" else f"{model}_{landscape}"
    path = folder / f"{tag}.npy"
    if path.exists():
        return np.load(path)
    frame, _, _ = data(name)
    seed = seed_for(5, name, c if model == "calibrated" else 0, landscape)
    if model == "calibrated":
        fitted = calibrated_model(name, c)
        save_json(folder / f"fit_{c}.json", fitted.to_dict())
        values = evaluate_all(fitted.draw(seed), frame)
    else:
        config = singles_configs(name)[model]
        values = evaluate_all(singles_model(name, config, seed, model == "contacts"), frame)
    np.save(path, values)
    return values


def c2st(name: str, rows: np.ndarray, values: np.ndarray, c: int) -> dict:
    frame, design, _ = data(name)
    singles, evaluation = frame.loc[frame.depth <= 1], frame.iloc[rows]
    real = completion_fidelity_features(singles, evaluation).to_numpy()
    synthetic = completion_fidelity_features(singles, evaluation, values[rows]).to_numpy()
    codes = design["codes"][rows]
    indicators = np.column_stack(
        [codes[:, s] == r for s, size in enumerate(design["shape"]) for r in range(size)]
    )
    # Real and synthetic copies of a genotype share a group, so they share a fold.
    groups = np.random.default_rng(seed_for(6, name, c)).permutation(len(rows))
    result = {}
    for label, x, y in [
        ("base", real, synthetic),
        ("sequence", np.column_stack([real, indicators]), np.column_stack([synthetic, indicators])),
    ]:
        score = classifier_two_sample_test(
            x,
            y,
            real_groups=groups,
            synthetic_groups=groups,
            classifier="histogram",
            n_permutations=0,
            seed=CLASSIFIER_SEED,
        )
        result[f"auc_{label}"] = float(score.auc)
    return result


def fidelity_task(name: str, model: str, landscape: int) -> None:
    """Score one generated landscape against every calibration split."""
    folder = OUT / name / "fidelity"
    folder.mkdir(exist_ok=True)
    target = folder / f"{model}_{landscape}.csv"
    if target.exists():
        return
    frame, design, _ = data(name)
    real = frame.fitness.to_numpy()
    multi = np.flatnonzero(frame.depth.to_numpy() > 1)
    measured_spectrum = walsh_spectrum(real, design["cubes"])
    records = []
    shared = None if model == "calibrated" else generated(name, model, 0, landscape)
    for c in range(CALIBRATION_SETS):
        values = generated(name, model, c, landscape) if shared is None else shared
        calibration = design["calibration"][c]
        rows = np.setdiff1d(multi, calibration)
        start = max(real[frame.depth <= 1].max(), real[calibration].max())
        spectrum = walsh_spectrum(values, design["cubes"])
        record = dict(
            protein=name,
            model=model,
            calibration=c,
            landscape=landscape,
            genotypes=len(rows),
            w1=wasserstein_distance(real[rows], values[rows]) / real[rows].std(),
            walsh=float(np.abs(measured_spectrum - spectrum).sum()),
            **{f"walsh_order{k + 1}": float(spectrum[k]) for k in range(3)},
            start=float(start),
            share_above_start=float(np.mean(values[rows] > start)),
            measured_share_above_start=float(np.mean(real[rows] > start)),
            generated_max=float(values[rows].max()),
            measured_max=float(real[rows].max()),
        )
        if c < C2ST_SETS and model in ("calibrated", "sigmoid"):
            record |= c2st(name, rows, values, c)
        records.append(record)
    pd.DataFrame(records).to_csv(target, index=False)
    print(f"Fidelity {name} {model} {landscape}", flush=True)


def fidelity(jobs: int) -> None:
    tasks = [
        (name, model, landscape)
        for name in PROTEINS
        for model in (*singles_configs(name), "calibrated")
        for landscape in range(1 if model == "additive" else LANDSCAPES)
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        Parallel(n_jobs=jobs)(delayed(fidelity_task)(*t) for t in tasks)
    # The measured spectrum is the same for every model, so record it once.
    spectra = []
    for name in PROTEINS:
        frame, design, _ = data(name)
        spectrum = walsh_spectrum(frame.fitness.to_numpy(), design["cubes"])
        spectra.append(dict(protein=name, **{f"order{k + 1}": spectrum[k] for k in range(3)}))
    pd.DataFrame(spectra).to_csv(OUT / "measured_spectra.csv", index=False)


def reference_task(name: str) -> None:
    """Compare sampled percentile maps with exact enumeration on calibration set zero."""
    target = OUT / name / "reference.csv"
    if target.exists():
        return
    frame, design, _ = data(name)
    real = frame.fitness.to_numpy()
    rows = np.setdiff1d(np.flatnonzero(frame.depth.to_numpy() > 1), design["calibration"][0])
    measured_spectrum = walsh_spectrum(real, design["cubes"])
    fitted = calibrated_model(name, 0)
    records = []
    for landscape in range(LANDSCAPES):
        seed = seed_for(5, name, 0, landscape)
        exact = evaluate_all(fitted.draw(seed, reference_size=None), frame)
        for reference_seed in range(3):
            sampled = evaluate_all(fitted.draw(seed, reference_seed=reference_seed), frame)
            difference = np.abs(sampled[rows] - exact[rows]) / real[rows].std()
            records.append(
                dict(
                    protein=name,
                    landscape=landscape,
                    reference_seed=reference_seed,
                    w1_change=abs(
                        wasserstein_distance(real[rows], sampled[rows])
                        - wasserstein_distance(real[rows], exact[rows])
                    )
                    / real[rows].std(),
                    walsh_change=abs(
                        np.abs(measured_spectrum - walsh_spectrum(sampled, design["cubes"])).sum()
                        - np.abs(measured_spectrum - walsh_spectrum(exact, design["cubes"])).sum()
                    ),
                    mean_abs_difference=float(difference.mean()),
                    max_abs_difference=float(difference.max()),
                )
            )
    pd.DataFrame(records).to_csv(target, index=False)
    print(f"Reference {name}", flush=True)


def landscape_values(name: str, model: str, c: int, landscape: int) -> np.ndarray:
    """Read a landscape that the fidelity stage generated."""
    tag = f"{model}_{c}_{landscape}" if model == "calibrated" else f"{model}_{landscape}"
    return np.load(OUT / name / "landscapes" / f"{tag}.npy")


class KnownOracle(BudgetedOracle):
    """Start from free known observations and count only new queries."""

    def __init__(self, values: np.ndarray, budget: int, known: np.ndarray):
        super().__init__(values, budget)
        self._known = len(known)
        self._rows = list(map(int, known))
        self._observed = list(map(float, values[known]))
        self._seen = dict(zip(self._rows, self._observed))

    @property
    def spent(self) -> int:
        return len(self._rows) - self._known


def search_task(name: str, c: int, model: str, landscape: int) -> None:
    folder = OUT / name / "search"
    folder.mkdir(exist_ok=True)
    target = folder / f"{model}_{c}_{landscape}.csv"
    if target.exists():
        return
    frame, design, _ = data(name)
    measured = frame.fitness.to_numpy()
    known = np.sort(np.r_[np.flatnonzero(frame.depth.to_numpy() <= 1), design["calibration"][c]])
    values = measured.copy() if model == "measured" else landscape_values(name, model, c, landscape)
    # Every arm starts from the same experimental observations.
    values[known] = measured[known]
    codes = design["codes"]
    space = SearchSpace(
        codes,
        tuple(int(s) for s in design["shape"]),
        np.arange(len(codes)),
        int(known[np.argmax(values[known])]),
        {tuple(int(v) for v in code): i for i, code in enumerate(codes)},
    )
    features = make_features(space)
    spread = max(float(np.ptp(values)), 1e-12)
    records = []
    for budget in BUDGETS:
        for trial in range(MEASURED_REPEATS if model == "measured" else SEARCH_REPEATS):
            seed = seed_for(7, name, c, trial)
            for method, function in METHODS.items():
                oracle = KnownOracle(values, budget, known)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    function(space, oracle, features, seed)
                new = oracle.rows[len(known) :]
                assert oracle.spent == budget == len(np.unique(new))
                assert not np.intersect1d(new, known).size
                records.append(
                    dict(
                        protein=name,
                        calibration=c,
                        model=model,
                        landscape=landscape,
                        budget=budget,
                        trial=trial,
                        method=method,
                        best=oracle.best(),
                        start=float(values[known].max()),
                        regret=float((values.max() - oracle.best()) / spread),
                        fallback=oracle.fallback_queries,
                        warnings=len(caught),
                    )
                )
    pd.DataFrame(records).to_csv(target, index=False)
    print(f"Search {name} {model} calibration {c} landscape {landscape}", flush=True)


def search(jobs: int) -> None:
    tasks = []
    for name in EVALUATION:
        for c in range(CALIBRATION_SETS):
            tasks.append((name, c, "measured", 0))
            tasks.append((name, c, "additive", 0))
            for model in ("calibrated", "sigmoid"):
                tasks += [(name, c, model, landscape) for landscape in range(LANDSCAPES)]
    tasks += [(n, c, "measured", 0) for n in MODEL_SELECTION for c in range(TRAINING_SETS)]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        Parallel(n_jobs=jobs)(delayed(search_task)(*t) for t in tasks)


def search_summary() -> pd.DataFrame:
    scores = pd.concat(
        [pd.read_csv(p) for n in PROTEINS for p in (OUT / n / "search").glob("*.csv")],
        ignore_index=True,
    )
    scores.to_csv(OUT / "search_scores.csv", index=False)
    training = (
        scores.loc[scores.protein.isin(MODEL_SELECTION)]
        .groupby(["budget", "protein", "method"])
        .regret.mean()
        .groupby(["budget", "method"])
        .mean()
    )
    results = []
    evaluation = scores.loc[scores.protein.isin(EVALUATION)]
    for (name, c, budget), group in evaluation.groupby(["protein", "calibration", "budget"]):
        means = (
            group.groupby(["model", "method", "trial"])
            .regret.mean()
            .reset_index()
            .rename(columns={"model": "arm"})
            .assign(dataset=name)
        )
        ranks = means.pivot_table(index="method", columns="arm", values="regret")
        eligible = reliable_method_pairs(means, seed=seed_for(8, name, c), n_bootstrap=BOOTSTRAP)
        arms = {arm: ranks[arm] for arm in ranks.columns if arm != "measured"}
        arms["fixed_ranking"] = training.loc[budget].reindex(ranks.index)
        for arm, synthetic in arms.items():
            correct = sum(
                np.sign(synthetic[r.first] - synthetic[r.second]) == r.sign
                for r in eligible.itertuples()
            )
            tied = np.ptp(synthetic.to_numpy()) < 1e-12
            results.append(
                dict(
                    protein=name,
                    calibration=c,
                    budget=budget,
                    arm=arm,
                    kendall_tau=np.nan if tied else kendalltau(ranks.measured, synthetic).statistic,
                    tied=tied,
                    eligible_pairs=len(eligible),
                    matching_pairs=int(correct),
                )
            )
    summary = pd.DataFrame(results)
    summary.to_csv(OUT / "search_summary.csv", index=False)
    return summary


def summarise() -> None:
    fidelity = pd.concat(
        [pd.read_csv(p) for n in PROTEINS for p in (OUT / n / "fidelity").glob("*.csv")],
        ignore_index=True,
    )
    fidelity.to_csv(OUT / "fidelity.csv", index=False)
    per_protein = (
        fidelity.groupby(["protein", "model", "calibration"])
        .median(numeric_only=True)
        .groupby(["protein", "model"])
        .mean()
        .drop(columns=["landscape"])
    )
    per_protein.to_csv(OUT / "fidelity_summary.csv")
    columns = ["w1", "walsh", "auc_sequence", "share_above_start", "measured_share_above_start"]
    print(per_protein.loc[list(EVALUATION), columns].round(3).to_string())
    summary = search_summary()
    table = (
        summary.groupby(["protein", "budget", "arm"])
        .agg(
            kendall_tau=("kendall_tau", "mean"),
            tied=("tied", "mean"),
            matching=("matching_pairs", "sum"),
            eligible=("eligible_pairs", "sum"),
        )
        .assign(agreement=lambda t: t.matching / t.eligible)
    )
    table.to_csv(OUT / "search_table.csv")
    print(table.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "select", "fidelity", "search", "reference", "summarise"]
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "select":
        select(args.jobs)
    elif args.stage == "fidelity":
        fidelity(args.jobs)
    elif args.stage == "search":
        search(args.jobs)
    elif args.stage == "reference":
        with parallel_config(backend="loky", inner_max_num_threads=1):
            Parallel(n_jobs=args.jobs)(delayed(reference_task)(name) for name in PROTEINS)
    else:
        summarise()


if __name__ == "__main__":
    main()
