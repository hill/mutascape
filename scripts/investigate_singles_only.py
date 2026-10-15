"""Evaluate simple generators without target combination inputs."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from joblib import Parallel, delayed
from scipy.linalg import hadamard
from scipy.stats import kendalltau, spearmanr, wasserstein_distance

from mutascape.benchmarks import roots
from mutascape.benchmarks.datasets import SSMULA_MANIFEST, load_landscape
from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features, run_method
from mutascape.benchmarks.validity import pairwise_winner_transfer, reliable_method_pairs
from mutascape.contacts import ContactMap
from mutascape.model import SimpleLandscape
from scripts.confirm_quantile_pilot import load as load_panel

OUT = Path("results/singles_only_restart_20260923")
DEVELOPMENT = ("DHFR", "eqFP611", "ParB-Noc", "ParD2", "CreiLOV")
CONFIRMATION = ("GB1", "TrpB4", "T7", "TEV")
SEEDS = (2026092311, 2026092312, 2026092313)
CONFIGS = {
    f"{mode}_m{margin:g}_a{strength:g}": dict(mode=mode, margin=margin, strength=strength)
    for mode in ("linear", "sigmoid", "pool")
    for margin in ((0.05, 0.5) if mode == "sigmoid" else (0.05,))
    for strength in (0.0, 0.1, 0.3, 1.0)
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    if name in DEVELOPMENT:
        frame, singles, sequence, codes, shape, source = load_panel(name)
        spec = next((s for s in SSMULA_MANIFEST if s.name == name), None)
    else:
        spec = next(s for s in SSMULA_MANIFEST if s.name == name)
        source = roots.ssmula() / spec.relative_path
        data = load_landscape(source.parents[3], spec)
        sequence = data.wild_type_sequence
        frame = data.variants.reset_index(drop=True)
        singles = frame.loc[frame.depth <= 1]
        supported = {item for m in singles.mutations for item in m.items()}
        frame = frame.loc[frame.mutations.map(lambda m: set(m.items()) <= supported)].reset_index(
            drop=True
        )
        singles = frame.loc[frame.depth <= 1]
        positions = sorted({p for p, _ in supported})
        alphabets = {
            p: sorted({sequence[p - 1]} | {a for q, a in supported if p == q}) for p in positions
        }
        codes = np.array(
            [
                [alphabets[p].index(m.get(p, sequence[p - 1])) for p in positions]
                for m in frame.mutations
            ]
        )
        shape = tuple(len(alphabets[p]) for p in positions)
    return frame, singles, sequence, codes, shape, source, spec


def contact_mask(name, spec):
    if spec is None:
        return None, {"available": False, "reason": "No verified residue mapping in this study"}
    stem = "TrpB" if name == "TrpB4" else name
    source = roots.ssmula() / "data" / stem / f"{stem}.pdb"
    structure = PDBParser(QUIET=True).get_structure(name, source)[0]
    for chain in structure:
        letters = "".join(
            seq1(chain[(" ", p, " ")].resname) if (" ", p, " ") in chain else "?"
            for p in spec.positions
        )
        if letters != spec.wild_type:
            continue
        contact = ContactMap.from_pdb(source, chain_id=chain.id, positions=spec.positions)
        if set(contact.positions) != set(spec.positions):
            continue
        mapping = {p: i + 1 for i, p in enumerate(spec.positions)}
        pairs = {tuple(sorted((mapping[p], mapping[q]))) for p, q in contact.contact_pairs()}
        return pairs, dict(
            available=True,
            source=str(source),
            sha256=digest(source),
            chain=chain.id,
            pairs=sorted(pairs),
        )
    return None, {"available": False, "reason": "Structure residues do not match the library WT"}


def sample_cubes(codes, shape, seed):
    rng = np.random.default_rng(seed)
    lookup = {tuple(c): i for i, c in enumerate(codes)}
    cubes, seen = [], set()
    bits = np.array(list(product((0, 1), repeat=3)))
    for _ in range(50_000):
        base = codes[rng.integers(len(codes))].copy()
        sites = rng.choice(codes.shape[1], 3, replace=False)
        corners = np.tile(base, (8, 1))
        for j, site in enumerate(sites):
            alternatives = rng.choice(shape[site], 2, replace=False)
            corners[:, site] = alternatives[bits[:, j]]
        rows = [lookup.get(tuple(c)) for c in corners]
        if any(r is None for r in rows):
            continue
        key = tuple(sorted(rows))
        if key in seen:
            continue
        seen.add(key)
        cubes.append(rows)
        if len(cubes) == 256:
            break
    if len(cubes) < 64:
        raise ValueError("too few complete three-site subcubes")
    return np.array(cubes, dtype=int)


def local_geometry(values, cubes):
    cells = values[cubes]
    coefficients = cells @ hadamard(8) / 8
    energy = coefficients[:, 1:] ** 2
    # Pool energy before normalisation so constant subcubes contribute zero.
    total = energy.sum()
    orders = np.array([i.bit_count() for i in range(1, 8)])
    spectrum = np.array([energy[:, orders == k].sum() for k in (1, 2, 3)]) / max(total, 1e-30)
    signs = np.zeros(len(cubes))
    for axis in (1, 2, 4):
        base = np.array([i for i in range(8) if not i & axis])
        diffs = cells[:, base + axis] - cells[:, base]
        signs += (diffs.min(axis=1) < 0) & (diffs.max(axis=1) > 0)
    return spectrum, float(np.mean(signs / 3))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        development=DEVELOPMENT,
        confirmation=CONFIRMATION,
        configs=CONFIGS,
        seeds=SEEDS,
        test_size=1000,
        cubes=256,
        cube_dimension=3,
        budgets=[48, 96],
        search_seeds=10,
        target_inputs="WT sequence, WT fitness and same-assay single-mutant fitness only",
        source_sha256={
            p: digest(p)
            for p in [
                __file__,
                "src/mutascape/model.py",
                "docs/singles_only_research_protocol.md",
            ]
        },
    )
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol)), "protocol changed"
    else:
        path.write_text(json.dumps(protocol, indent=2) + "\n")
    inventory = []
    for number, name in enumerate((*DEVELOPMENT, *CONFIRMATION)):
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        frame, singles, sequence, codes, shape, source, spec = load(name)
        contacts, mapping = contact_mask(name, spec)
        test = np.random.default_rng(2026092300 + number).choice(
            np.flatnonzero(frame.depth > 1), min(1000, int((frame.depth > 1).sum())), replace=False
        )
        cubes = sample_cubes(codes, shape, 2026092300 + number)
        frame.to_pickle(folder / "frame.pkl")
        np.savez(folder / "design.npz", codes=codes, shape=shape, test=test, cubes=cubes)
        metadata = dict(
            sequence=sequence, source=str(source), source_sha256=digest(source), structure=mapping
        )
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        inventory.append(
            dict(
                protein=name,
                phase="development" if name in DEVELOPMENT else "confirmation",
                measured=len(frame),
                declared=int(np.prod(shape)),
                singles=len(singles) - 1,
                cubes=len(cubes),
                contacts=len(contacts) if contacts is not None else None,
            )
        )
    pd.DataFrame(inventory).to_csv(OUT / "inventory.csv", index=False)
    print(pd.DataFrame(inventory).to_string(index=False), flush=True)


def data(name):
    folder = OUT / name
    return (
        pd.read_pickle(folder / "frame.pkl"),
        dict(np.load(folder / "design.npz")),
        json.loads((folder / "metadata.json").read_text()),
    )


def build(name, config, seed, contacts=False, structure_only=False):
    frame, _, meta = data(name)
    singles = frame.loc[frame.depth <= 1].copy()
    if structure_only:
        rng = np.random.default_rng(seed + 9101)
        singles["fitness"] = rng.standard_normal(len(singles))
        singles.loc[singles.depth == 0, "fitness"] = 0.0
    mask = {tuple(pair) for pair in meta["structure"]["pairs"]} if contacts else None
    return SimpleLandscape.from_singles(
        meta["sequence"], singles, seed=seed, contacts=mask, **config
    )


def diagnostics(name, values, frame, design):
    test = design["test"]
    real = frame.fitness.to_numpy()
    measured, real_signs = local_geometry(real, design["cubes"])
    generated, synthetic_signs = local_geometry(values, design["cubes"])
    real_std = np.std(real[test])
    syn_std = np.std(values[test])
    return dict(
        w1=wasserstein_distance(real[test], values[test]) / real_std,
        shape_w1=wasserstein_distance(
            (real[test] - np.mean(real[test])) / real_std,
            (values[test] - np.mean(values[test])) / max(syn_std, 1e-12),
        ),
        local_walsh_l1=float(np.abs(measured - generated).sum()),
        measured_local_additive=measured[0],
        generated_local_additive=generated[0],
        measured_sign_switch=real_signs,
        generated_sign_switch=synthetic_signs,
        spearman=float(spearmanr(real[test], values[test]).statistic),
    )


def screen(name):
    folder = OUT / name
    destination = folder / "screen.csv"
    if destination.exists():
        return
    frame, design, _ = data(name)
    requested = np.unique(
        np.r_[design["test"], design["cubes"].ravel(), np.flatnonzero(frame.depth <= 1)]
    )
    variants = frame.iloc[requested].mutations.tolist()
    rows = []
    for label, config in CONFIGS.items():
        for seed in SEEDS:
            model = build(name, config, seed)
            values = np.full(len(frame), np.nan)
            values[requested] = model.evaluate_batch(variants)
            np.testing.assert_array_equal(
                values[frame.depth <= 1], frame.loc[frame.depth <= 1, "fitness"]
            )
            rows.append(
                dict(
                    protein=name,
                    config=label,
                    seed=seed,
                    **diagnostics(name, values, frame, design),
                )
            )
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"Screened {name}", flush=True)


def select():
    scores = pd.concat([pd.read_csv(OUT / n / "screen.csv") for n in DEVELOPMENT])
    medians = scores.groupby(["protein", "config"])[["w1", "local_walsh_l1"]].median()
    ranks = (
        medians.groupby(level="protein").rank(pct=True).mean(axis=1).groupby(level="config").mean()
    )
    chosen = {}
    for mode in ("linear", "sigmoid", "pool"):
        candidates = ranks[[CONFIGS[k]["mode"] == mode for k in ranks.index]]
        chosen[mode] = str(candidates.idxmin())
    (OUT / "selected.json").write_text(json.dumps(chosen, indent=2) + "\n")
    ranks.sort_values().to_csv(OUT / "development_setting_scores.csv", header=["mean_rank"])
    print(json.dumps(chosen), flush=True)


def final_models(name):
    selected = json.loads((OUT / "selected.json").read_text())
    configs = {mode: CONFIGS[label] for mode, label in selected.items()}
    configs["additive"] = dict(mode="linear", strength=0.0, margin=0.05)
    configs["fixed_pairs"] = dict(mode="linear", strength=0.3, margin=0.05)
    _, _, meta = data(name)
    if meta["structure"]["available"]:
        configs["sigmoid_contacts"] = configs["sigmoid"]
        configs["structure_only"] = dict(mode="linear", strength=0.3, margin=0.05)
    return configs


def final(name, label, config, seed):
    folder = OUT / name
    path = folder / f"metrics_{label}_{seed}.json"
    if path.exists():
        return
    frame, design, _ = data(name)
    model = build(
        name,
        config,
        seed,
        contacts=label in {"sigmoid_contacts", "structure_only"},
        structure_only=label == "structure_only",
    )
    values = model.evaluate_batch(frame.mutations.tolist())
    np.save(folder / f"fitness_{label}_{seed}.npy", values)
    record = dict(protein=name, model=label, seed=seed, **diagnostics(name, values, frame, design))
    if label != "structure_only":
        test = frame.iloc[design["test"]]
        real = completion_fidelity_features(frame.loc[frame.depth <= 1], test).to_numpy()
        generated = completion_fidelity_features(
            frame.loc[frame.depth <= 1], test, values[design["test"]]
        ).to_numpy()
        codes = design["codes"][design["test"]]
        indicators = np.column_stack(
            [codes[:, j] == a for j, n in enumerate(design["shape"]) for a in range(n)]
        )
        for suffix, x, y in [
            ("base", real, generated),
            (
                "sequence",
                np.column_stack([real, indicators]),
                np.column_stack([generated, indicators]),
            ),
        ]:
            result = classifier_two_sample_test(
                x,
                y,
                real_groups=design["test"],
                synthetic_groups=design["test"],
                classifier="histogram",
                n_permutations=0,
                seed=2026092300,
            )
            record[f"auc_{suffix}"] = result.auc
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Evaluated {name} {label} {seed}", flush=True)


def search(name, label, seed):
    folder = OUT / name
    destination = folder / f"search_{label}_{seed}.csv"
    if destination.exists():
        return
    frame, design, _ = data(name)
    selected = np.flatnonzero(frame.depth > 1)
    flat = np.ravel_multi_index(design["codes"].T, design["shape"])
    selected = selected[np.argsort(flat[selected])]
    codes = design["codes"][selected]
    values = (
        frame.fitness.to_numpy()
        if label == "measured"
        else np.load(folder / f"fitness_{label}_{seed}.npy")
    )
    values = values[selected]
    space = SearchSpace(
        codes, tuple(design["shape"]), selected, None, {tuple(c): i for i, c in enumerate(codes)}
    )
    features = make_features(space)
    rows = []
    for budget in (48, 96):
        for method in METHODS:
            for trial in range(10):
                best = run_method(
                    method, space, values, features, budget=budget, seed=2026092400 + 137 * trial
                )
                rows.append(
                    dict(
                        protein=name,
                        model=label,
                        seed=seed,
                        budget=budget,
                        method=method,
                        trial=trial,
                        regret=(values.max() - best) / max(np.ptp(values), 1e-12),
                    )
                )
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"Searched {name} {label} {seed}", flush=True)


def summarise():
    names = (*DEVELOPMENT, *CONFIRMATION)
    fidelity = pd.DataFrame(
        [json.loads(p.read_text()) for name in names for p in (OUT / name).glob("metrics_*.json")]
    )
    fidelity.to_csv(OUT / "fidelity.csv", index=False)
    scores = pd.concat(
        [pd.read_csv(p) for name in names for p in (OUT / name).glob("search_*.csv")],
        ignore_index=True,
    )
    scores.to_csv(OUT / "search_scores.csv", index=False)
    results = []
    measured = (
        scores.loc[scores.model == "measured"]
        .groupby(["protein", "budget", "method"])
        .regret.mean()
        .unstack()
    )
    for (name, budget), group in scores.groupby(["protein", "budget"]):
        means = (
            group.groupby(["model", "method", "trial"])
            .regret.mean()
            .reset_index()
            .rename(columns={"model": "arm"})
            .assign(dataset=name)
        )
        pairs = pairwise_winner_transfer(means, seed=2026092300, n_bootstrap=2000)
        ranks = means.pivot_table(index="method", columns="arm", values="regret")
        for row in pairs.to_dict("records"):
            model = row.pop("arm")
            row.pop("dataset")
            results.append(
                dict(
                    protein=name,
                    budget=budget,
                    model=model,
                    kendall_tau=kendalltau(ranks.measured, ranks[model]).statistic,
                    **row,
                )
            )
        donors = [p for p in DEVELOPMENT if p != name]
        prior = measured.loc[[(p, budget) for p in donors]].mean()
        eligible = reliable_method_pairs(means, seed=2026092300, n_bootstrap=2000)
        correct = sum(
            np.sign(prior[r.first] - prior[r.second]) == r.sign for r in eligible.itertuples()
        )
        results.append(
            dict(
                protein=name,
                budget=budget,
                model="fixed_ranking",
                kendall_tau=kendalltau(ranks.measured, prior.reindex(ranks.index)).statistic,
                scored_pairs=len(eligible),
                correct_pairs=correct,
                winner_accuracy=correct / len(eligible) if len(eligible) else np.nan,
            )
        )
    summary = pd.DataFrame(results)
    summary["phase"] = np.where(summary.protein.isin(DEVELOPMENT), "development", "confirmation")
    summary.to_csv(OUT / "search_summary.csv", index=False)
    print(
        summary.groupby(["phase", "model"])[["kendall_tau", "winner_accuracy"]]
        .mean()
        .round(3)
        .to_string(),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["prepare", "screen", "select", "final", "search", "summarise"]
    )
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    names = CONFIRMATION if args.confirm else DEVELOPMENT
    if args.phase == "prepare":
        prepare()
    elif args.phase == "screen":
        Parallel(n_jobs=args.jobs)(delayed(screen)(name) for name in names)
    elif args.phase == "select":
        select()
    elif args.phase == "final":
        Parallel(n_jobs=args.jobs)(
            delayed(final)(name, label, config, seed)
            for name in names
            for label, config in final_models(name).items()
            for seed in (SEEDS[:1] if config["strength"] == 0 else SEEDS)
        )
    elif args.phase == "search":
        tasks = [(name, "measured", 0) for name in names]
        tasks += [
            (name, label, seed)
            for name in names
            for label, config in final_models(name).items()
            for seed in (SEEDS[:1] if config["strength"] == 0 else SEEDS)
        ]
        Parallel(n_jobs=args.jobs)(delayed(search)(*task) for task in tasks)
    else:
        summarise()


if __name__ == "__main__":
    main()
