"""Check CreiLOV source data, stored predictions and sigmoid ablations."""

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils import seq1
from scipy.special import expit, logit

from mutascape.model import _pair
from scripts.investigate_singles_only import SEEDS, build, data, final_models

out = Path("results/creilov_sigmoid_audit_20261006")
out.mkdir(parents=True, exist_ok=True)
f, d, meta = data("CreiLOV")
variants = f.mutations.tolist()
real = f.fitness.to_numpy()
raw = pd.read_excel(meta["source"])
assert hashlib.sha256(Path(meta["source"]).read_bytes()).hexdigest() == meta["source_sha256"]

parsed = []
for label in raw.iloc[:, 0].astype(str):
    parts = [
        (seq1(wt), p, seq1(aa))
        for wt, p, aa in re.findall(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", label)
    ]
    assert all(meta["sequence"][int(p) - 1] == wt for wt, p, aa in parts)
    parsed.append({int(p): aa for wt, p, aa in parts})
assert parsed == variants
wtrow = np.flatnonzero(f.depth.to_numpy() == 0)
assert len(wtrow) == 1
np.testing.assert_array_equal(raw.mean_log.to_numpy() - raw.mean_log.iloc[wtrow[0]], real)
np.testing.assert_allclose(raw.mean_log, np.log10(raw["mean"]), atol=1e-14)
np.testing.assert_allclose(raw["mean"], raw[["Rep1", "Rep2", "Rep3"]].mean(axis=1))
print("raw and WT mapping pass", raw.iloc[wtrow[0]].to_dict(), flush=True)
rows = np.load("results/manuscript_review_20261006/CreiLOV/distribution_rows.npy")
models = {}
checks = []
for seed in SEEDS:
    model = build("CreiLOV", final_models("CreiLOV")["sigmoid"], seed)
    y = model.evaluate_batch(variants)
    saved = np.load(f"results/singles_only_restart_20260923/CreiLOV/fitness_sigmoid_{seed}.npy")
    np.testing.assert_allclose(y, saved, rtol=0, atol=1e-14)
    keys = list(model.effects)
    X = np.array([[m.get(p) == aa for p, aa in keys] for m in variants], dtype=float)
    L = model.readout.floor
    U = model.readout.ceiling
    offset = logit(-L / (U - L))
    obs = np.array([model.observations[(k,)] for k in keys])
    effects = logit((obs - L) / (U - L)) - offset
    a = X @ effects
    pair = np.zeros(len(f))
    for i, j in combinations(range(len(keys)), 2):
        if keys[i][0] != keys[j][0]:
            pair += (
                X[:, i]
                * X[:, j]
                * _pair(keys[i], keys[j], meta["sequence"], seed)
                * model.strength
                * model.scale
            )
    independent = L + (U - L) * expit(a + pair + offset)
    np.testing.assert_allclose(y, independent, rtol=0, atol=2e-15)
    idx = np.random.default_rng(4).choice(len(f), 50, replace=False)
    np.testing.assert_allclose(
        y[idx], np.array([model.evaluate(variants[i]) for i in idx]), atol=1e-15
    )
    np.testing.assert_allclose(
        y[idx[::-1]], model.evaluate_batch([variants[i] for i in idx[::-1]]), atol=1e-15
    )
    models[f"Mutascape {seed}"] = y
    checks.append(
        dict(
            seed=seed,
            max_saved_error=float(abs(y - saved).max()),
            max_formula_error=float(abs(y - independent).max()),
            mean_additive_latent=float(a[rows].mean()),
            mean_pair_latent=float(pair[rows].mean()),
            pair_sd=float(pair[rows].std()),
        )
    )
models["Measured"] = real
models["Zero pair sigmoid"] = L + (U - L) * expit(a + offset)
models["Additive"] = X @ obs
for margin in (0.5, 2.0):
    model = build("CreiLOV", dict(mode="sigmoid", margin=margin, strength=0), SEEDS[0])
    models[f"Zero pair margin {margin}"] = model.evaluate_batch(variants)
summary = []
depths = []
for name, y in models.items():
    v = y[rows]
    summary.append(
        dict(
            model=name,
            n=len(v),
            mean=float(v.mean()),
            median=float(np.median(v)),
            min=float(v.min()),
            max=float(v.max()),
            above_WT=float(np.mean(v > 0)),
            near_ceiling=float(np.mean(v > U - 0.01 * (U - L))),
        )
    )
    for depth in (2, 4, 8, 12, 15):
        subset = np.flatnonzero(f.depth.to_numpy() == depth)
        depths.append(
            dict(
                model=name,
                depth=depth,
                n=len(subset),
                mean=float(y[subset].mean()),
                above_WT=float(np.mean(y[subset] > 0)),
            )
        )
pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
pd.DataFrame(depths).to_csv(out / "by_depth.csv", index=False)
(out / "checks.json").write_text(
    json.dumps(
        dict(
            checks=checks,
            floor=L,
            ceiling=U,
            single_positive=int((obs > 0).sum()),
            single_negative=int((obs < 0).sum()),
        ),
        indent=2,
    )
)
print(pd.DataFrame(summary).to_string(index=False))
print(checks)
side = json.loads(Path("../mutascape_manu_v1/figures/creilov_shape_data.json").read_text())
bins = np.array(side["bin_edges"])
for name, ys in [("Mutascape", [models[f"Mutascape {s}"] for s in SEEDS])]:
    hist = np.array([np.histogram(y[rows], bins=bins, density=True)[0] for y in ys])
    np.testing.assert_allclose(hist, side["histograms"][name])
    print(
        "figure verified",
        name,
        "median histogram integral",
        float(np.sum(np.median(hist, axis=0) * np.diff(bins))),
    )
