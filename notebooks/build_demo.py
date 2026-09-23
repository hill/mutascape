"""Write notebooks/demo.ipynb.

The notebook is generated so its cell boundaries and prose stay under review
alongside the package. Run this, then execute the notebook to fill outputs.
"""

import subprocess
from pathlib import Path

import nbformat as nbf

OUTPUT = Path(__file__).parent / "demo.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Mutascape, end to end

Mutascape turns a wild type and a set of measured single mutants into a
combinatorially complete oracle you can query at any depth. This notebook builds
one for CreiLOV, then opens the 165,428-variant combinatorial landscape Chen et
al. measured over 15 positions and checks the synthetic surface against it.

Four questions.

1. What does the oracle guarantee, and how do you build one?
2. How close is a synthetic CreiLOV landscape to the measured one?
3. Does a benchmark run on the synthetic landscape reach the same conclusion as
   one run on the measured landscape?
4. How do you point all of this at a different protein?

Every number below is computed here. Nothing is quoted from a paper.

## Before you start

The measured landscapes come from CombinGym and SSMuLA. Fetch them once.

```bash
uv run python scripts/fetch_benchmark_data.py --combingym --ssmula
```

That writes to `.mutascape/benchmark-data/`, which Git ignores.
""")

md("""
## 1. Configure

This is the only cell you edit to change protein.
""")

code("""
TARGET = "CreiLOV_fluorescence"
SEED = 20260909
COUPLING = 0.1  # frozen default, validated across the benchmark corpora
BUDGET = 96  # queries a search method may spend
TRIALS = 5  # search seeds per method and arm

BENCHMARK_ROOT = "../.mutascape/benchmark-data"

# Other targets, all one-line swaps.
#
#   CombinGym : GB1_binding, CR9114_H1_binding, CR6261_H1_binding,
#               eqFP611_red, eqFP611_blue, HIV1_activity, RhlA_RLs
#   SSMuLA    : GB1, TrpB4, TEV, DHFR, ParD3, T7, TrpB3A
""")

code('''
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from mutascape import from_dms, load_oracle, run_campaign
from mutascape.benchmarks import (
    classifier_two_sample_test,
    completion_fidelity_features,
    load_combingym_landscapes,
    load_ssmula_landscapes,
    mutation_order_split,
)
from mutascape.benchmarks.generators import (
    dms_pairwise_readout_tensor,
    nk_tensor,
    potts_tensor,
)
from mutascape.benchmarks.library import SearchSpace
from mutascape.benchmarks.methods import METHODS, make_features, run_method
from mutascape.benchmarks.topology import additive_tensor, measured_tensor, supported_cube
from mutascape.validation.fourier import walsh_decompose
from mutascape.validation.metrics import local_optima_from_tensor

pd.set_option("display.width", 130)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

ROOT = Path(BENCHMARK_ROOT)


def corpus() -> dict:
    """Every measured landscape on disk, keyed by name."""
    found = load_combingym_landscapes(ROOT / "combingym", strict=False)
    found += load_ssmula_landscapes(ROOT / "ssmula", strict=False)
    if not found:
        raise FileNotFoundError(
            f"no landscapes under {ROOT.resolve()}. "
            "Run scripts/fetch_benchmark_data.py --combingym --ssmula first."
        )
    return {d.name: d for d in found}


LANDSCAPES = corpus()
dataset = LANDSCAPES[TARGET]
print(f"{dataset.name}   protein {dataset.protein}   phenotype {dataset.spec.phenotype}")
print(f"{dataset.n_variants:,} measured genotypes over {len(dataset.mutable_positions)} positions")
print(f"positions {dataset.mutable_positions}")
print(f"source    {dataset.spec.source_url}")
''')

md("""
## 2. Split the measured landscape

Mutascape sees the wild type and the single mutants. Everything of depth two and
above is held out, and reappears only at the end as the answer key.
""")

code("""
split = mutation_order_split(dataset.variants, max_train_depth=1)
sequence = dataset.wild_type_sequence
wild_type_fitness = float(split.train.loc[split.train.depth == 0, "fitness"].iloc[0])

print(f"visible to mutascape : {len(split.train):>7,}  (wild type and singles)")
print(f"held out             : {len(split.test):>7,}  (depth 2 and above)")
print(f"wild-type assay value: {wild_type_fitness:.4f}")

dms = pd.DataFrame(
    [
        {
            "position": position,
            "wildtype": sequence[position - 1],
            "mutant": residue,
            "fitness": fitness - wild_type_fitness,
        }
        for mutations, fitness in split.train.loc[
            split.train.depth == 1, ["mutations", "fitness"]
        ].itertuples(index=False, name=None)
        for position, residue in mutations.items()
    ]
)
dms.sort_values("fitness", ascending=False).head(8)
""")

md("""
Fitness is relative to wild type throughout, so wild type sits at zero. The
measured tensor is recentred the same way, which keeps the two comparable.

Note how little mutascape gets. CreiLOV supplies twenty singles across fifteen
positions, and from that it must produce a surface over the whole cube. Every
weakness later in the notebook traces back to this line.
""")

md("""
## 3. Build the oracle

Three calls. `from_dms` conditions a fitted prior on your measurements, `draw`
fixes one landscape out of that posterior, and the result is subscriptable.
""")

code("""
generator = from_dms(dms, wt_sequence=sequence)
oracle = generator.draw(SEED)
oracle
""")

code("""
top = dms.sort_values("fitness", ascending=False).head(4)
variant = "+".join(f"{r.wildtype}{r.position}{r.mutant}" for r in top.itertuples())
single = f"{top.wildtype.iloc[0]}{top.position.iloc[0]}{top.mutant.iloc[0]}"

print(f"{variant}  ->  {oracle[variant]:+.4f}")
print(f"as a dict     ->  {oracle.evaluate({top.position.iloc[0]: top.mutant.iloc[0]}):+.4f}")
print()
print(oracle.score_many([variant, single]))
""")

md("""
## 4. What the oracle guarantees

Four properties make this a benchmark oracle rather than a random function
generator.
""")

code("""
singles = split.train.loc[split.train.depth == 1]
predicted = oracle.evaluate_batch(singles.mutations.tolist())
observed = singles.fitness.to_numpy() - wild_type_fitness
error = np.abs(predicted - observed).max()

checks = {
    "wild type is exactly zero": oracle.evaluate({}) == 0.0,
    "measured singles are exact": error < 1e-12,
    "a repeated query never moves": oracle[variant] == oracle[variant],
    "held-out depths are answerable": np.isfinite(
        oracle.evaluate_batch(split.test.mutations.head(50).tolist())
    ).all(),
}
for label, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {label}")
print(f"\\nlargest single-mutant error over {len(singles):,} measurements: {error:.2e}")
""")

md("""
### Provenance

Ask the oracle where any value came from. Measured values come back untouched,
so a benchmark cannot quietly grade itself against invented data.
""")

code("""
probe = [{}, singles.mutations.iloc[0], *split.test.mutations.head(3).tolist()]
for mutations in probe:
    label = "+".join(f"{sequence[p - 1]}{p}{a}" for p, a in sorted(mutations.items()))
    value = oracle.evaluate(mutations)
    print(f"{oracle.provenance(mutations):>9}  {value:+.4f}  {label or '(wild type)'}")
""")

md("""
### A seed is a landscape

`draw` samples a posterior. Different seeds give different landscapes that share
every measured anchor. That is what lets you report a benchmark result with an
error bar over landscapes instead of one lucky surface.
""")

code("""
draws = {seed: generator.draw(seed) for seed in (SEED, SEED + 1, SEED + 2)}
held_out = split.test.mutations.head(6).tolist()

table = pd.DataFrame(
    {f"seed {seed}": np.round(draw.evaluate_batch(held_out), 4) for seed, draw in draws.items()},
    index=[f"depth {len(m)}" for m in held_out],
)
anchor = singles.mutations.iloc[0]
spread = max(abs(draw.evaluate(anchor) - oracle.evaluate(anchor)) for draw in draws.values())
print(f"spread across seeds on a measured single: {spread:.2e}")
table
""")

md("""
### One file reproduces the oracle

`save_manifest` records the generator, the seed, the coupling, the prior and a
digest of the source data. `load_oracle` rebuilds it and refuses if the data
moved underneath it.
""")

code("""
manifest_path = Path("demo_oracle.json")
oracle.save_manifest(manifest_path)
restored = load_oracle(manifest_path, dms=dms, wt_sequence=sequence)

check = split.test.mutations.head(2000).tolist()
gap = np.abs(oracle.evaluate_batch(check) - restored.evaluate_batch(check)).max()
print(f"largest disagreement over 2,000 genotypes: {gap:.2e}\\n")
print(json.dumps(oracle.manifest(), indent=2))
""")

code("""
tampered = dms.copy()
tampered.loc[0, "fitness"] += 0.01
try:
    load_oracle(manifest_path, dms=tampered, wt_sequence=sequence)
except ValueError as failure:
    print(f"refused: {failure}")
""")

md("""
## 5. Lazy over a domain you could never enumerate

The oracle never builds a table. It scores what you ask for, so the declared
domain can be far larger than any machine could hold.
""")

code("""
domain = oracle.domain
print(f"positions      {len(domain.positions)}")
print(f"substitutions  {len(domain.substitutions)}")
for depth in (1, 2, 3, 5, len(domain.positions)):
    print(f"genotypes at depth {depth:>2}: {domain.count(depth=depth):>12,}")
print(f"whole domain      : {domain.count():>12,}")
print()
for mutant in domain.sample(depth=4, n=4, seed=SEED):
    print(f"{mutant.label:>28}  {oracle.evaluate(mutant):+.4f}")
""")

code("""
batch = domain.sample(depth=6, n=5000, seed=SEED)
start = time.perf_counter()
values = oracle.evaluate_batch(batch)
elapsed = time.perf_counter() - start
print(f"{len(batch):,} depth-6 genotypes in {elapsed:.2f}s "
      f"({len(batch) / elapsed:,.0f} per second)")
print(f"range {values.min():+.3f} to {values.max():+.3f}")
""")

md("""
### The same idea at protein scale

CreiLOV also has a published deep mutational scan covering the whole protein.
Conditioning on that gives an oracle over a domain with more genotypes than a
combinatorial assay could ever reach.
""")

code("""
scan = Path("../data/creilov/creilov.csv")
if dataset.protein == "CreiLOV" and scan.exists():
    whole = from_dms(scan, fasta="../data/creilov/creilov_wt.fasta").draw(SEED)
    print(whole)
    for depth in (2, 4, 6, 8):
        print(f"genotypes at depth {depth}: {whole.domain.count(depth=depth):.3e}")
    print()
    for mutant in whole.domain.sample(depth=6, n=3, seed=SEED):
        print(f"{mutant.label:>34}  {whole.evaluate(mutant):+.4f}")
else:
    print("skipped: this section needs the tracked CreiLOV whole-protein scan")
""")

md("""
## 6. Against the measured combinatorial landscape

Open the answer key. `supported_cube` restricts the comparison to substitutions
backed by a measured single, so both surfaces describe the same object.
""")

code("""
cube = supported_cube(dataset, split.train)
measured, coverage = measured_tensor(dataset, cube)
finite = np.isfinite(measured)
synthetic = dms_pairwise_readout_tensor(cube, generator, coupling=COUPLING, seed=SEED)

print(f"cube shape  {measured.shape}")
print(f"cube size   {measured.size:,} genotypes")
print(f"measured    {finite.sum():,} ({coverage:.1%} of the cube)")
""")

md("""
`dms_pairwise_readout_tensor` is the array form of the model the oracle
evaluates one genotype at a time. If they disagree, one of them is wrong.
""")

code("""
variants, indices = cube.variants()
picked = np.random.default_rng(0).choice(len(variants), 500, replace=False)
direct = oracle.evaluate_batch([variants[i] for i in picked])
via_tensor = np.array([synthetic[indices[i]] for i in picked])
print(f"largest disagreement over 500 genotypes: {np.abs(direct - via_tensor).max():.2e}")
""")

md("""
### Rank agreement and scale

Two separate questions. Does the synthetic landscape order genotypes like the
measured one, and does it span the same range?
""")

code("""
m, s = measured[finite], synthetic[finite]
print(f"Spearman  {spearmanr(m, s).statistic:+.3f}")
print(f"Pearson   {np.corrcoef(m, s)[0, 1]:+.3f}")
print(f"std       measured {m.std():.3f}   mutascape {s.std():.3f}   ratio {s.std() / m.std():.2f}")

wild_type_index = tuple(cube.alphabets[p].index(cube.wild_type[p]) for p in cube.positions)
grids = np.indices(measured.shape)
depth_tensor = sum(
    (grids[axis] != wild_type_index[axis]).astype(int) for axis in range(len(cube.positions))
)

by_depth = (
    pd.DataFrame({"depth": depth_tensor[finite], "measured": m, "mutascape": s})
    .groupby("depth")
    .agg(n=("measured", "size"), measured=("measured", "mean"), mutascape=("mutascape", "mean"))
)
by_depth.round(3)
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

axes[0].hist(m, bins=60, alpha=0.6, density=True, label="measured")
axes[0].hist(s, bins=60, alpha=0.6, density=True, label="mutascape")
axes[0].set_xlabel("fitness relative to wild type")
axes[0].set_title("raw scale")
axes[0].legend(frameon=False)

axes[1].hist((m - m.mean()) / m.std(), bins=60, alpha=0.6, density=True, label="measured")
axes[1].hist((s - s.mean()) / s.std(), bins=60, alpha=0.6, density=True, label="mutascape")
axes[1].set_xlabel("standardised fitness")
axes[1].set_title("shape")

axes[2].plot(by_depth.index, by_depth["measured"], "o-", label="measured")
axes[2].plot(by_depth.index, by_depth["mutascape"], "s-", label="mutascape")
axes[2].set_xlabel("mutation depth")
axes[2].set_ylabel("mean fitness")
axes[2].set_title("depth trend")

fig.tight_layout()
""")

md("""
Read the depth column before anything else. The measured landscape falls as
mutations accumulate, down to about -0.57 at depth fifteen. The synthetic one
rises and then flattens. The twenty singles are all mildly beneficial, so a
model built from them alone has no way to learn that CreiLOV breaks when you
stack them.

That is the honest limit of conditioning on singles. Section 9 shows the same
measurement on landscapes where the singles do carry the signal.
""")

md("""
### Walsh spectrum and local optima

The Walsh spectrum splits fitness variance by interaction order. Order one is
additive, everything above it is epistasis. Local optima count the traps a
hill-climbing search can fall into.

An exact optimum count over the full cube is too slow, so both quantities use
the complete binary sub-cube, holding every multi-substitution position at wild
type. That sub-cube is a proper hypercube, which is what the Walsh transform
needs.
""")

code("""
binary_axes = [i for i, p in enumerate(cube.positions) if len(cube.alphabets[p]) == 2]
slicer = tuple(
    slice(None) if i in binary_axes else cube.alphabets[p].index(cube.wild_type[p])
    for i, p in enumerate(cube.positions)
)
sub_measured, sub_synthetic = measured[slicer], synthetic[slicer]
print(f"binary sub-cube: {len(binary_axes)} positions, "
      f"{np.isfinite(sub_measured).sum():,} of {sub_measured.size:,} measured")

rows = {}
for name, tensor in (("measured", sub_measured), ("mutascape", sub_synthetic)):
    spectrum = walsh_decompose(tensor)
    rows[name] = {
        "std": float(np.nanstd(tensor)),
        "additive share": spectrum[1],
        "epistatic share": sum(v for order, v in spectrum.items() if order >= 2),
        "pairwise share": spectrum.get(2, 0.0),
        "order 4+ share": sum(v for order, v in spectrum.items() if order >= 4),
        "local optima": local_optima_from_tensor(tensor)["n_optima"],
    }
pd.DataFrame(rows).round(3)
""")

code("""
fig, ax = plt.subplots(figsize=(5.4, 3.2))
width, orders = 0.38, range(1, len(binary_axes) + 1)
for offset, (name, tensor) in zip(
    (-width / 2, width / 2), (("measured", sub_measured), ("mutascape", sub_synthetic))
):
    spectrum = walsh_decompose(tensor)
    ax.bar([o + offset for o in orders], [spectrum.get(o, 0.0) for o in orders], width, label=name)
ax.set_yscale("log")
ax.set_xlabel("interaction order")
ax.set_ylabel("share of variance")
ax.set_title("Walsh spectrum")
ax.legend(frameon=False)
fig.tight_layout()
""")

md("""
## 7. An adversarial check, and what it says

A classifier two-sample test trains a model to tell measured variants from
synthetic ones on shared features. AUC near 0.5 means they are hard to separate.
AUC near 1.0 means a classifier picks them apart immediately.

Report this whichever way it lands. Mutascape does not claim its raw fitness
distribution matches an assay.
""")

code("""
test = split.test.sample(min(4000, len(split.test)), random_state=SEED)
generated = oracle.evaluate_batch(test.mutations.tolist()) + wild_type_fitness

real_features = completion_fidelity_features(split.train, test).to_numpy()
fake_features = completion_fidelity_features(split.train, test, generated).to_numpy()
groups = np.arange(len(test))

for classifier in ("logistic", "histogram"):
    result = classifier_two_sample_test(
        real_features,
        fake_features,
        real_groups=groups,
        synthetic_groups=groups,
        classifier=classifier,
        n_permutations=0,
        seed=SEED,
    )
    print(f"{classifier:>10}  AUC {result.auc:.3f} +/- {result.auc_std:.3f}   "
          f"accuracy {result.accuracy:.3f}")
""")

md("""
## 8. The benchmark that matters

The reason to build a synthetic landscape is to rank search methods. So test
that directly. Run twelve budget-matched methods on the measured landscape, run
them again on synthetic landscapes, and ask whether the ranking survives.

Kendall tau over the twelve methods measures it. Tau of 1.0 means every ordered
pair agrees, tau of 0.0 means the synthetic ranking carries no information about
the measured one.

Additive, NK and Potts arms are included because they are what a benchmark would
otherwise use.
""")

code("""
space = SearchSpace.from_cube(cube, finite)
features = make_features(space)
print(f"search space: {space.size:,} genotypes over {space.n_sites} sites\\n")

arms = {
    "measured": measured,
    "mutascape": synthetic,
    "mutascape seed 2": dms_pairwise_readout_tensor(
        cube, generator, coupling=COUPLING, seed=SEED + 1
    ),
    "additive": additive_tensor(split.train, cube),
    "nk k=2": nk_tensor(cube, k=2, seed=SEED),
    "potts": potts_tensor(cube, split.train, coupling=0.25, seed=SEED),
}

records = []
for arm, tensor in arms.items():
    values = space.arm_values(tensor)
    start = time.perf_counter()
    for method in METHODS:
        for trial in range(TRIALS):
            records.append({
                "arm": arm,
                "method": method,
                "best": run_method(
                    method, space, values, features, budget=BUDGET, seed=SEED + 137 * trial
                ),
            })
    print(f"{arm:<18} {time.perf_counter() - start:5.1f}s")

scores = pd.DataFrame(records).groupby(["arm", "method"]).best.mean().unstack("arm")[list(arms)]
scores.round(4)
""")

code("""
transfer = pd.DataFrame(
    [
        {
            "arm": arm,
            "kendall tau": kendalltau(scores["measured"], scores[arm]).statistic,
            "pairs preserved": (kendalltau(scores["measured"], scores[arm]).statistic + 1) / 2,
            "top method": scores[arm].idxmax(),
        }
        for arm in scores.columns
        if arm != "measured"
    ]
).set_index("arm")
print(f"top method on the measured landscape: {scores['measured'].idxmax()}")
transfer.sort_values("kendall tau", ascending=False).round(3)
""")

code("""
fig, ax = plt.subplots(figsize=(6.0, 3.6))
order = scores["measured"].sort_values().index
for arm in ("measured", "mutascape", "additive"):
    ax.plot(range(len(order)), scores.loc[order, arm].rank().to_numpy(), "o-", label=arm)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=60, ha="right")
ax.set_ylabel("rank, 12 is best")
ax.set_title(f"method ranking at {BUDGET} queries")
ax.legend(frameon=False)
fig.tight_layout()
""")

md("""
Look at the spread within each column of the score table above. On the measured
landscape the twelve methods separate. On the mutascape arm most of them land on
almost the same value, because the synthetic surface is single-peaked with a
wide plateau near the top, so any method that climbs at all arrives. Tau is
computed over a near-tie, which is why it comes out low and unstable between
seeds.

CreiLOV is the hard case. Run section 9 before drawing a conclusion from this
one number.
""")

md("""
## 9. Where this landscape sits

One landscape is one data point. The same loop over several measured landscapes
shows the spread and puts the target in context.
""")

code('''
SWEEP = ["CreiLOV_fluorescence", "GB1_binding", "eqFP611_red", "HIV1_activity",
         "TrpB3A", "ParD3", "DHFR", "T7"]
SWEEP_TRIALS = 3


def build_generator(landscape):
    """Condition a generator on one landscape's own wild type and singles."""
    parts = mutation_order_split(landscape.variants, max_train_depth=1)
    seq = landscape.wild_type_sequence
    wt = float(parts.train.loc[parts.train.depth == 0, "fitness"].iloc[0])
    frame = pd.DataFrame(
        [
            {"position": p, "wildtype": seq[p - 1], "mutant": a, "fitness": f - wt}
            for mutations, f in parts.train.loc[
                parts.train.depth == 1, ["mutations", "fitness"]
            ].itertuples(index=False, name=None)
            for p, a in mutations.items()
        ]
    )
    return from_dms(frame, wt_sequence=seq), parts


def transfer_tau(name, *, budget=BUDGET, trials=SWEEP_TRIALS):
    """Kendall tau between measured and synthetic method rankings."""
    landscape = LANDSCAPES[name]
    posterior, parts = build_generator(landscape)
    spec = supported_cube(landscape, parts.train)
    real, cover = measured_tensor(landscape, spec)
    keep = np.isfinite(real)
    grid = SearchSpace.from_cube(spec, keep)
    feats = make_features(grid)
    tensors = {
        "measured": real,
        "mutascape": dms_pairwise_readout_tensor(spec, posterior, coupling=COUPLING, seed=SEED),
        "additive": additive_tensor(parts.train, spec),
    }
    means = {
        arm: pd.Series({
            method: float(np.mean([
                run_method(method, grid, grid.arm_values(tensor), feats,
                           budget=budget, seed=SEED + 137 * t)
                for t in range(trials)
            ]))
            for method in METHODS
        })
        for arm, tensor in tensors.items()
    }
    return {
        "landscape": name,
        "protein": landscape.protein,
        "genotypes": int(keep.sum()),
        "coverage": cover,
        "mutascape": kendalltau(means["measured"], means["mutascape"]).statistic,
        "additive": kendalltau(means["measured"], means["additive"]).statistic,
    }


sweep = []
for name in SWEEP:
    if name not in LANDSCAPES:
        print(f"skipped {name}, not on disk")
        continue
    start = time.perf_counter()
    sweep.append(transfer_tau(name))
    print(f"{name:<24} {time.perf_counter() - start:5.1f}s")
sweep = pd.DataFrame(sweep).set_index("landscape")
sweep.round(3)
''')

code("""
fig, ax = plt.subplots(figsize=(6.4, 3.4))
order = sweep.sort_values("mutascape").index
x = np.arange(len(order))
ax.bar(x - 0.2, sweep.loc[order, "mutascape"], 0.4, label="mutascape")
ax.bar(x + 0.2, sweep.loc[order, "additive"], 0.4, label="additive")
ax.axhline(0, color="k", lw=0.8)
ax.axhline(sweep["mutascape"].median(), ls="--", lw=0.9, color="C0", label="mutascape median")
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=60, ha="right")
ax.set_ylabel("Kendall tau vs measured")
ax.legend(frameon=False)
fig.tight_layout()
""")

md("""
## 10. Benchmark your own method

`Campaign` hands a method a budget, a candidate pool and a measurement channel.
The oracle stays private, so a method only learns what it paid for. Write a
function that spends the budget, and `run_campaign` scores it against the pool
optimum. The same function runs unchanged on any oracle in this notebook.
""")

code('''
def uniform(campaign):
    """Spend the whole budget on random plates."""
    while campaign.remaining:
        take = min(campaign.batch_size, campaign.remaining)
        campaign.measure(campaign.sample(take, seed=campaign.round))


def greedy_neighbourhood(campaign):
    """Random opening plate, then plates sharing mutations with the incumbent."""
    campaign.measure(campaign.sample(campaign.batch_size, seed=0))
    while campaign.remaining:
        seen = campaign.observed
        incumbent = set(seen.loc[seen.fitness.idxmax(), "variant"].split("+"))
        pool = campaign.sample(min(3000, campaign.remaining * 20), seed=campaign.round)
        ranked = sorted(pool, key=lambda m: -len(incumbent & set(m.label.split("+"))))
        campaign.measure(ranked[: min(campaign.batch_size, campaign.remaining)])
''')

md("""
A campaign takes any object with `domain`, `evaluate` and `evaluate_batch`, so
the measured landscape can stand in as an oracle and run the same method code.
""")

code('''
class MeasuredOracle:
    """The measured landscape, answering the same calls the synthetic one does."""

    def __init__(self, variants, domain, wild_type_fitness):
        self.domain = domain
        self._values = {
            tuple(sorted(mutations.items())): float(fitness) - wild_type_fitness
            for mutations, fitness in variants[["mutations", "fitness"]].itertuples(
                index=False, name=None
            )
        }

    def evaluate(self, mutations):
        key = tuple(sorted(self.domain.parse(mutations).items()))
        return self._values.get(key, float("nan"))

    def evaluate_batch(self, variants):
        return np.asarray([self.evaluate(variant) for variant in variants], dtype=float)


truth = MeasuredOracle(dataset.variants, oracle.domain, wild_type_fitness)

# Candidates both oracles can answer, so the two runs are comparable.
pool = dataset.variants.loc[dataset.variants.depth >= 2, "mutations"].sample(
    20000, random_state=SEED
).tolist()

outcome = pd.DataFrame([
    {"landscape": arm, "method": name, "budget": budget,
     **run_campaign(fn, target, pool, budget=budget, seed=SEED).to_dict()}
    for budget in (96, 384)
    for arm, target in (("measured", truth), ("mutascape", oracle))
    for name, fn in (("uniform", uniform), ("greedy neighbourhood", greedy_neighbourhood))
]).set_index(["budget", "landscape", "method"])
outcome[["best_found", "optimum", "simple_regret"]].round(4)
''')

md("""
Regret is far lower on the synthetic landscape than on the measured one at the
same budget. The synthetic surface is the easier problem, which is worth knowing
before you quote an absolute success rate from it.

Assay noise changes the score, because `result` reads the true fitness of the
variant a campaign would ship rather than the noisy read that selected it. A
method that chases noise pays for it.
""")

code("""
noisy = pd.DataFrame([
    {"method": name, "noise": level,
     **run_campaign(fn, oracle, pool, budget=384, seed=SEED, noise=level).to_dict()}
    for level in (0.0, 0.25, 0.5)
    for name, fn in (("uniform", uniform), ("greedy neighbourhood", greedy_neighbourhood))
]).set_index(["method", "noise"])
noisy[["best_found", "best_observed", "simple_regret"]].round(4)
""")

md("""
## 11. Change protein

Go back to cell 1, set `TARGET` to another name from the list, and run the
notebook again. Nothing below that cell mentions CreiLOV except the optional
whole-protein section, which skips itself.

For a protein that is not in the corpora, skip the split and pass your own
measurements in. `from_dms` takes a path or a frame, with either
`position, wildtype, mutant, fitness` or `variant, fitness` columns.

```python
oracle = from_dms("my_protein.csv", fasta="my_protein.fasta").draw(seed=0)
oracle["A29H+R60D"]
```

Fitness must be relative to wild type, with wild type at zero. If your scan has
gaps, `from_dms(..., complete="esm2")` calibrates masked ESM2 scores on the
singles you did measure and fills the rest, leaving every measurement exact.
""")

md("""
## What to take away

- The oracle is exact where you measured and lazy everywhere else, so a
  benchmark can span depths no assay reaches.
- A manifest plus a seed reproduce a landscape exactly, and refuse if the source
  data changed.
- Raw fitness distributions do not match an assay, and the classifier test says
  so. Do not use mutascape as a fitness predictor, and do not read an absolute
  success rate off a synthetic landscape.
- Method rankings often survive the substitution, which is the claim the
  benchmark rests on. Section 9 shows how far that varies, from strong transfer
  on GB1 to none on ParD3.
- CreiLOV is a hard case, not a showcase. Twenty singles cannot tell a model
  that the protein breaks when you stack them, and the synthetic landscape rises
  with depth where the measured one falls.
""")


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "markdown" else nbf.v4.new_code_cell(body)
        for kind, body in CELLS
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(notebook, OUTPUT)
    # Ruff formats notebooks, and `just lint` checks this one.
    subprocess.run(["ruff", "format", "--quiet", str(OUTPUT)], check=True)
    print(f"wrote {OUTPUT} with {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
