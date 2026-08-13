<p align="center">
  <img src="docs/logo_ai_image.png" alt="mutascape" width="500">
</p>

# mutascape

Synthetic combinatorially complete protein fitness landscapes with tuneable, structurally informed epistasis.

Mutascape combines single-site DMS effects, structural contacts, and a swappable pairwise epistasis model. A `Landscape` oracle returns fitness for any `Mutant` combination. Tune ruggedness with `alpha`, then choose random, sign-consistent, or MSA-derived epistasis. Add global epistasis and Gaussian noise for a more realistic readout.

## Installation

Requires Python 3.12+.

```bash
uv sync              # core dependencies
uv sync --extra dev  # + pytest/ruff
```

## Quick start

```python
from mutascape import Landscape, ContactMap, Mutant, RandomEpistasis

# Contact map from a PDB file (or predicted from sequence with ESMFold)
contacts = ContactMap.from_pdb("structure.pdb", threshold=8.0)

# Build a synthetic landscape grounded in DMS singles + structural contacts
landscape = Landscape.from_dms(
    dms_path="dms_data.csv",
    wt_sequence="MAEQLA...",
    contact_map=contacts,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    alpha=1.0,
)

# Query the oracle by mutant label or composed Mutant objects
landscape["A29H"]
landscape[Mutant("A29H") + Mutant("R60D")]
landscape[Mutant("A29H") + Mutant("R60D") + Mutant("K93T")]
```

## Posterior whole-protein completion

The research API learns interaction priors from complete empirical landscapes. It then conditions a whole-protein oracle on WT, singles, and optional doubles.

```python
from mutascape import LandscapePrior, PosteriorLandscape
from mutascape.benchmarks import (
    iter_protein_holdouts,
    load_combingym_landscapes,
    mutation_order_split,
)

datasets = load_combingym_landscapes(".mutascape/benchmark-data/combingym")
fold = next(fold for fold in iter_protein_holdouts(datasets) if fold.protein == "GB1")
prior = LandscapePrior.fit(list(fold.train), max_order=3)

target = next(dataset for dataset in fold.test if dataset.name == "GB1_binding")
split = mutation_order_split(target.variants, max_train_depth=1)
posterior = PosteriorLandscape.condition(
    target.wild_type_sequence,
    split.train,
    prior,
)

# One seed defines one stable whole-protein oracle.
oracle = posterior.sample(seed=42)
fitness = oracle.evaluate({39: "A", 40: "W", 54: "F"})

# An ensemble exposes uncertainty in the unmeasured combination.
summary = posterior.predict(
    [{39: "A", 40: "W", 54: "F"}],
    n_samples=100,
    seed=42,
)
```

Fetch the pinned benchmark sources and run grouped evaluation with these commands.

```bash
just benchmark-data
just benchmark-completion \
  --ssmula-root .mutascape/benchmark-data/ssmula \
  --combingym-root .mutascape/benchmark-data/combingym
```

See [the benchmark protocol](docs/benchmark_protocol.md) for splits, metrics, and current pilot status.

### Posterior web studio

The studio exposes the posterior as a queryable model. One seed fixes a stable oracle draw. An ensemble reports uncertainty for unseen combinations.

The bundled `combinatorial-v1` prior pools 30 complete landscapes from SSMuLA and CombinGym. Its metadata records source revisions, proteins, and fitted scales.

Upload DMS data with either supported CSV schema. Both schemas require a matching FASTA sequence.

```csv
position,wildtype,mutant,fitness
39,V,A,-0.42
```

Variant-level files must include `WT`. They may also include measured combinations.

```csv
variant,fitness
WT,1.0
V39A,0.58
V39A:D40W,0.91
```

Run the studio and API locally with `just dev`. Posterior routes are available at `/api/v1/priors`, `/api/v1/posterior-queries`, and `/api/v1/posterior-previews`.

## Loading landscapes

### From a PDB or mmCIF structure

```python
contacts = ContactMap.from_pdb("structure.pdb", threshold=8.0, chain_id="A")
```

### From an ESMFold prediction (no PDB available)

```python
contacts = ContactMap.from_sequence("MAEQLA...", threshold=8.0)  # cached locally
```

### From a precomputed distance matrix

```python
import numpy as np

distances = np.load("distances.npy")  # shape (L, L)
contacts = ContactMap.from_distance_matrix(distances, positions=range(1, 120), threshold=8.0)
```

### Hybrid DMS with ESM2 fill-in

```python
from mutascape import ESM2Scorer

scorer = ESM2Scorer()
landscape = Landscape.from_dms(
    dms_path="partial_dms.csv",
    wt_sequence="MAEQLA...",
    contact_map=contacts,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    scorer=scorer,  # fills unscreened positions with masked-marginal ESM2 scores
)
```

### Pure pLM landscape (no DMS)

```python
landscape = Landscape.from_plm(
    sequence="MAEQLA...",
    contact_map=contacts,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    scorer=scorer,
)
```

## Querying the oracle

```python
# Index by label or by Mutant
landscape["A29H"]
landscape[Mutant("A29H")]

# Compose mutants with + (and remove with -)
m = Mutant("A29H") + Mutant("R60D")
landscape[m]
m.depth         # 2
dict(m)         # {29: 'H', 60: 'D'}

# Stream variants at a fixed mutation depth
for v in landscape.generate(n_mutations=3, n=1000, seed=42):
    print(v.mutations, v.fitness)

# Bulk sample to a pandas DataFrame across a depth range
df = landscape.to_dataframe(n_variants=10_000, n_mutations=(1, 5), seed=42)

# Compare actual against the additive prediction
actual   = landscape.evaluate({29: "H", 60: "D"})
additive = landscape.additive_fitness({29: "H", 60: "D"})

# Cheap descriptive stats
landscape.stats(n_samples=5000, n_mutations=3)
# -> {'mean_fitness': ..., 'additive_correlation': ...,
#     'mean_epistatic_deviation': ..., 'fraction_sign_epistasis': ...}
```

## Tuning ruggedness

`alpha` is a global multiplier on the pairwise term. `alpha=0` recovers the purely additive landscape. Raising it progressively introduces sign epistasis and local optima.

```python
smooth   = Landscape.from_dms(..., epistasis=RandomEpistasis(sigma=0.3, seed=42), alpha=0.0)
moderate = Landscape.from_dms(..., epistasis=RandomEpistasis(sigma=0.3, seed=42), alpha=1.0)
rugged   = Landscape.from_dms(..., epistasis=RandomEpistasis(sigma=0.3, seed=42), alpha=2.5)
```

Swap the epistasis model itself to change the *character* of the interactions, not just their magnitude:

```python
from mutascape import SignConsistentEpistasis, MSACouplingsEpistasis

# Sign-consistent: beneficial+beneficial and deleterious+deleterious cooperate
sc = Landscape.from_dms(..., epistasis=SignConsistentEpistasis(sigma=0.3, seed=42))

# Evolutionarily informed: precomputed couplings from plmDCA / EVcouplings
msa = Landscape.from_dms(
    ...,
    epistasis=MSACouplingsEpistasis(
        coupling_matrix=J,                    # shape (L, L, 20, 20)
        position_order=tuple(positions),
        scale=0.5,
    ),
)
```

Add a global-epistasis function, genotype-stable measurement noise, and an optional censored readout texture on top of the latent score:

```python
from mutascape import GlobalEpistasisSigmoid

nonlin = GlobalEpistasisSigmoid.fit_from_singles(
    landscape.single_site_effects,
    combinatorial_upside=2.0,
)

landscape = Landscape.from_dms(
    ...,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    nonlinearity=nonlin,
    noise_sigma=0.15,
    readout_texture_sigma=0.005,   # tiny post-readout texture on the floor
    readout_texture_order=3,
    seed=42,
)
```

## Reference benchmark landscapes

For benchmarking work, mutascape ships two named reference configurations calibrated against the complete factorial GB1 and TrpB landscapes. Use these when you want a published, regression-tested benchmark landscape rather than building one from scratch.

```python
from mutascape.validation.reference_configs import GB1_HEADLINE, TRPB_HEADLINE

gb1  = GB1_HEADLINE.build_landscape()
trpb = TRPB_HEADLINE.build_landscape()

gb1["V39A"]
trpb[Mutant("V183A") + Mutant("F184I")]
```

At the shared defaults, GB1 reaches `walsh_L1=0.317`, `fitness_W1=0.036`, and `n_optima=200`. TrpB reaches `0.105`, `0.027`, and `843` for those metrics. The real landscapes have `n_optima=182` and `802`.

## Reproducing the published outputs

```bash
./.venv/bin/python scripts/rebuild_reference_outputs.py
./.venv/bin/python scripts/run_gfp_supplemental_validation.py
./.venv/bin/pytest -q tests/test_reference_configs.py
```

The rebuild script regenerates the GB1 / TrpB sweep CSVs and the headline figures under `notebooks/validation/output_*`.

## Known limitations

- GB1 still shows an order-1 / order-3 Walsh mismatch.
- Complete combinatorial references calibrate the censored readout texture. For unvalidated proteins, treat it as a tunable topology parameter rather than a universal default.
- Sparse full-protein GFP remains smoother than the real landscape under the same default settings. The structural pairwise term (`alpha > 0`), MSA-derived couplings, and Contact-NK higher-order terms are the supported routes for closing this gap.
- Functional-contact overrides should only be used when a ligand/cofactor or annotated active-site mechanism justifies them.
- Posterior intervals describe uncertainty under the fitted prior. They do not prove that an unmeasured landscape matches biology.

## Docs

- [Project status](docs/project_status.md)
- [Global-epistasis function design](docs/nonlinearity_design.md), current mechanism with historical validation notes
- [Design document](docs/design.md), historical design background
