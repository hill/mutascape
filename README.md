<p align="center">
  <img src="docs/logo_ai_image.png" alt="mutascape" width="500">
</p>

# mutascape

Synthetic combinatorially complete protein fitness landscapes with tuneable, structurally informed epistasis.

Real single-site DMS effects, 3D structural contacts, and a swappable pairwise epistasis model, all wrapped behind a `Landscape` oracle that you can index with `Mutant` objects to get fitness for any multi-mutant. Tune ruggedness with one knob (`alpha`); pick the epistasis story you want (random, sign-consistent, MSA-derived). Wrap the latent score in a thermodynamic non-linearity and Gaussian noise when you want a more realistic readout.

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

`alpha` is a global multiplier on the pairwise term. `alpha=0` recovers the purely additive landscape; raising it progressively introduces sign epistasis and local optima.

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

Add a global non-linearity and measurement noise on top of the latent score:

```python
from mutascape import ThermodynamicSigmoid

nonlin = ThermodynamicSigmoid.fit_from_singles(
    landscape.single_site_effects,
    combinatorial_upside=2.0,
)

landscape = Landscape.from_dms(
    ...,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    nonlinearity=nonlin,
    noise_sigma=0.15,
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

At the shared default `noise_sigma=0.3` and `sigma=0`, these reach `walsh_L1=0.317` / `fitness_W1=0.036` on GB1 and `walsh_L1=0.107` / `fitness_W1=0.028` on TrpB, against the complete factorial real data.

## Reproducing the published outputs

```bash
./.venv/bin/python scripts/rebuild_reference_outputs.py
./.venv/bin/python scripts/run_gfp_supplemental_validation.py
./.venv/bin/pytest -q tests/test_reference_configs.py
```

The rebuild script regenerates the GB1 / TrpB sweep CSVs and the headline figures under `notebooks/validation/output_*`.

## Known limitations

- GB1 still shows an order-1 / order-3 Walsh mismatch.
- The current default pipeline under-generates local optima on both GB1 and TrpB.
- Sparse full-protein GFP remains smoother than the real landscape under the same default settings; the structural pairwise term (`alpha > 0`), MSA-derived couplings, and Contact-NK higher-order terms are the supported routes for closing this gap.
- Functional-contact overrides should only be used when a ligand/cofactor or annotated active-site mechanism justifies them.

## Docs

- [Project status](docs/project_status.md)
- [Experiment log](docs/experiments_log.md), chronological history (older entries are superseded by project status)
- [Thermodynamic nonlinearity design](docs/nonlinearity_design.md), current mechanism with historical validation notes
- [GB1 validation analysis](docs/gb1_validation_analysis.md), historical alpha-era analysis
- [Design document](docs/design.md), historical design background
