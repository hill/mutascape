<p align="center">
  <img src="docs/logo_ai_image.png" alt="mutascape" width="500">
</p>

# mutascape

Synthetic combinatorial protein fitness landscapes with tuneable, structurally informed epistasis.

Built on real single-site DMS data, with epistatic topology derived from 3D protein structure. Control landscape ruggedness with a single parameter (`alpha`), and query arbitrary multi-mutant fitness on the fly.

## Installation

Requires Python 3.12+.

```bash
uv sync              # core dependencies
uv sync --extra dev  # + pytest/ruff
```

## Quick start

```python
from mutascape import Landscape, ContactMap, Mutant, RandomEpistasis

# Contact map from PDB file or predicted from sequence (ESMFold)
contacts = ContactMap.from_pdb("structure.pdb", threshold=8.0)
contacts = ContactMap.from_sequence("MAEQLA...", threshold=8.0)

# Build landscape from DMS data
landscape = Landscape.from_dms(
    dms_path="dms_data.csv",
    wt_sequence="MAEQLA...",
    contact_map=contacts,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    alpha=1.0,
)

# Query fitness (relative to wild type: 0.0 = WT, positive = better, negative = worse)
landscape["A29H"]
landscape[Mutant("A29H") + Mutant("R60D")]

# Stream variants from the combinatorial landscape
for v in landscape.generate(n_mutations=3, n=1000, seed=42):
    print(v.mutations, v.fitness)

# Bulk sample to DataFrame
df = landscape.to_dataframe(n_variants=10000, n_mutations=(1, 5), seed=42)

# Characterise and plot
landscape.stats(n_samples=5000, n_mutations=3)
landscape.plot_epistasis()
landscape.plot_alpha_sweep()
```

### Full-protein landscapes with pLM scoring

When DMS data only covers a subset of positions, use ESM2 masked marginal scoring to predict single-site effects for the rest:

```python
from mutascape import Landscape, ContactMap, ESM2Scorer, RandomEpistasis

scorer = ESM2Scorer()

# Pure pLM landscape (no DMS data needed)
landscape = Landscape.from_plm(
    sequence="MAEQLA...",
    contact_map=contacts,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    scorer=scorer,
)

# Hybrid: DMS where available, pLM for remaining positions
landscape = Landscape.from_dms(
    dms_path="dms_data.csv",
    wt_sequence="MAEQLA...",
    contact_map=contacts,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    scorer=scorer,  # fills unscreened positions
)
```

## Why mutascape?

Single-site DMS benchmarks only test methods on variants one mutation from wild type. Real protein engineering stacks mutations, where epistasis creates rugged landscapes with local optima. Existing synthetic landscape approaches leave a gap:

- **SLIP** (Thomas et al. 2022) derives everything from MSA-fitted Potts models. Requires an MSA, and single-site effects come from evolutionary inference rather than experimental measurements.
- **NK landscapes** (Sandhu et al. 2025) are abstract and have no biophysical grounding. Short sequences, reduced alphabets.

mutascape fills this gap: real DMS data for single-site effects, 3D structural contacts for epistatic topology, full 20-amino-acid alphabet, full-length proteins.

## Docs

- [Design document](docs/design.md) - theoretical background, fitness function, epistasis models, positioning vs prior work, pLM scoring literature
