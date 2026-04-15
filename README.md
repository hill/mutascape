# mutascape

Synthetic combinatorial protein fitness landscapes with tunable, structurally informed epistasis. Built on real single-site DMS data from [mutantopt](../mutantopt/).

## Motivation

Single-site DMS benchmarks only test methods on variants one mutation from wild type. Real protein engineering stacks mutations, where epistasis (non-additive interactions between sites) creates rugged landscapes with local optima. No existing MLDE benchmark systematically controls for epistatic structure.

mutascape generates synthetic combinatorial landscapes that:

- Start from real single-site fitness effects (DMS ground truth)
- Add pairwise (and optionally higher-order) epistatic interactions
- Use predicted 3D structure (AlphaFold2/ESMFold) to make epistasis structurally plausible
- Allow tuning of epistatic strength, sparsity, and sign distribution

## Core idea

A combinatorial fitness function decomposes as:

```
f(x) = sum_i h_i(x_i) + alpha * sum_{i<j} J_ij(x_i, x_j) + noise
```

Where:

- `h_i(x_i)` = single-site DMS fitness effect at position i for amino acid x_i (real data)
- `J_ij(x_i, x_j)` = pairwise epistatic coupling between positions i and j
- `alpha` = global epistatic strength parameter (alpha=0 recovers the additive landscape)
- Epistatic topology is derived from structural contact maps

## Structural priors for epistasis

Residue pairs in 3D contact (C-beta distance < 8A in predicted structure) are assigned non-zero epistatic couplings. Distant pairs are zero or near-zero. This produces landscapes where:

- Active site / binding interface mutations interact strongly
- Surface mutations far from functional regions are approximately additive
- The ruggedness correlates with the density of structural contacts in the mutated region

Contact maps sourced from:

- **ESMFold** predictions (fast, no MSA required, good for single-chain)
- **AlphaFold2** predictions via ColabFold or AlphaFold DB (higher accuracy, slower)

## Epistatic coupling models

Several options for generating `J_ij`, from simple to biophysically grounded:

1. **Random sparse**: draw couplings from N(0, sigma) for contacting pairs, zero elsewhere
2. **Sign-consistent**: beneficial single-site mutations at contacting positions have positive coupling (cooperative), deleterious ones have negative (destabilising)
3. **Potts-derived**: extract couplings from inverse Potts models (e.g. plmDCA, GREMLIN) fit to MSAs of homologous sequences - reflects real evolutionary covariation
4. **Rosetta/FoldX-calibrated**: use physics-based energy functions to estimate pairwise interaction energies, then scale to match DMS fitness units

## Installation

```bash
uv sync                     # core dependencies
uv sync --extra structure    # + ESMFold for structure prediction
uv sync --extra viz          # + matplotlib/seaborn
uv sync --extra dev          # + pytest/ruff
```

## Quick start

```python
from mutascape import Landscape, ContactMap, RandomEpistasis

# Load single-site DMS data and structure
contacts = ContactMap.from_pdb("data/structures/creilov.pdb", threshold=8.0)
landscape = Landscape.from_dms(
    dms_path="data/dms/creilov.csv",
    contacts=contacts,
    epistasis=RandomEpistasis(strength=0.5, seed=42),
)

# Evaluate a multi-mutant
fitness = landscape.evaluate({"A57G": True, "L72V": True})

# Generate n random multi-site variants and their fitness
variants, scores = landscape.sample(n=1000, n_mutations=3)
```

## Planned experiments

1. **Method stress-testing**: run mutantopt benchmarks across a grid of epistatic strengths. At what alpha does each method break down?
2. **Encoding importance vs epistasis**: does the PLM embedding advantage over one-hot grow with epistatic strength?
3. **Exploration-exploitation balance**: do exploration-heavy strategies (UCB, Thompson Sampling) gain more on rugged landscapes?
4. **Autoresearch on synthetic landscapes**: let an LLM agent discover epistasis-aware acquisition strategies
5. **Landscape difficulty prediction**: train a meta-model to predict method rankings from pilot screen features

## Calibration

Synthetic landscapes are validated against known combinatorial DMS datasets (GB1 4-site, TEM-1 combinatorial) by matching fraction of sign epistasis, magnitude epistasis, and number of local optima.

## Project structure

```
mutascape/
  src/mutascape/
    __init__.py
    landscape.py       # Core landscape generator
    contacts.py        # Structure prediction and contact map extraction
    epistasis.py       # Coupling models (random, sign-consistent, Potts, etc.)
    calibration.py     # Validate against real combinatorial data
  notebooks/
  experiments/
  data/
    structures/        # Predicted structures (PDB/mmCIF)
    contact_maps/      # Extracted contact matrices
    dms/               # Single-site DMS data
  tests/
```

## Relationship to thesis

Extends the mutantopt benchmarking work from single-site to combinatorial landscapes. The thesis narrative: single-site benchmarks reveal that landscape properties predict method performance; synthetic combinatorial landscapes let us systematically vary those properties (especially epistasis) and test methods in regimes inaccessible to real DMS data.
