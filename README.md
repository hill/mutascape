<p align="center">
  <img src="docs/logo_ai_image.png" alt="mutascape" width="500">
</p>

# mutascape

Synthetic combinatorial protein fitness landscapes for benchmarking MLDE and ALDE methods.

Mutascape builds benchmark landscapes from real single-site DMS effects plus protein structure. The current publication reference pipeline is intentionally simple: additive DMS pre-fitness, a fitted censored thermodynamic sigmoid, and Gaussian noise. Optional structural pairwise couplings still exist in the package, but they are *off by default* because they worsened the headline validation metrics on both GB1 and TrpB.

For larger proteins, Mutascape can still use structural and protein-language-model priors: contact maps can come from PDB/mmCIF structures, AlphaFold-style predicted structures saved as structure files, precomputed distance matrices, or the built-in ESMFold sequence-to-structure route; missing single-site effects can be filled with masked-marginal ESM2 scores. That is an enabled workflow, not the headline validated result.

## Installation

Requires Python 3.12+.

```bash
uv sync
uv sync --extra dev
```

## Quick start

Use the frozen publication configs when you want the validated reference pipeline:

```python
from mutascape.validation.reference_configs import GB1_HEADLINE, TRPB_HEADLINE

gb1 = GB1_HEADLINE.build_landscape()
trpb = TRPB_HEADLINE.build_landscape()

gb1["V39A"]
trpb["V183A:F184I"]
```

For custom builds, go through the shared validation builder so the paper-critical parameters are explicit:

```python
from mutascape.validation.calibration import build_landscape

landscape = build_landscape(
    "data/gb1/gb1.csv",
    "data/gb1/gb1_distances.npz",
    "data/gb1/gb1_wt.fasta",
    alpha=1.5,
    sigma=0.0,
    nonlinearity="fit",
    combinatorial_upside=3.0,
    scale=1.0,
    floor_hard=-1.0,
    noise_sigma=0.3,
    seed=42,
)
```

## Frozen reference story

The publication draft is built around two named reference configurations in `src/mutascape/validation/reference_configs.py`:

- `gb1_headline`: GB1 complete combinatorial landscape, `sigma=0`, `noise_sigma=0.3`, `scale=1.0`, `floor_hard=-1.0`
- `trpb_headline`: TrpB complete combinatorial landscape, `sigma=0`, `noise_sigma=0.3`, `scale` fit from singles, `floor_hard=singles_min`

Headline validation at the frozen settings:

- GB1: `walsh_L1 = 0.317`, `fitness_W1 = 0.036`
- TrpB: `walsh_L1 = 0.107`, `fitness_W1 = 0.028`

These are the main publication metrics. Per-order Walsh composition and `n_optima` are still reported, but as secondary diagnostics.

## Reproducibility

Reference outputs can be rebuilt without notebooks:

```bash
./.venv/bin/python scripts/rebuild_reference_outputs.py
./.venv/bin/python scripts/run_gfp_supplemental_validation.py
./.venv/bin/pytest -q tests/test_reference_configs.py
../mutantopt/.venv/bin/python scripts/run_mutantopt_pilot.py \
  --config gb1_headline --noise-sigmas 0.0 0.3
```

The rebuild script regenerates the GB1/TrpB sweep CSVs and the headline figures in `notebooks/validation/output_*`.
The pilot runner exports compact 4-position benchmark datasets and runs a first `mutantopt` comparison over a `noise_sigma` sweep.

## What the package is for

Mutascape is now framed as a *benchmark-generation tool*, not a digital twin of any one protein. The core use case is:

1. Freeze a reference landscape generator grounded in DMS and structure.
2. Validate it against complete combinatorial landscapes with `walsh_L1` and `fitness_W1`.
3. Sweep `noise_sigma` to make the benchmark harder.
4. Run MLDE / ALDE methods against that sweep.

## Larger proteins and priors

The validated reference configs use complete four-site GB1 and TrpB landscapes because they allow exact Walsh spectra and local-optima counts. The package itself is not restricted to four sites:

- `ContactMap.from_pdb(...)` loads experimental PDB/mmCIF structures or predicted structures from AlphaFold/ColabFold/ESMFold if they are available as local files.
- `ContactMap.from_distance_matrix(...)` loads prepared distance matrices for any selected position set.
- `ContactMap.from_sequence(...)` calls the ESMFold API and caches the predicted PDB before extracting contacts.
- `ESM2Scorer` can fill unscreened single-site effects when a DMS scan covers only part of a protein.

Use these routes to construct larger benchmark candidates, then validate them where multi-mutant data exists. The current paper does not claim that ESM or AlphaFold priors alone solve the GFP/full-protein smoothness gap.

## Known limitations

- GB1 still shows an order-1 / order-3 Walsh mismatch.
- The current pipeline under-generates local optima on both GB1 and TrpB.
- Sparse full-protein GFP remains smoother than the real landscape under the same pipeline.
- Functional-contact overrides should only be used when a ligand/cofactor or annotated active-site mechanism justifies them.

## Docs

- [Project status](docs/project_status.md)
- [Experiment log](docs/experiments_log.md) - chronological history; older entries are superseded by project status
- [Thermodynamic nonlinearity design](docs/nonlinearity_design.md) - current mechanism with historical validation notes
- [GB1 validation analysis](docs/gb1_validation_analysis.md) - historical alpha-era analysis
- [Design document](docs/design.md) - historical design background, not the canonical method spec
