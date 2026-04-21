# Project status

Checkpoint summary as of 2026-04-21. For the chronological experiment trail see `experiments_log.md`; for theoretical background see `design.md`; for the thermodynamic-layer math see `nonlinearity_design.md`; for the deeper GB1-specific analysis see `gb1_validation_analysis.md`; for citations and paper context see `references.md`.

## Goal

Produce a **synthetic combinatorial protein fitness-landscape generator for benchmarking ML-assisted directed evolution**. The current publication framing is deliberately narrow:

- Mutascape is a **benchmark-generation tool**, not a digital twin of one protein.
- The headline metrics are `walsh_L1` and `fitness_W1`.
- Per-order Walsh composition, `n_optima`, and SSMuLA-style descriptive attributes are secondary diagnostics.

## Current headline method

The validated reference pipeline is:

1. Additive single-site effects from DMS.
2. A fitted thermodynamic sigmoid with optional censoring via `floor_hard`.
3. Gaussian noise with `noise_sigma` as the primary ruggedness parameter.

Important consequence: **pairwise random couplings are off by default**. On both GB1 and TrpB, setting `sigma > 0` worsened the headline validation metrics. The best current default is therefore `sigma = 0`, fitted censored nonlinearity, and `noise_sigma = 0.3`.

The publication-critical parameter sets are frozen in `src/mutascape/validation/reference_configs.py`.

## Larger proteins, structure, and pLM priors

The current code still supports structure- and pLM-assisted construction for larger proteins, but that support is separate from the frozen headline validation.

- Structural priors enter through `ContactMap`: load PDB/mmCIF files with `from_pdb`, load prepared distance matrices with `from_distance_matrix`, or predict a structure from sequence with `from_sequence` via ESMFold.
- AlphaFold/ColabFold structures are supported when saved as local PDB/mmCIF files and passed to `ContactMap.from_pdb`; there is no built-in AlphaFold runner.
- ESM2 masked-marginal scoring is available through `ESM2Scorer` and can fill unscreened single-site effects when DMS is incomplete.
- These routes are enabled workflows for larger benchmark candidates, not yet headline evidence that large-protein landscapes are quantitatively matched.

## Frozen reference configurations

### GB1 (`gb1_headline`)

- `alpha = 1.5`
- `sigma = 0.0`
- `noise_sigma = 0.3`
- `scale = 1.0`
- `floor_hard = -1.0`
- `combinatorial_upside = 3.0`

Headline metrics:

- `walsh_L1 = 0.317`
- `fitness_W1 = 0.036`
- real `n_optima = 182`
- synthetic `n_optima = 3`

Order-wise Walsh fractions at the frozen setting:

- real: `(0.186, 0.429, 0.308, 0.077)`
- synthetic: `(0.304, 0.470, 0.155, 0.072)`

### TrpB (`trpb_headline`)

- `alpha = 1.5`
- `sigma = 0.0`
- `noise_sigma = 0.3`
- `scale = 0.553` (fit from singles)
- `floor_hard = singles_min ~= -0.421`
- `combinatorial_upside = 3.0`

Headline metrics:

- `walsh_L1 = 0.107`
- `fitness_W1 = 0.028`
- real `n_optima = 802`
- synthetic `n_optima = 23`

Order-wise Walsh fractions at the frozen setting:

- real: `(0.255, 0.356, 0.234, 0.155)`
- synthetic: `(0.242, 0.410, 0.201, 0.148)`

## Supplementary GFP rerun

The sparse full-protein GFP rerun is now complete under the same headline architecture:

- `alpha = 1.5`
- `sigma = 0.0`
- fitted censored thermodynamic sigmoid
- `floor_hard = singles_min`
- `noise_sigma = 0.3`

What it shows:

- The synthetic landscape reproduces the *qualitative* depth trends: additive correlation decreases, sign epistasis increases, and deviation magnitudes broaden with mutation depth.
- It remains too smooth to serve as a headline validation result.

Representative values:

- depth 2 additive correlation: real `0.900`, synthetic `0.936`
- depth 7 additive correlation: real `0.419`, synthetic `0.795`
- roughness-to-slope ratio: real `0.951`, synthetic `0.605`

Interpretation: GFP is useful supplementary evidence for generalization, but the paper should stay centered on GB1 + TrpB.

## What is working

- **The core package is real and tested.** DMS ingestion, contact maps, nonlinearity, validation metrics, and figure generation are implemented.
- **The reference pipeline is frozen in code.** Named builders now exist for the publication configs.
- **Regression protection is in place.** `tests/test_reference_configs.py` checks the GB1 and TrpB headline metrics against tolerance bands.
- **Notebook dependence is reduced.** `scripts/rebuild_reference_outputs.py` rebuilds the paper figures from prepared data.
- **The paper story is coherent.** The manuscript now matches the code and figures instead of the older alpha-centric draft.

## What is not working

- **Local-optima counts remain badly under-generated.** This is the clearest current failure mode.
- **GB1 still has an order-1 / order-3 mismatch.** Noise improves the scalar match but does not fully fix the order composition.
- **Sparse full-protein landscapes remain harder.** GFP is still smoother than real under the same pipeline.
- **Functional coupling is not always visible in raw contact maps.** TrpB needs an explicit policy for justified contact overrides.

## Modeling rule for contact overrides

Raw spatial contacts and functional coupling are not identical. The project now adopts a stricter rule:

- only override the structural contact graph when an independent mechanistic justification exists, such as a shared ligand/cofactor or an annotated active-site mechanism

This keeps TrpB defensible while preventing arbitrary hand-tuning.

## Recommended next steps

1. Run a small MLDE / ALDE pilot on the frozen pipeline with 2-3 methods and a `noise_sigma` sweep.
2. Keep `noise_sigma` as the first ruggedness dial; only add `sigma_nk` if it fixes the order-3 mismatch without damaging `fitness_W1`.
3. Tighten the paper discussion after one more pass over wording and figure density.

An initial GB1 quick pilot now exists via `scripts/run_mutantopt_pilot.py`; it exports compact benchmark datasets and writes results under `notebooks/benchmark_pilot/`.

## Reproducibility hooks

- `src/mutascape/validation/reference_configs.py` - frozen publication configs
- `tests/test_reference_configs.py` - metric regression tests
- `scripts/rebuild_reference_outputs.py` - rebuild GB1/TrpB headline outputs
- `scripts/run_gfp_supplemental_validation.py` - rerun GFP under the current headline pipeline
- `scripts/run_mutantopt_pilot.py` - export compact pilot datasets and run a small mutantopt benchmark sweep

## Status flags

- [ready] nonlinearity module + censored thermodynamic fit
- [ready] GB1 reference config and figure outputs
- [ready] TrpB reference config and figure outputs
- [ready] GFP supplementary rerun under the current pipeline
- [ready] paper draft rewritten around the frozen reference story
- [ready] small MLDE / ALDE pilot benchmark hook on `noise_sigma`
- [pending] hybrid `noise_sigma + sigma_nk` exploration
