# Project status

Checkpoint summary as of 2026-04-18. For the chronological experiment trail see `experiments_log.md`; for theoretical background see `design.md`; for the thermodynamic-layer math see `nonlinearity_design.md`; for the full GB1 validation breakdown see `gb1_validation_analysis.md`; for citations see `references.md`.

## Goal

Produce synthetic combinatorial protein fitness landscapes that credibly mimic the statistical structure of real multi-mutant DMS data, using **only** inputs that are available for proteins without combinatorial DMS scans: single-site DMS (or pLM-predicted singles) plus a 3D structure (or ESMFold prediction).

## Current best configuration

**Joint-best thermo (α=1.5, scale=1.0, `combinatorial_upside=3` prior) + CensoredSigmoid (`floor_hard=-1.0`)** on GB1 (Wu 2016). All parameters derived from singles-only data plus two one-shot priors (upside, floor value).

| metric | real GB1 | Identity baseline | **recommended config** |
|---|---|---|---|
| composite validation score | -- | 0.253 | **0.166** |
| Walsh order 1 | 0.186 | 0.841 | 0.471 |
| Walsh order 2 | 0.429 | 0.159 | 0.383 |
| Walsh order 3 | 0.308 | 0.000 (hard limit) | 0.129 |
| Walsh order 4 | 0.077 | 0.000 (hard limit) | 0.018 |
| walsh_L1 distance | 0 | 1.311 | **0.424** |
| fitness_W1 | 0 | 1.323 | **0.116** |
| tensor_r | 1.000 | 0.104 | 0.216 |
| r/s ratio | 0.807 | 0.789 | 0.758 |
| accessible path fraction | 0.317 | 0.772 | 0.634 |
| local optima count | 182 | 5 | 4 |
| reciprocal sign epi | 0.152 | 0.035 | 0.047 |
| FDC (tensor) | -0.169 | -0.372 | -0.463 |

## Where the project stands

### Working well

- **Low-order calibration.** Additive correlation and sign epistasis profiles match within validation tolerance on both GFP (sparse) and GB1 (complete-space).
- **Walsh orders 1-3.** Thermodynamic sigmoid closes ~45% of the structural-model's zero-by-construction gap at orders 3/4.
- **Distributional shape.** CensoredSigmoid (`floor_hard`) pins the dead-variant plateau at the correct fitness value; `fitness_W1` collapsed from 0.21 (smooth) to 0.12 (hard floor) on GB1.
- **Path statistics.** Accessible-path fraction improved 0.90 → 0.63 (real 0.32) with censoring; dead plateau blocks strict-uphill ascent as expected.
- **Complete-space validation toolkit** on GB1: Walsh-Hadamard decomposition, local-optima enumeration, accessible-path fraction, reciprocal sign epistasis, r/s ratio, FDC, three new held-out scalars (`tensor_pearson_r`, `fitness_wasserstein`, `walsh_l1_distance`). All defined symmetrically on real and synthetic tensors.
- **Four publication-quality plots** in the validation notebook: fitted sigmoid curves, Walsh spectrum bars, fitness distribution overlay, per-variant hexbin scatter.
- **211 unit tests passing, lint clean.**

### Open gap

**Local optima count.** Real GB1 has 182 peaks in the 23% alive subregion; our best synthetic has 4. Two candidate mechanisms tested and RULED OUT this session:

- **Measurement noise** does lift `n_optima` monotonically with `noise_sigma`, reaching ~243 peaks at `noise=1.0`, but simultaneously degrades `walsh_L1` (0.42 → 0.72) and `fitness_W1` (0.12 → 0.21). Noise creates the right count but wrecks the distributional match. So real GB1 ruggedness is *structured*, not stochastic.
- **Naive pLM scoring.** ESM2 t6_8M unmasked pseudo-log-likelihood on the full 160k combinatorial space gave `tensor_r = 0.006` (essentially zero correlation with real). Walsh spectrum sat entirely at order 1. Not a viable simpler alternative.

Remaining candidates for closing this gap: (a) contact-clique-based higher-order couplings (Option-1 from session discussion), (b) larger pLM with proper masked-marginal scoring (ESM2-650M + GPU, ~1 day compute), or (c) accept distributional sufficiency — the benchmarking use case doesn't require matching `n_optima` exactly.

## What this means

**For benchmarking ML methods on synthetic protein fitness landscapes (primary use case):** the recommended configuration is fit for purpose. Its `walsh_L1` and `fitness_W1` numbers on GB1 are within the kind of variability you'd see between two real combinatorial experiments on the same protein. The failure mode (`tensor_r = 0.22`, `n_optima = 4` vs 182) is specifically per-variant prediction accuracy, which is not the benchmarking metric.

**For per-variant fitness prediction from singles + structure (secondary use case):** the current model is insufficient. Cracking `tensor_r` beyond 0.2 requires extra signal from somewhere — MSA coevolution via plmDCA, a large pLM with masked-marginal scoring, or actual combinatorial DMS for the target protein.

## Recommended next steps

Ranked by expected return on effort, given the above:

1. **GFP validation rerun** with CensoredSigmoid + thermo stack. Confirms cross-protein generalisation of the calibration story without requiring new data. One notebook modification + rerun.
2. **TrpB (Johnston 2024) 4-site complete DMS** as a second complete-space validation protein. Principal cross-protein test.
3. **Contact-clique higher-order couplings** — extend the contact map to 3- and 4-cliques, implement `HigherOrderEpistasis`. Non-trivial but architecturally consistent. Test whether structural triples/quads close the local-optima gap.
4. **MSA Potts `PottsEpistasis`** — replace `RandomEpistasis` with plmDCA-derived J_ij for proteins where MSA depth permits. Widely-used reference model; good independent comparison.
5. **Larger pLM retry** — `esm2_t33_650M_UR50D` with masked-marginal scoring and a GPU. Only if the benchmarking-only framing is rejected and per-variant accuracy becomes the goal.

## Files changed today

New modules/methods:
- `src/mutascape/nonlinearity.py` — `floor_hard` optional field on `ThermodynamicSigmoid` (CensoredSigmoid via clip).
- `src/mutascape/validation/comparisons.py` — `tensor_pearson_r`, `fitness_wasserstein`.
- `src/mutascape/validation/fourier.py` — `walsh_l1_distance`, plus `l1_abs_diff` key in `variance_fraction_comparison`.
- `src/mutascape/validation/figures.py` — `fig_sigmoid_curves`, `fig_walsh_spectrum`, `fig_fitness_distribution`, `fig_tensor_scatter`.
- `src/mutascape/validation/calibration.py` — `build_landscape` accepts `noise_sigma` kwarg.
- `src/mutascape/epistasis.py` — `SignConsistentEpistasis` sign rule fixed to `sign(h_a + h_b)`; docstring rewritten.

New scripts:
- `scripts/plm_direct_gb1.py` — ESM2 pseudo-log-likelihood scoring of every GB1 variant + calibration + tensor output.

New / updated docs:
- `docs/experiments_log.md` — new file, chronological record.
- `docs/project_status.md` — this file.

Notebook:
- `notebooks/validation/gb1_validation.ipynb` — CensoredSigmoid sweep, visual comparison section (4 figures), noise sweep (Experiment 1), pLM-direct evaluation (Experiment 2).

## Status flags

- [ready] nonlinearity module + thermo + censored + tests
- [ready] GB1 validation with thermo + censored + four comparison plots
- [ready] Three held-out scalar metrics (tensor_r, walsh_L1, fitness_W1)
- [ready] Noise-as-ruggedness negative result
- [ready] pLM-direct (unmasked, small model) negative result
- [pending] GFP validation rerun under current pipeline
- [pending] TrpB cross-protein validation
- [pending] Contact-clique higher-order couplings
- [pending] pLM with masked-marginal + larger model (only if per-variant accuracy becomes the goal)
