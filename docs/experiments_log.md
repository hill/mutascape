# Experiments log

Chronological record of experiments, hypotheses, and findings. Prior experiments are summarised in `project_status.md`; this file captures the living trail of "what we tried, what we learned."

Read this file chronologically: earlier recommendations can be superseded by later entries. For the current publication state, use `project_status.md`.

Append new entries at the bottom. Keep each entry short: hypothesis, what was run, key numbers, takeaway. Link to the notebook cell / CSV that has the detail.

---

## 2026-04-17: Thermodynamic sigmoid (Otwinowski 2018, Sailer & Harms 2017)

**Hypothesis.** Wrapping the additive + pairwise landscape in a global non-linearity `g(z)` will manufacture apparent higher-order epistasis from a pairwise-only underlying model.

**Ran.** `ThermodynamicSigmoid.fit_from_singles` + joint (alpha, scale) sweep under `combinatorial_upside=3` prior. See `notebooks/validation/gb1_validation.ipynb` and `docs/gb1_validation_analysis.md`.

**Found.** Walsh orders 3/4 went from 0 (structurally unreachable) to 0.137/0.024, closing ~45% of the gap to real GB1's 0.308/0.077. Composite validation score improved 25% (0.253 → 0.19). Local-optima count barely moved (5 → 9 vs real 182), accessible path fraction also only marginal (0.77 → 0.73 vs real 0.32).

**Takeaway.** Smooth sigmoid captures saturation-driven apparent-epistasis at Walsh-3/4 but does nothing for the dead-variant plateau structure.

---

## 2026-04-18: SignConsistentEpistasis sign rule fix

**Hypothesis.** The original product-sign rule `sign(J) = sign(h_a) * sign(h_b)` actually *compensates* doubly-deleterious pairs rather than amplifying damage as the docstring claimed. Changing to `sign(J) = sign(h_a + h_b)` implements the stated intent.

**Ran.** Three-line code change + doc rewrite + test update. Re-ran ablation. See `src/mutascape/epistasis.py` and `notebooks/validation/gb1_validation.ipynb`.

**Found.** Under the new rule at alpha=1.5, SignConsistent shifts more variance into Walsh order 1 (0.934 vs Random 0.841). Mechanism: sign(sum)-driven J correlates with the additive signal, acting as gain on order 1 rather than as independent pairwise noise. That *duplicates* what the sigmoid already does, so the two mechanisms now overlap.

**Takeaway.** The fix is correct on docstring-code alignment but SignConsistent is no longer the "independent ingredient" we might have wanted. For closing the ruggedness gap, look elsewhere.

---

## 2026-04-18: CensoredSigmoid (hard floor)

**Hypothesis.** Real GB1 has 76.9% of variants pinned at fitness = -1.0 (assay detection limit). A smooth sigmoid cannot reproduce that spike; clipping `g(z) = max(g_smooth(z), floor_hard)` should close `fitness_W1` and `accessible_frac` gaps and (I predicted) lift `n_optima` via alive-subgraph fragmentation.

**Ran.** One optional field on `ThermodynamicSigmoid` + three lines of clipping logic. Swept `floor_hard ∈ {None, -1.0, -0.9966}` at joint-best config. See `notebooks/validation/output_gb1/gb1_censored_sweep.csv`.

**Found.**
- `fitness_W1`: 0.213 → 0.116 ✓ (halved)
- `walsh_L1`: 0.571 → 0.424 ✓ (26% closer)
- `accessible_frac`: 0.901 → 0.634 ✓ (real 0.317)
- `n_optima`: 4 → 4 ✗ (**no change; my prediction was wrong**)
- `tensor_r`: 0.218 → 0.216 (flat)

**Takeaway.** Pinning cannot create new local maxima because peak-ness is defined by strict inequality and clipping the minimum of a value set preserves all `a > b` relations. What pinning CAN do is block strict-uphill ascent (ties are not uphill), which is why `accessible_frac` responded sharply while `n_optima` didn't. Real GB1's 182 peaks come from ruggedness *within* the 23% alive region, not from the dead plateau.

**Implication for next work.** Need a genuine ruggedness source inside the alive region. Two candidates: measurement noise, or explicit higher-order couplings. Noise is the cheap falsification.

---

## 2026-04-18: Scalar scoreboard + visualisation

Added three held-out scalars to the comparison table and four plots to the notebook:

- `tensor_pearson_r`: per-variant fitness prediction correlation (answers "do we rank variants correctly?")
- `walsh_l1_distance`: sum of |real - synth| across Walsh orders (answers "do we put variance in the right orders?")
- `fitness_wasserstein`: Wasserstein-1 between marginal fitness distributions (answers "do we get the right overall shape?")

Plots: `fig_sigmoid_curves` (g(z) overlay + rug), `fig_walsh_spectrum` (grouped bar), `fig_fitness_distribution` (density overlay), `fig_tensor_scatter` (hexbin per-variant).

**Takeaway.** With these three scalars the model fits the question "simplest model that reproduces the combinatorial distribution?": `fitness_W1` and `walsh_L1` track distributional fidelity, `tensor_r` tracks per-variant accuracy. For use case #1 (ML benchmarking) we care about the first two; for use case #2 (pre-experiment prediction) we care about the third. Joint-best + censored scores well on distributional metrics; `tensor_r` is weak across all configs.

---

## 2026-04-18: Experiment plan — noise-as-ruggedness vs pLM-direct

**Context.** After the CensoredSigmoid finding, the open question is where real GB1's 182-peak ruggedness comes from. Two hypotheses worth testing before reaching for expensive higher-order-coupling infrastructure.

**Experiment 1 hypothesis (noise).** Real DMS has measurement noise; two nearby variants read as slightly different fitnesses, breaking ties and creating many tiny local peaks. If true, `Landscape.noise_sigma > 0` should lift `n_optima` from 9 toward real's 182 without changing distributional metrics much.

**Experiment 2 hypothesis (pLM-direct).** Using ESM2 masked-marginal log-probabilities as the landscape (calibrated to DMS singles via a monotonic transform) is a completely different signal source than "DMS singles + contact-map pairwise + sigmoid." The pLM has implicit higher-order information from 10⁸ training proteins. If pLM-direct hits `fitness_W1` and `walsh_L1` comparable to the structural model, it's simpler overall.

Running both below.

---

## 2026-04-18: Experiment 1 results — noise-as-ruggedness

Swept `noise_sigma ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}` with joint-best thermo + `floor_hard=-1.0` as the base. See `notebooks/validation/output_gb1/gb1_noise_sweep.csv`.

| noise | n_optima | accessible_frac | tensor_r | walsh_L1 | fitness_W1 |
|---|---|---|---|---|---|
| real | **182** | **0.317** | 1.000 | 0.000 | 0.000 |
| 0.00 | 4 | 0.634 | 0.216 | 0.424 | 0.116 |
| 0.50 | 34 | 0.564 | 0.200 | 0.462 | 0.137 |
| 0.75 | 104 | 0.492 | 0.184 | 0.595 | 0.165 |
| **1.00** | **243** | **0.415** | 0.168 | **0.724** | 0.205 |
| 1.50 | 614 | 0.170 | 0.137 | 0.914 | 0.321 |
| 2.00 | 1008 | 0.130 | 0.113 | 1.015 | 0.483 |
| 3.00 | 1470 | 0.091 | 0.081 | 1.213 | 0.900 |

**Finding.** Noise DOES lift `n_optima` monotonically with `noise_sigma` and at `noise=1.0` lands in the right ballpark for both `n_optima` (243 vs 182) and `accessible_frac` (0.42 vs 0.32). But the price is a severely degraded Walsh spectrum (`walsh_L1` 0.42 → 0.72) and a worse distributional match (`fitness_W1` 0.12 → 0.21).

**Takeaway.** Real GB1's 182 peaks are NOT purely noise-induced. A noise level that reproduces peak count simultaneously scrambles the variance structure that the sigmoid + censoring worked hard to build. This is the crucial negative result: noise is not the missing ingredient; ruggedness in GB1 has *structure*, not just stochasticity. That structure must come from somewhere — higher-order couplings, MSA coevolution, or a language model's implicit knowledge.

**Implication.** Experiment 2 (pLM-direct) is now the critical test. If pLM scoring produces a landscape with high `n_optima` *without* degrading Walsh spectrum, we have evidence that real ruggedness is structured in a way language models capture. If not, higher-order explicit couplings become the remaining option.

---

## 2026-04-18: Experiment 2 results — pLM-direct (unmasked, ESM2 t6_8M) — **negative**

Scored every GB1 variant (160k total) via ESM2 `t6_8M_UR50D` unmasked pseudo-log-likelihood over 56-residue mutant sequences, calibrated against DMS singles. Script `scripts/plm_direct_gb1.py`, ran in 31.9 min on MPS. Tensor saved to `data/gb1/plm_direct_tensor.npz`.

**Headline comparison (at α=1.5 for thermo configs):**

| metric | real | Identity | joint-best thermo + floor_hard | **pLM-direct** |
|---|---|---|---|---|
| walsh_1 | 0.186 | 0.841 | 0.471 | **0.844** |
| walsh_2 | 0.429 | 0.159 | 0.383 | **0.145** |
| walsh_3 | 0.308 | 0.000 | 0.129 | **0.011** |
| walsh_4 | 0.077 | 0.000 | 0.018 | **0.000** |
| n_optima | 182 | 5 | 4 | **8** |
| accessible_frac | 0.317 | 0.772 | 0.634 | **0.841** |
| tensor_r | 1.000 | 0.104 | 0.216 | **0.006** |
| walsh_L1 | 0.000 | 1.311 | 0.424 | **1.317** |
| fitness_W1 | 0.000 | 1.323 | 0.116 | **1.195** |

**Finding.** pLM-direct is *worse than Identity on every single metric*. `tensor_r = 0.006` means essentially zero correlation with real fitness. The calibration linear-regression slope on singles came out **negative** (a = -0.289), implying higher ESM2 pseudo-log-likelihood corresponds to LOWER DMS fitness at the singletons. That's the opposite of the expected variant-effect-prediction direction.

**Diagnosis.** Two likely causes interacting:
1. **Unmasked scoring is uninformative.** The model sees the mutant AA as input when computing log P at that position. Trivially, `log P(input token | whole input)` tends toward zero regardless of whether the mutation is functional. We are measuring "how confident is the model in its own input" rather than "how likely is this AA given context."
2. **ESM2 t6_8M is too small.** The smallest model (6 layers, 8M params). Published variant-effect benchmarks use t33_650M (~80x larger). Zero-shot variant effect prediction degrades sharply with model size.

Walsh spectrum shows the mechanism: pLM-direct sits almost entirely in order 1 (0.844), WORSE than Identity. Implication: after linear calibration the pLM scores act basically as a rescaled additive model, with no pairwise or higher-order signal reaching the fitness tensor.

**Takeaway.** The simplest pLM-direct recipe does NOT replace or improve on the structural + thermo model. To get real signal from a pLM we would need (a) masked-marginal scoring with other positions still at mutant context (~4x compute → ~2 hours at t6_8M), and (b) a larger model (t33_650M). Combined, that's ~1 day of compute. At that point the simplicity argument evaporates.

**Revised recommendation.** For the "simplest model that replicates the combinatorial distribution" goal: **the current joint-best thermo + censored sigmoid is the answer**. Neither noise nor naive pLM improves it. The last open gap (local-optima count) is structured in a way that neither can reach. For the benchmarking use case the distributional metrics (`walsh_L1 = 0.42`, `fitness_W1 = 0.12`) are the critical scores, and both are close to real. Ship what we have.

---

## 2026-04-18: Summary of today's session

**Done.**
- Fixed SignConsistent docstring/code divergence (`sign(h_a + h_b)` rule).
- Added CensoredSigmoid via `floor_hard` field on ThermodynamicSigmoid (3-line change).
- Added `tensor_pearson_r`, `fitness_wasserstein`, `walsh_l1_distance` scalar metrics.
- Added four publication-quality plots (`fig_sigmoid_curves`, `fig_walsh_spectrum`, `fig_fitness_distribution`, `fig_tensor_scatter`).
- Ran noise-sigma sweep: structured ruggedness required, noise alone insufficient.
- Ran pLM-direct (t6_8M unmasked): clear negative result.
- Test count: 195 → 211. All green.

**Current best configuration.**
Joint-best thermo (α=1.5, scale=1.0, upside=3 prior) + CensoredSigmoid `floor_hard=-1.0`. On GB1: fitness_W1 = 0.12, walsh_L1 = 0.42, tensor_r = 0.22, accessible_frac = 0.63 (real 0.32), n_optima = 4 (real 182).

**Still-open gap.** Local optima count. Not fixable by noise (negative result) or naive pLM (negative result). Remaining options: bigger pLM + masked-marginal (~1 day compute), explicit higher-order couplings from contact-map cliques (non-trivial implementation), or accept distributional sufficiency for the benchmarking use case.

**Next session candidates.**
1. Run GFP validation with the updated CensoredSigmoid / thermo pipeline to confirm cross-protein generalisation.
2. TrpB (Johnston 2024) 4-site complete DMS — second combinatorial validation.
3. Contact-clique higher-order couplings (Option 1 from the earlier discussion) — if alive-region ruggedness really does need structural interactions.
4. Retry pLM with `esm2_t33_650M_UR50D` + masked-marginal scoring on GPU — only if the benchmarking use case isn't sufficient.

---

## 2026-04-20: Experiment 3 — `ContactNKEpistasis` ruggedness parameter

**Reframe.** Today's session reframed mutascape's goal from "accurate digital twin of GB1" to "tunable benchmarking tool for MLDE methods (ALDE, EvolvePro)". The SSMuLA paper (Li et al., Cell Systems 2025) defines the navigability-attribute framework as the standard for this kind of benchmark.

**Hypothesis.** A position-centric NK-style higher-order coupling layer informed by the contact map gives mutascape a single parameter (`sigma_nk`) that tunes ruggedness from smooth to highly rugged, without abandoning the structural prior (unlike classical NK where partners are random). Implementation: per-position random functions over (position + structural neighbours), pinned so f(all WT) = 0, added to the fitness function with scale `alpha_ho`.

**Implementation.** `ContactNKEpistasis` + `ContactNKTensor` in `src/mutascape/epistasis.py`; `higher_order` and `alpha_ho` fields on `Landscape`; `ContactMap.neighbours(pos)` helper. Additional metric `ssmula_attributes(tensor)` computing the four scalars computable directly from a fitness tensor (`frac_active`, `frac_local_optima`, `kurtosis`, `kde_peak_count`). Test count 211 → 225.

**Results (GB1, joint-best thermo + floor_hard=-1.0, sweep over sigma_nk):**

| sigma_nk | n_optima | accessible | walsh_3 | walsh_4 | walsh_L1 | fitness_W1 |
|---|---|---|---|---|---|---|
| real | **182** | **0.317** | 0.308 | 0.077 | 0 | 0 |
| 0.00 | 4 | 0.634 | 0.159 | 0.023 | 0.424 | 0.116 |
| 0.10 | 5 | 0.661 | 0.148 | 0.032 | 0.461 | 0.159 |
| 0.25 | 23 | 0.648 | 0.148 | 0.078 | 0.479 | 0.253 |
| **0.50** | **144** | **0.405** | 0.183 | 0.182 | 0.631 | 0.515 |
| 0.75 | 340 | **0.334** | 0.233 | 0.266 | 0.692 | 0.926 |
| 1.00 | 565 | 0.238 | 0.278 | 0.322 | 0.688 | 1.461 |

**Finding.** The parameter works. `n_optima` spans 4 → 565 monotonically with `sigma_nk ∈ [0, 1]`. At `sigma_nk = 0.5`, `n_optima = 144` (79% of real 182). At `sigma_nk = 0.75`, `accessible_frac = 0.334` is essentially real 0.317. Walsh order 4 goes from 0.023 (thermo alone) to 0.182 at sigma_nk=0.5 — real is 0.077 so we overshoot, but higher-order variance is being manufactured by the right mechanism.

**Trade-off.** Distributional metrics degrade with `sigma_nk`: `walsh_L1` 0.42 → 0.63 at sigma_nk=0.5; `fitness_W1` 0.12 → 0.52. The ruggedness parameter and the distributional-match metrics are in tension — adding higher-order variance shifts mass away from the dead plateau that `floor_hard` was pinning.

**Publication framing.** This parameter gives mutascape three complementary tuning axes for benchmarking:
- `alpha` (pairwise epistatic strength) — low-order ruggedness
- `noise_sigma` (measurement noise) — stochastic tie-breaking
- `sigma_nk` (Contact-NK strength) — higher-order structured ruggedness

Together these span the SSMuLA navigability-attribute space reported in Li et al. 2025, which is the paper-level defense for "mutascape as a benchmarking tool." Per-protein parameter choice is a per-experiment decision, not a flaw.

**Failure of strict success criteria.** The plan's combined targets (`n_optima ∈ [100, 300]`, `walsh_L1 < 0.5`, `fitness_W1 < 0.2`) are not jointly achievable on GB1 with this single parameter. Report as a finding: the ruggedness-vs-distribution trade-off is real; the parameter is useful for spanning benchmark difficulty, not for reconstructing a specific real landscape.

**Cleanup.** Removed dead code (`bootstrap_ci`, `cross_validate_alpha`, `fig_additive_decay`, `fig_sign_epistasis`, `_plot_real_line`); archived the pLM-direct script and the legacy GFP validation notebook to `_archive/` folders. Test count held at 225 after cleanup.

---

## 2026-04-20 (afternoon): TrpB4 cross-protein validation + two new visualisations

**Goal.** Validate that the ContactNK ruggedness parameter generalises to a second protein (TrpB4, Johnston 2024) and add the two plot types the user asked for: parameter-trajectory and cross-protein overlay.

**Data.** Downloaded from the SSMuLA Zenodo bundle (zenodo.org/records/13910506). TrpB4 measures growth-rate fitness at positions V183/F184/V227/S228 of engineered Tm9D8\* (Thermotoga maritima TrpB). 159,128 variants after filtering stop codons, fitness range [-0.572, +0.592] after re-centring to parent = 0. Singles range narrower than GB1 ([-0.421, +0.235] vs GB1's [-1.0, +2.9]). `data/trpb/prepare_trpb.py` mirrors `prepare_gb1.py`.

**Structural surprise.** TrpB4's four active-site residues aren't all in pairwise 3D contact: 183-184 and 227-228 are each in contact (two adjacent pairs), but the 183/184 cluster sits 14-20 Å from the 227/228 cluster in the chain-A apo structure. All four functionally couple via the shared PLP cofactor, so real TrpB4 has huge higher-order structure (Walsh orders 3/4 hold ~39% of variance — same as GB1).

**Two parallel sweeps.** To quantify the gap between "structural contacts" and "functional coupling":

- **Structural (as-is):** Contact-NK uses the PDB contact graph. K=1 per position. Generates mostly pairwise variance; max `n_optima` reached = 110 at sigma_nk=1.0 (real 802).
- **Active-site-augmented:** the four focal positions are force-connected into a 4-clique before building the NK tensor. K=3 per position. Generates full higher-order variance.

**TrpB4 results (active-site-augmented, the one paper would report):**

| sigma_nk | n_optima | accessible | walsh_3 | walsh_4 | walsh_L1 | fitness_W1 |
|---|---|---|---|---|---|---|
| real | **802** | **0.329** | 0.234 | 0.155 | — | — |
| 0.0 | 6 | 0.535 | 0.136 | 0.051 | 0.404 | 0.072 |
| 0.25 | 51 | 0.362 | 0.153 | 0.164 | 0.174 | 0.058 |
| 0.5 | 314 | 0.160 | 0.149 | 0.378 | 0.445 | 0.058 |
| **0.75** | **712** | 0.083 | 0.147 | 0.539 | 0.769 | 0.064 |
| 1.0 | 1052 | 0.058 | 0.150 | 0.634 | 0.958 | 0.073 |

At sigma_nk=0.75 the active-site Contact-NK lands within 11% of real TrpB4's `n_optima` count (712 vs 802). Walsh_L1 and fitness_W1 trade off with ruggedness as expected. Unlike GB1, TrpB's narrower fitness range keeps `fitness_W1` small across the whole sweep (~0.06).

**Cross-protein take-away.** The Contact-NK parameter behaves qualitatively identically on both proteins: monotonic with sigma_nk, crosses real n_optima somewhere in [0.5, 0.8], trades off distributional fidelity. GB1 crosses at sigma_nk ≈ 0.55, TrpB at sigma_nk ≈ 0.8. Both proteins' SSMuLA `frac_local_optima` values fall inside the span of the mutascape sweep.

**Methodological lesson for the paper.** Spatial contact maps don't always capture functional coupling. For active-site residues that interact via a shared ligand, mutascape supports overriding the distance matrix to force a focal clique. Flagged as a deliberate modelling choice, not a bug.

**New visualisations.**
- `fig_parameter_trajectory` (in `figures.py`): one subplot per metric, synth trajectory over sigma_nk with a dashed line at the real landscape's value. Shows immediately where the sweep crosses each target.
- `fig_cross_protein_nk` (in `figures.py`): same subplot layout but overlays GB1 and TrpB synthetic curves plus their real reference lines.

Outputs are under `notebooks/validation/output_cross_protein/` and the per-protein `output_gb1/` / `output_trpb/`.

**Display fix.** Self-comparison columns on the "REAL" row (tensor_r, walsh_L1, fitness_W1) now show NaN instead of the misleading 0/1 values — they are trivially zero by definition and were confusing in the CSV outputs.

---

## 2026-04-20 (later afternoon): pivot to noise-based headline + two-scalar metric framing

**Trigger.** "I feel we are overcomplicating this. Tell me what our goal is." The Contact-NK approach produced a paper story with too many parameters and a structural bias (K=3 tensors concentrate Walsh variance at order 4), and the implicit objective was a six-attribute SSMuLA coverage claim that nobody reports as a headline number.

**Decision.** Two changes:

1. Commit to two headline scalars: `walsh_L1` (Walsh spectrum L1 distance, structural match) and `fitness_W1` (Wasserstein-1 on fitness distribution, distributional match). Per-order Walsh vector and SSMuLA attributes stay as reported descriptive figures, not optimisation targets.
2. Make `noise_sigma` the paper's primary ruggedness parameter. i.i.d. Gaussian noise per variant pre-nonlinearity, which distributes Walsh variance approximately uniformly across orders (vs Contact-NK's K+1 concentration). Already implemented in `Landscape` and `build_landscape`; no new mechanism code.

Contact-NK demoted, not deleted. All classes, tests, and kwargs remain; defaults are inert. Kept as an experimental alternative in both validation notebooks.

**Noise sweep results.**

GB1 at `noise_sigma=0.3` (joint minimum of walsh_L1 + fitness_W1):

| | walsh_1 | walsh_2 | walsh_3 | walsh_4 | walsh_L1 | fitness_W1 | n_optima |
|---|---|---|---|---|---|---|---|
| real | 0.186 | 0.429 | 0.308 | 0.077 | - | - | 182 |
| synth | 0.385 | 0.396 | 0.155 | 0.064 | **0.399** | **0.123** | 11 |

TrpB at `noise_sigma=0.3` (same parameter, same protein calibration path):

| | walsh_1 | walsh_2 | walsh_3 | walsh_4 | walsh_L1 | fitness_W1 | n_optima |
|---|---|---|---|---|---|---|---|
| real | 0.255 | 0.356 | 0.234 | 0.155 | - | - | 802 |
| synth | 0.407 | 0.374 | 0.115 | 0.104 | **0.339** | **0.035** | 27 |

Both proteins share the same `noise_sigma` optimum, which is unplanned but convenient evidence for the parameter's cross-protein behaviour.

**Success criteria from the pivot plan:**

1. [PASS] GB1: walsh_L1 < 0.4 AND fitness_W1 < 0.15. (0.399, 0.123)
2. [PASS] TrpB: walsh_L1 < 0.4 AND fitness_W1 < 0.20. (0.339, 0.035)
3. [FAIL] Per-order Walsh within ±0.10 per order at best noise_sigma. Order-1 overshoots by ~0.15-0.20 on both proteins; order-3 undershoots by ~0.12-0.15. Orders 2 and 4 within tolerance.
4. Criterion 4 (SSMuLA range coverage) not formally evaluated; qualitatively the noise sweep moves attributes in the expected direction (see `ssmula_coverage.png`).

**Walsh order-composition bias (the paper's honest limitation section).** The pairwise-only anchor model has:
- walsh_1 fixed by DMS singles (`h_i` terms),
- walsh_2 populated by random couplings on contact pairs (`J_ij` terms),
- walsh_3 and walsh_4 unreached by construction.

Noise spreads variance uniformly across orders 1..L, which closes the order-4 gap and slightly narrows the order-3 gap but does not redistribute mass from order 1 to order 3. Real proteins have the mass distribution skewed toward orders 2-3; our synth has it stuck at orders 1-2. Contact-NK was the attempt to fix this and has the opposite bias (concentrates at order 4). A hybrid `noise + Contact-NK` parameter is on the shortlist for future work.

**Why noise, not Contact-NK, for headline:**
- Zero new user-facing kwargs; `noise_sigma` was already there.
- Classical ruggedness mechanism from the NK/RMF literature; reviewers have seen it before.
- Uniform Walsh distribution is the "null" bias; structural bias would need justification.
- Order-4 overshoot of Contact-NK was the observed failure mode; noise doesn't reproduce it.

**New files.**
- `scripts/run_noise_sweep.py`: standalone, takes `--protein`, writes `{protein}_noise_sweep.csv` with full metric set.
- `scripts/build_noise_figures.py`: reads the CSVs, writes `headline.png` (2x2 cross-protein), `walsh_spectrum_at_best_sigma.png` (per-protein), `ssmula_coverage.png` (cross-protein SSMuLA attributes).

**Deleted outputs.** `output_cross_protein/cross_protein_nk.{png,pdf}` replaced by the new headline + coverage figure set.

**Kept alive.** `composite_validation_score` stays because `alpha_sweep` and `thermo_sweep` still use it to pick optimal alpha. Its deletion was in the pivot plan but "verify zero call sites first" caught the issue; deleting would have required rewiring alpha calibration.

---

## 2026-04-20 (evening): MSA couplings assessment + surprise finding on random couplings

**Setup.** Implemented `MSACouplingsEpistasis` as an optional drop-in for `RandomEpistasis`. Fetched UniProt homologs by Pfam family (PF01378 for GB1, PF00291 for TrpB), pairwise-aligned each to the WT sequence with Biopython, extracted the focal-position residues, and computed log-odds couplings J[i,j,a,b] = log p(a,b|i,j) - log p(a|i) - log p(b|j) with Laplace smoothing. This is a cheap heuristic DCA (not plmDCA); pydca was incompatible with our biopython version.

**Sample sizes.**
- GB1 PF01378: 44 UniProt hits, 43 usable after alignment (tiny family).
- TrpB PF00291: 2000 UniProt hits, 1589 usable after alignment.

**MSA couplings assessment at fixed noise_sigma=0.3:**

GB1:

| config | walsh_1 | walsh_2 | walsh_3 | walsh_4 | walsh_L1 | fitness_W1 | n_optima |
|---|---|---|---|---|---|---|---|
| REAL | 0.186 | 0.429 | 0.308 | 0.077 | | | 182 |
| random(sigma=0.3) | 0.385 | 0.396 | 0.155 | 0.064 | 0.399 | 0.123 | 11 |
| MSA scale=0.0 | 0.304 | 0.470 | 0.155 | 0.072 | **0.317** | **0.036** | 3 |
| MSA scale=0.05 | 0.512 | 0.365 | 0.055 | 0.068 | 0.652 | 0.101 | 2 |
| MSA scale=0.1 | 0.705 | 0.217 | 0.022 | 0.056 | 1.038 | 0.591 | 14 |

TrpB:

| config | walsh_1 | walsh_2 | walsh_3 | walsh_4 | walsh_L1 | fitness_W1 | n_optima |
|---|---|---|---|---|---|---|---|
| REAL | 0.255 | 0.356 | 0.234 | 0.155 | | | 802 |
| random(sigma=0.3) | 0.407 | 0.374 | 0.115 | 0.104 | 0.339 | 0.035 | 27 |
| MSA scale=0.0 | 0.242 | 0.410 | 0.201 | 0.148 | **0.107** | **0.028** | 23 |
| MSA scale=0.1 | 0.326 | 0.404 | 0.149 | 0.120 | 0.239 | 0.026 | 10 |
| MSA scale=0.2 | 0.407 | 0.375 | 0.117 | 0.101 | 0.340 | 0.055 | 14 |

Observation 1: log-odds MSA couplings do not improve over random on either protein. At small scale they are barely different from zero (noisy), at larger scale they dominate and make the spectrum much worse. GB1 with only 43 sequences is hopelessly under-sampled for 20x20 coupling estimation; TrpB with 1589 is not much better than random either.

**Observation 2 (the real finding).** The `MSA scale=0.0` row is the no-pairwise-couplings baseline: additive h_i from DMS + censored thermo sigmoid + Gaussian noise, with J_ij=0 everywhere. It produces the best `walsh_L1` and `fitness_W1` on both proteins, dramatically better than any random coupling scale.

**Confirmed with a dedicated sigma sweep at fixed noise=0.3 on both proteins:** `walsh_L1` monotonically increases with `sigma` (the RandomEpistasis strength) on GB1 (0.317 at 0.0 up to 0.490 at 0.5) and TrpB (0.107 at 0.0 up to 0.418 at 0.5). Random pairwise couplings are actively hurting the Walsh spectrum match. The censored thermodynamic sigmoid applied to the additive preimage + i.i.d. noise already produces a spectrum that tracks real closely; random couplings just inject order-1/2 variance the real landscape does not have.

**Headline numbers updated to sigma=0 (no random couplings) + noise_sigma=0.3:**

| protein | walsh_L1 | fitness_W1 | walsh_1/2/3/4 vs real |
|---|---|---|---|
| GB1 | 0.317 | 0.036 | order-1/3 still off, orders 2/4 within 0.04 |
| TrpB | **0.107** | 0.028 | all four orders within 0.06 of real |

GB1 gap: walsh_L1 improved from 0.399 to 0.317 (-21%); fitness_W1 from 0.123 to 0.036 (-71%).
TrpB gap: walsh_L1 improved from 0.339 to 0.107 (-68%); fitness_W1 from 0.035 to 0.028 (-20%).

**Why the censored thermo sigmoid alone is sufficient.** The `floor_hard` threshold implements a biophysical fold/unfold boundary (Otwinowski-Plotkin two-state). When two deleterious mutations are summed additively below this floor, they saturate at the same output, which is *emergent higher-order epistasis* without explicit J_ij. This captures the real structural signal and matches more than half of real's variance at orders 3 and 4 on TrpB.

**Implications for the paper.** The architectural story simplifies considerably: the core generative mechanism is *additive DMS preimage + censored thermodynamic sigmoid + Gaussian noise*, nothing else required for the two-scalar headline. Random pairwise `J_ij` becomes an optional ruggedness mechanism (for users who want specific coupling structure) but the default is now `sigma=0`. MSA couplings are an optional feature for users with high-quality plmDCA output; our log-odds from UniProt homologs did not help, but proper plmDCA with deeper MSAs might. Contact-NK remains an experimental alternative for explicit higher-order structure.

**n_optima remains an open limitation.** Even at the new best config: GB1 has 3 optima vs real 182; TrpB has 23 vs real 802. The n_optima gap is downstream of missing higher-order structure that neither noise nor random couplings recover. Users who care about n_optima-matching benchmarks can increase noise_sigma (at the cost of walsh_L1) or add Contact-NK.

**New/changed code today:**
- `src/mutascape/epistasis.py`: `MSACouplingsEpistasis` class with `from_npz` and `from_msa_fasta` loaders.
- `src/mutascape/validation/calibration.py`: `msa_couplings` kwarg on `build_landscape`.
- `scripts/fetch_msa_couplings.py`: UniProt homolog fetch + pairwise alignment + log-odds coupling computation.
- `scripts/assess_msa_couplings.py`: MSA scale sweep comparing to random baseline.
- `scripts/pairwise_sigma_sweep.py`: the confirmatory sweep over RandomEpistasis sigma.
- `scripts/run_noise_sweep.py`: updated to use pairwise sigma=0 as default.
- `tests/test_epistasis.py`: 8 new tests for `MSACouplingsEpistasis`, all green.

**Test count:** 225 -> 233 (+8 MSA tests).

**Honesty note.** The MSA path as currently implemented (log-odds from pairwise-aligned UniProt homologs) is a cheap heuristic. Publication-grade MSA couplings would use plmDCA (pydca or equivalent) on a proper HMMER alignment. The infrastructure is in place via `MSACouplingsEpistasis.from_npz`; users can plug their own `.npz` with plmDCA-derived J.
