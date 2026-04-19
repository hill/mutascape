# Experiments log

Chronological record of experiments, hypotheses, and findings. Prior experiments are summarised in `project_status.md`; this file captures the living trail of "what we tried, what we learned."

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
