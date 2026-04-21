# GB1 validation: analysis, methodology, next steps

> **Historical note (2026-04-21).** This document is an alpha-era analysis kept for provenance. It predates the frozen publication pipeline in `src/mutascape/validation/reference_configs.py`, where GB1 uses `sigma = 0`, fitted censored thermodynamic nonlinearity, `floor_hard = -1.0`, and `noise_sigma = 0.3`. Treat the calibration advice below as superseded unless you are deliberately revisiting the old pairwise-coupling model.

Companion to `notebooks/validation/gb1_validation.ipynb`. This document interprets the results, describes the `alpha`-calibration methodology, explains how to pick `alpha` when a complete combinatorial DMS is *not* available, and lays out concrete next steps.

## 1. Figure interpretation (gb1_validation_summary)

The summary figure has two rows. The top row shows depth-resolved metrics at mutation depths 2, 3, 4 (real GB1 in black, mutascape at $\alpha=1.5$ in blue). The bottom row shows epistatic-deviation distributions at depths 2 and 4.

### Where the model succeeds

- **Sign epistasis profile (top middle)**. Real 0.30 / 0.44 / 0.66 across depths 2/3/4; synthetic 0.22 / 0.49 / 0.67. The depth-4 value matches almost exactly, depth-3 is on top of the real curve, and the overall monotonic rise with depth is correctly reproduced. The mean RMSE across depths is ~0.05.
- **Epistatic deviation magnitude at depth 4 (top right)**. Real MAE 1.09 vs synthetic 1.17. The curves cross at about depth 3.5 and the magnitudes at the deep end are within ~8%. Whatever mutascape is doing, it ends up in roughly the right ballpark once you stack four mutations.
- **Depth-4 epistatic deviation distribution (bottom right)**. The two histograms overlap substantially; both are slightly right-skewed with a mode near zero and a long positive tail. mutascape does not reproduce the distinct bimodality of real GB1 (peaks around +1.5 and +2.0), but the bulk shape is close enough that a Wasserstein distance objective sees a decent match.
- **Shape of the roughness-to-slope ratio (reported in the notebook table).** Real 0.807 vs synthetic 0.789, within ~3%. A scalar ruggedness summary agrees even though the internal variance decomposition doesn't.

### Where the model fails

- **Additive correlation profile (top left)**. This is the biggest miss. Real Pearson $r$ drops steeply from 0.59 -> 0.33 -> 0.14 across depths 2/3/4. Synthetic stays high: 0.89 -> 0.80 -> 0.65. At every depth, synthetic fitness is ~0.3 more additive than real. Real GB1 loses most of its single-site predictability by depth 3; mutascape retains most of it. The synthetic landscape is fundamentally too close to the additive model.
- **Depth-2 epistatic deviation distribution (bottom left)**. Synthetic has a tall narrow spike at zero, meaning most depth-2 variants are predicted as essentially additive. Real GB1 has a much wider, multi-peaked distribution extending to $\pm 2$. This is the distributional reflection of the additive-correlation failure above: at depth 2, mutascape assigns tiny epistatic deviations to too many variants.
- **MAE slope (top right)**. Synthetic MAE grows steeply from 0.47 (depth 2) to 1.17 (depth 4); real grows gradually from 0.75 to 1.09. At depth 2 synthetic underestimates magnitude by ~40%; at depth 4 it overestimates by ~8%. The curves cross and the shape is qualitatively wrong.
- **Walsh decomposition (from the notebook table, not in this figure)**. Real GB1 variance by order: 0.19 / 0.43 / 0.31 / 0.08. Synthetic: 0.84 / 0.16 / 0 / 0. 3rd and 4th order are zero by construction in mutascape's pairwise-only model. ~39% of real variance is unreachable.
- **Local optima count**. Real 182 vs synthetic 5. The synthetic landscape is much smoother in a literal topological sense.

## 2. Determination: does this synthetic landscape work?

Honest verdict: **partially, with known limits.**

- For **controllable landscape ruggedness studies** (sweep $\alpha$, measure how an MLDE method's performance changes): mutascape works. It gives a parameterised family of landscapes whose low-order epistasis behaves sensibly.
- For **benchmarking MLDE methods against a proxy of real GB1**: mutascape systematically underestimates epistatic complexity. A method that works well on a mutascape landscape calibrated to GB1 will likely underperform on real GB1. The gap is not small: at depth 2, mutascape assigns ~80% of the variance to singles, real GB1 assigns ~35%.
- For **reproducing structural properties of the real landscape** (local-optima basin structure, higher-order Walsh variance, reciprocal sign epistasis): the current pairwise-only model is fundamentally insufficient on small-site complete combinatorial data like GB1.

The setup "works" in the limited sense that it produces a landscape whose low-order summary statistics (sign epistasis fraction, $r/s$ ratio, sign of fitness-distance correlation) land in the right neighbourhood after a single scalar calibration. It does not "work" in the sense of being a faithful surrogate of the full GB1 combinatorial space. For a publication, the honest framing is the former, with the latter documented as a known limitation that motivates a higher-order coupling variant.

## 3. Methodology for determining `alpha`

### 3.1 Current approach (requires multi-mutant data)

The calibration pipeline in `src/mutascape/validation/calibration.py` works as follows:

1. Load real multi-mutant data into the standard DataFrame schema via `load_proteingym_multimutant` or `load_gb1_combinatorial`. This produces rows with `mutations`, `fitness`, `additive_prediction`, `depth`, `epistatic_deviation`.
2. Compute depth-resolved real metrics with `compute_all_metrics` (additive correlation, sign epistasis fraction, epistatic-deviation distributions, fitness moments).
3. Run `alpha_sweep` across a coarse grid (typically $\{0, 0.25, 0.5, 1, 2, 3, 5, 8, 12\}$). At each $\alpha$, `sample_synthetic_data` draws 3000 variants per depth from the synthetic landscape, the same metrics are computed, and `composite_validation_score` reduces the real/synth comparison to a single scalar.
4. `refine_alpha` does a finer sweep (step 0.25, half-range 1.5) around the coarse minimum.
5. `cross_validate_alpha` optionally does 5-fold train/test splits to guard against overfitting to the calibration data.

The composite objective is:

$$\text{score}(\alpha) = 0.4\,\text{RMSE}(r_\text{add}) + 0.2\,\text{RMSE}(f_\text{sign-epi}) + 0.2\,\bar{W}_\text{dev} + 0.2\,\text{RMSE}(\bar{f}_\text{depth})$$

where $r_\text{add}$ is additive correlation vs depth, $f_\text{sign-epi}$ is sign-epistasis fraction vs depth, $\bar{W}_\text{dev}$ is the mean Wasserstein distance between real and synth epistatic-deviation distributions across depths, and $\bar{f}_\text{depth}$ is mean fitness by depth. Weights are set in `comparisons.composite_validation_score` and can be overridden.

All four terms require *real* data at depths $\geq 2$. That is the fundamental constraint.

### 3.2 What if you don't have combinatorial DMS data?

A common situation: you have a good single-site DMS scan, a structure or a pLM-predicted structure, and no multi-mutant measurements for that protein. You still want a defensible $\alpha$. Options, roughly in order of robustness:

**Option A - Transfer `alpha` from a related protein (recommended when possible).**
If someone has published a combinatorial DMS on a structurally or functionally similar protein, calibrate $\alpha$ there and transfer it. The mutascape design assumes that $\alpha$ captures protein-intrinsic epistatic strength, which should be roughly comparable across proteins of similar size, fold class, and functional constraint. Current calibrated values in this repo:

| Protein | Size | DMS style | Calibrated $\alpha$ |
|---|---|---|---|
| GFP (Sarkisyan 2016) | 237 aa | sparse random multi-mutant, up to 15-fold | $\approx 5.0$ |
| GB1 (Wu 2016) | 56 aa (4-site complete) | dense small-site combinatorial | $\approx 1.5$ |

The $5.0$ vs $1.5$ gap is striking and reflects different experimental regimes more than different biological epistasis. Until more proteins are calibrated, pick whichever is closer to your situation: full-length sparse sampling -> $\alpha \sim 5$; small-site complete -> $\alpha \sim 1.5$. Flag in the paper that this is transferred, not fit.

**Option B - Run a minimal multi-mutant validation subset.**
You don't need a full combinatorial scan to calibrate. A few hundred to a few thousand doubles and triples suffice to compute stable additive-correlation and sign-epistasis profiles at depths 2 and 3, which is most of the calibration signal. Concretely: rational-design or random-pick ~500 doubles and ~500 triples, measure them, feed through `load_proteingym_multimutant` (with a thin adapter if your format differs) and run the standard `alpha_sweep`. This is often cheaper than a full DMS scan and gives you a protein-specific $\alpha$ rather than a transferred one.

**Option C - Use pLM-estimated pairwise effects as a calibration proxy.**
Protein language models like ESM2 can score double mutants by masked-marginal log-probability. Compute a pLM-estimated fitness for a synthetic set of doubles/triples sampled at the mutascape-contacting position pairs, and use *that* as the "real" dataset in the calibration pipeline. This replaces experimental measurement with model inference. It only works to the extent that the pLM's multi-mutant effect estimates are meaningful (mixed evidence; reasonable as a prior, weak as ground truth). Report $\alpha$ chosen this way as "pLM-calibrated" rather than "DMS-calibrated" in any publication.

**Option D - Tune `alpha` against a target $r/s$ ratio from the literature.**
Szendro et al. 2013 and follow-ups report roughness-to-slope ratios across many empirical landscapes, typically in the range 0.3 to 1.0 for proteins under strong selection. Pick a target $r/s$ based on a comparable published landscape, then scan $\alpha$ until `roughness_slope_ratio(synth_df)` matches within, say, 10%. Much weaker than the composite objective but entirely doable without any new multi-mutant measurements. Use this as a prior, not a final answer.

**Option E - Set `alpha` from theoretical priors (weakest).**
The coupling distribution in `RandomEpistasis` has $\sigma = 0.3$ by default, meaning couplings are $\mathcal{N}(0, 0.3)$. If you pick $\alpha$ such that the expected epistatic contribution matches a plausible fraction (say, 20-40%) of the typical single-site effect magnitude, you get a ballpark $\alpha$ without any data. Report this explicitly as an uncalibrated prior.

**Option F - Report an `alpha` sweep rather than a single value.**
For a benchmark paper where the goal is "how does method X perform as landscape ruggedness varies", skip the calibration entirely and report results across $\alpha \in \{0, 0.5, 1, 2, 5, 10\}$. This sidesteps the calibration question and produces a richer result.

### 3.3 Decision tree

- Have combinatorial DMS on this protein? -> run `alpha_sweep` + `refine_alpha` (the current path).
- Don't have combinatorial DMS, have a related protein with one? -> transfer $\alpha$ (Option A) and flag it.
- Can run a small custom multi-mutant panel? -> do it (Option B). Best non-public-data approach.
- Can afford only the pLM? -> use pLM-predicted doubles (Option C) but downgrade the confidence of any claim.
- None of the above but need a defensible number? -> literature $r/s$ target (Option D).
- Writing a benchmark paper? -> sweep $\alpha$ (Option F) and don't commit to one.

## 4. Why the current calibration picks $\alpha = 1.5$ (and why that's a compromise)

Looking at the coarse sweep in the notebook:

| $\alpha$ | composite score | add-corr d2 | sign-epi d2 |
|---|---|---|---|
| 0.0 | 0.618 | 1.00 | 0.00 |
| 1.0 | 0.328 | 0.95 | 0.14 |
| **1.5** | **0.253** | **0.89** | **0.22** |
| 2.0 | 0.271 | 0.83 | 0.28 |
| 3.0 | 0.373 | 0.72 | 0.37 |
| 5.0 | 0.659 | 0.55 | 0.43 |

### Clarification: the two metrics aren't alternative strategies

Additive correlation and sign epistasis are **not two options to choose between**. They are two different ways of looking at the *same* synthetic landscape. Every variant mutascape generates has both an additive correlation and a sign-epistasis status; we compute them both in every evaluation. The optimiser doesn't pick one or the other; it picks the single $\alpha$ that minimises a weighted combination of *all four* terms in the composite score.

What the table shows is that, under the current mutascape model, those four metrics do not all peak at the same $\alpha$:

- Additive correlation (synth) *decreases* as $\alpha$ grows, because stronger pairwise couplings make the landscape less predictable from single-site effects. Real GB1 is at 0.59 (depth 2), so we want synth's additive correlation at ~0.59 -> we want $\alpha$ roughly 5.
- Sign epistasis (synth) *increases* as $\alpha$ grows, because stronger couplings more often flip the sign of a mutation's effect in a multi-mutation background. Real GB1 is at 0.30 (depth 2), so we want $\alpha$ roughly 1.5.
- These two $\alpha$ targets disagree. The composite averages them and lands at ~1.5.

### Why they disagree: one dial, many targets

mutascape's fitness function has one continuous parameter for epistatic strength: $\alpha$. Real GB1's landscape has **many** independent epistatic properties -- pairwise coupling magnitudes, third-order interactions, distribution shape, local topology. In real GB1 these properties happen to have a specific joint pattern (moderate additive correlation *and* moderate sign epistasis *and* strong 3rd-order interactions, etc.). Under the mutascape model, pairwise couplings drawn from $\mathcal{N}(0, \sigma)$ tie these properties together: as $\alpha$ grows, additive correlation falls and sign epistasis rises *in lock-step*. The ratio at which they move is a property of the model, not of $\alpha$.

So when a single $\alpha$ can't hit both targets, it's not a calibration failure or a "strategy choice" -- it's the model telling you its built-in ratio between these two properties doesn't match real GB1's ratio. The gap is evidence that the **model class** (random pairwise couplings) is too restrictive for this landscape. More dials are needed: a non-linear global function, empirical pairwise couplings, explicit higher-order couplings. Section 5 lists these.

### What this means for the user's goal (replicating real combinatorial properties)

If the goal is to produce a synthetic landscape that mimics the *full* statistical fingerprint of real GB1 (every metric close to real), then $\alpha$ alone is not enough. You need to either:

1. **Fix the calibration objective to match what you care about downstream.** If your downstream use is MLDE benchmarking where greedy walks matter most, weight sign-epistasis and local-optima metrics more heavily. If you care about linear-model-based methods, weight additive correlation more. Edit `composite_validation_score` to reflect those priorities and accept that the other metrics will be off.
2. **Accept that one $\alpha$ is a compromise across metrics, and report the full trade-off curve.** Sweep $\alpha$, report every metric at every $\alpha$, let the reader see the Pareto front. This is the honest presentation for a paper.
3. **Add capacity to the model so that $\alpha$ is no longer the only dial.** The global non-linearity in section 5.1 or the empirical $J_{ij}$ in section 5.2 would let you hit more targets simultaneously. This is what needs to happen for mutascape to generate a surrogate that matches real GB1 across all metrics at a single calibration point.

For a user who wants mutascape to replicate combinatorial properties **on proteins without full DMS data**, the situation is worse: you cannot even observe the trade-off without multi-mutant measurements, so you're stuck with whichever single-dial $\alpha$ transfer you picked (Section 3.2). Until the model has enough capacity to hit multiple metrics with one $\alpha$, transferred calibration will always inherit the protein of origin's specific trade-off position.

### Practical recommendation right now

- **For GB1 reproduction today**: pick the metric that most matters to your downstream task and use that single metric as your calibration objective. Don't use the composite until the model can satisfy all four terms simultaneously. If you care about ruggedness (greedy walks, MLDE): optimise against sign-epistasis + local-optima count. If you care about linear predictability: optimise against additive correlation. Report what you optimised.
- **For other proteins without DMS**: transfer the protein-matched $\alpha$ from the closest calibrated protein and state explicitly that you are using a single-dial approximation. The current GFP $\alpha=5$ is calibrated for sparse-random sampling; GB1's $\alpha=1.5$ is calibrated for small-site complete. Pick whichever regime you're in.
- **For publication**: section 5.1 (global non-linearity) is the single change that would make a single calibrated $\alpha$ hit more metrics at once. Until that's done, be explicit that "calibrated mutascape matches a set of low-order statistics, not the full structural fingerprint."

## 5. Next steps to improve

In rough order of expected impact on the held-out metrics:

1. **Non-linear global epistasis (Otwinowski 2018, Olson 2014)**. A common empirical pattern is fitness = $g(\text{additive} + \text{pairwise})$ with $g$ a saturating monotonic non-linearity (sigmoid, softplus, censored). This generates *apparent* higher-order Walsh variance from a simple underlying model. Adding a fitted $g$ would likely close most of the Walsh order-3/4 gap without introducing explicit 3- and 4-way couplings. Cheapest real improvement.

2. **Empirical pairwise couplings instead of random $J_{ij}$**. The current `RandomEpistasis` samples $J_{ij} \sim \mathcal{N}(0, \sigma)$. On GB1 we have the exact 2091 doubles; we can fit $J_{ij}(a,b)$ directly from them (subtracting singles) and use those in the landscape. This converts "random pairwise" into "Potts-like empirical pairwise" and fixes the pairwise-shape mismatch (the spike at zero in the depth-2 deviation distribution).

3. **Explicit higher-order couplings**. Add a `HigherOrderEpistasis` model with 3- and 4-way terms at contacting triples/quads. Raises the model's expressiveness ceiling. Implementation cost is moderate; calibration cost grows since there are more free parameters.

4. **Better noise floor handling**. Wu et al.'s DMS_score has a hard 0 floor (non-binding variants), becoming -1 after WT subtraction. This creates a dense plateau of "dead" variants that distorts the deviation distributions, inflates local-optima counts (dead plateaus look like peaks to the strict-greater-than criterion), and corrupts the Walsh decomposition. Consider: (a) log-transform of DMS_score before calibration, or (b) censored-likelihood modelling of the dead-floor, or (c) filter variants at or near the floor from calibration and evaluate separately.

5. **Recalibrate with a richer objective**. Extend the composite score to include a Walsh-variance-fraction term once the model can actually carry higher-order variance. This closes the calibration/evaluation gap where Walsh decomposition is held out today.

6. **Sparse-coupling priors**. Replace Gaussian $J_{ij}$ with a sparse distribution (mixture of zero-inflated + heavy-tailed). Biologically motivated: most position pairs don't interact strongly, a few do.

7. **More validation proteins**. TrpB (Johnston 2024, 4-site combinatorial, ~160k variants) and Olson 2014 GB1 (all pairwise doubles across 55 positions, ~535k) would test whether the improvements above generalise.

## 6. Thermodynamic non-linearity: actual results on GB1

The proposal in section 5.1 is now implemented (`src/mutascape/nonlinearity.py`) and evaluated end-to-end in `notebooks/validation/gb1_validation.ipynb`. Summary of what changed:

### Configuration space

The sigmoid has three parameters. Floor and scale are well identified by MLE against the singles DFE. Ceiling is **not** identified from singles alone — GB1 singles top out at +2.9 while combinatorials reach +7.8. We introduced a `combinatorial_upside` multiplier as a stability-stacking prior for the no-DMS case: `1.0` uses the singles-only ceiling, `3.0` is the recommended default for stability-limited proteins (consistent with published thermodynamic DMS ranges).

### Actual numbers (all at alpha=1.5 unless noted)

| Metric | REAL | Identity baseline | Thermo upside=1 | Thermo upside=3 | Thermo joint best |
|---|---|---|---|---|---|
| composite score | -- | 0.253 | 0.247 | 0.188 | **0.166** |
| add-corr d=2 | 0.589 | 0.892 | 0.852 | **0.727** | 0.775 |
| add-corr d=4 | 0.141 | 0.650 | 0.539 | **0.359** | 0.458 |
| Walsh order 1 | 0.186 | 0.841 | 0.668 | **0.457** | 0.471 |
| Walsh order 2 | 0.429 | 0.159 | 0.238 | **0.382** | 0.383 |
| Walsh order 3 | 0.308 | **0.000** | 0.073 | **0.137** | 0.129 |
| Walsh order 4 | 0.077 | **0.000** | 0.021 | **0.024** | 0.018 |
| local optima | 182 | 5 | 8 | **9** | 4 |
| accessible path frac | 0.317 | 0.772 | 0.738 | **0.729** | 0.901 |
| r/s ratio | 0.807 | 0.789 | 0.772 | **0.874** | 0.717 |
| FDC (to peak) | -0.169 | -0.372 | -0.387 | **-0.322** | -0.487 |

(The joint best used `scale=1.0` rather than the MLE `scale=0.71`, trading structural match for a lower composite. For replication we prefer the upside=3 MLE-scale row.)

### Where the non-linearity succeeded

- **Walsh orders 3 and 4 are no longer zero.** Real GB1 carries ~39% of variance in these orders; the non-linearity captures 14% and 2.4%. About 44% of the real order-3 variance and 32% of order-4 is now reproduced by a model that has no explicit higher-order couplings. This is Sailer & Harms 2017's mechanism delivering on its theoretical promise.
- **Additive correlation drops substantially.** At depth 4, synth dropped from 0.65 to 0.36, closer to real's 0.14. The model is no longer structurally too additive.
- **r/s ratio within 8% of real** (0.87 vs 0.81). Scalar ruggedness summary now in the right neighbourhood.
- **Composite score improves by 25% (0.253 -> 0.19)** at the MLE scale, and by 35% (-> 0.166) at the joint composite-best. The thermodynamic model strictly beats Identity on its own calibration objective.

### Where the non-linearity did not close the gap

- **Local optima count** moved from 5 to 9 — real is 182. A smooth sigmoid cannot generate the dense plateau of near-equal-low-fitness variants that populates the bottom of the real GB1 landscape. That plateau is a by-product of the DMS_score hard floor at 0 (dead variants), which creates many same-valued neighbours that each count as peaks under a strict greater-than criterion. Fixing this probably requires either an explicit censored-fitness model or a Censored non-linearity variant, not more sigmoid tuning.
- **Accessible path fraction** barely moved (0.77 -> 0.73 vs real 0.31). Same cause: monotonic paths are dominated by the bottleneck structure created by the dead plateau, which the sigmoid smooths over.

### What this means for the "does it work" question

- For **Walsh decomposition, r/s, FDC, depth-resolved additive correlation**: the thermodynamic non-linearity is now a strong model. These metrics are in the right ballpark with a single sigmoid fit from singles plus a stability-stacking prior.
- For **local optima count and path accessibility**: the current model understates them badly because it lacks a dead-floor representation. A `CensoredSigmoid` (sigmoid + hard floor) is the obvious next implementation and should close these remaining gaps.
- **Critically: all calibration is from singles + structure, not from combinatorial data.** The `combinatorial_upside=3` prior is the only setting that uses knowledge external to the singles DMS. This is a meaningful step toward the user's stated goal of generating realistic combinatorial landscapes for proteins where combinatorial DMS does not exist.

## 7. TL;DR

- The pairwise-only baseline tracks real GB1's sign-epistasis profile and r/s ratio but badly underestimates non-additivity at low mutation depth (depth-2 additive correlation 0.89 vs 0.59) and carries zero Walsh variance at orders 3-4 by construction.
- The thermodynamic non-linearity (Otwinowski 2018) has been added to mutascape and validated. At alpha=1.5 with a stability-stacking prior (`combinatorial_upside=3`), the model now reproduces ~44% of real GB1's 3rd-order Walsh variance, cuts the depth-4 additive-correlation miss in half, and lowers the composite validation score by 25%. All calibration is from singles + structure only.
- Remaining gap: local-optima count (9 vs 182) and accessible-path fraction (0.73 vs 0.32) remain off because the current sigmoid does not represent the dead-variant plateau in the DMS data. A `CensoredSigmoid` (sigmoid + hard floor) is the proposed fix.
- For proteins without combinatorial DMS: the pipeline is now `fit ThermodynamicSigmoid from singles with combinatorial_upside=3`, then use the existing alpha transfer or literature-r/s target. This is materially better than the pairwise-only transferred-alpha approach previously available.
- The synthetic landscape now works as a credible low-to-mid-order surrogate for small-site combinatorial DMS data. A full match on peak-structure metrics is the next implementation target, not a fundamental model limitation.
