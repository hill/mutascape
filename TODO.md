# Mutascape Research Direction

## Research aim

Build a probabilistic generator of complete combinatorial protein fitness landscapes for ML-assisted directed evolution benchmarks.

The main research question is:

> Can lower-order DMS, structure, evolutionary information, and assay metadata constrain realistic higher-order fitness landscapes?

A second question tests practical value:

> Do sampled synthetic landscapes predict which search method performs best on the matching real landscape?

Mutascape should supply protein-specific ground-truth oracles to an established search harness such as [FLEXS](https://flexs.readthedocs.io/en/latest/). It should not reproduce FLEXS explorers.

## Why ground landscapes in DMS

DMS grounding is better when a benchmark claims biological relevance. Fully synthetic oracles remain useful controls for isolated stress tests.

Real single-mutant measurements provide several anchors:

- They preserve the protein's observed mutation vocabulary and wild-type context.
- They capture assay-specific effect scales, skew, inactive fractions, and censoring.
- They preserve which positions tolerate change and which substitutions destroy activity.

They also support direct validation:

- They separate measured first-order effects from invented higher-order effects.
- They permit direct tests against held-out double and higher-order mutants.
- They reduce the gap between a simulated campaign and the matching laboratory assay.

An NK landscape supplies exact control over interaction order. It does not explain why one amino-acid substitution differs from another. A learned neural oracle can imitate measured data, but its errors become hidden parts of the benchmark.

DMS grounding also has limits. Singles do not identify higher-order interactions. DMS contains sampling bias, assay noise, and floor effects. Calibration and evaluation must therefore use separate mutation orders or proteins.

## Current evidence gap

GB1 and TrpB are nearly complete four-site validation landscapes. They test whether single-mutant inputs can reconstruct unseen combinations. The frozen `sigma=0` result shows that the current random contact-coupling model does not improve the selected aggregate metrics. The fitted nonlinear map and genotype-stable variation reproduce some landscape statistics, but they do not establish the biological source of epistasis.

- [ ] Rename genotype-stable `noise_sigma` as latent micro-ruggedness.
- [ ] Add a separate repeated-measurement noise model.
- [ ] Treat structural benefit as an open hypothesis.
- [ ] Stop selecting parameters on the same complete landscape used for evaluation.

## Benchmark design

### Data bank

- [ ] Import and deduplicate the 16 [SSMuLA](https://doi.org/10.1016/j.cels.2025.101387) landscapes.
- [ ] Import the 14 [CombinGym](https://doi.org/10.64898/2026.03.24.714074) datasets when released in a stable form.
- [ ] Record assay, wild type, mutable positions, structure, MSA, replicates, and measurement floors.

### Evaluation protocol

- [ ] Fit from wild type and singles, then test on all higher mutation orders.
- [ ] Repeat with a small, fixed double-mutant budget.
- [ ] Hold out whole proteins to test transfer of fitted priors.
- [ ] Add matched additive, global-epistasis, NK, Rough Mount Fuji, SLIP, and learned-oracle baselines.
- [ ] Add a FLEXS adapter around `Landscape.evaluate_batch`.

## Probabilistic landscape generation

One parameter set and seed give one invented completion. Mutascape should return an ensemble of landscapes compatible with the available evidence.

- [ ] Define priors over pairwise effects, higher-order effects, thresholds, and latent texture.
- [ ] Fit posterior parameter distributions from lower-order observations.
- [ ] Sample many complete landscapes for each protein.
- [ ] Check coverage of held-out fitness values and landscape statistics.
- [ ] Report uncertainty in local optima, accessible paths, and optimiser rankings.

Replace the single latent phenotype with several simple constraints:

\[
x \rightarrow
(z_{\mathrm{fold}}, z_{\mathrm{bind}}, z_{\mathrm{expression}}, z_{\mathrm{catalysis}})
\rightarrow F(x, c) \rightarrow y
\]

Here, \(c\) represents the assay condition or substrate. Constraint changes may generate localised ruggedness without arbitrary high-order noise. This direction follows recent evidence on [composed biophysical constraints](https://doi.org/10.1101/2025.04.12.648556).

- [ ] Start with folding and activity as two latent traits.
- [ ] Fit a condition-specific map from latent traits to fitness.
- [ ] Add structure-based contact triangles or communities as higher-order terms.
- [ ] Compare real contact graphs with degree-matched shuffled graphs.

## Experimental observation model

Separate biological fitness from its measured readout.

- [ ] Model repeated observations with genotype-independent and genotype-dependent variance.
- [ ] Support count-based enrichment, detection floors, and batch effects.
- [ ] Preserve raw counts and replicates when source datasets provide them.
- [ ] Compare against the Bayesian observation approach used by [FLIGHTED](https://pmc.ncbi.nlm.nih.gov/articles/PMC10996587/).

## Real-versus-synthetic classifier

The proposed test already exists in statistics. A classifier two-sample test, or C2ST, learns to distinguish samples from two distributions. Chance performance supports distributional similarity within the classifier's power and chosen representation. [Lopez-Paz and Oquab](https://arxiv.org/abs/1610.06545) established this method for generative-model evaluation.

A targeted literature search found no landscape-specific C2ST in protein fitness research. Protein sequence generators have used natural-versus-generated discriminators, including [ProGen](https://www.nature.com/articles/s41587-022-01618-2). This project would apply the idea to fitness-labelled genotype graphs.

### Proposed first version

Represent each focal genotype as a local landscape patch containing:

- Normalised focal fitness and mutational depth
- Fitness changes to all observed one-step neighbours
- Additive residual and sign-epistasis indicators
- Local rank, uphill fraction, and best available step
- Contact counts and structural proximity
- Local graph or Walsh features where tractable

Train a classifier to predict `real` or `synthetic`. Use AUC with confidence intervals instead of raw training accuracy.

- [ ] Split by protein, assay, or disconnected Hamming regions.
- [ ] Keep related neighbourhoods within one split to prevent leakage.
- [ ] Compare logistic regression, gradient-boosted trees, and a small graph model.
- [ ] Run label-permutation tests to measure chance performance.

Use independent controls:

- [ ] Compare real replicate A against real replicate B.
- [ ] Compare independent synthetic draws as a negative control.
- [ ] Reserve a final classifier architecture that never tunes the generator.
- [ ] Inspect classifier feature importance to locate synthetic artefacts.

Chance AUC does not prove biological realism. Weak features, low sample size, or an underfit classifier can also cause chance performance. A high AUC provides stronger evidence because it exposes a detectable mismatch.

Do not use C2ST alone. Pair it with [maximum mean discrepancy](https://www.jmlr.org/papers/v13/gretton12a.html), distributional precision and recall, topology metrics, and downstream task tests. Precision and recall separate sample fidelity from coverage failure. [Sajjadi et al.](https://papers.nips.cc/paper_files/paper/2018/hash/f7696a9b362ac5a51c3dc8f098b73923-Abstract.html) formalised this distinction.

## Evaluation hierarchy

### Distribution and topology

- [ ] Marginal fidelity with fitness distributions, active fractions, and depth profiles.
- [ ] Local fidelity with neighbour effects, sign epistasis, and accessible steps.
- [ ] Global fidelity with Walsh spectra, graph spectra, optima, and basin sizes.
- [ ] Detectability with C2ST, MMD, and distributional precision and recall.

### Prediction and use

- [ ] Predictive validity with held-out higher-order fitness and top-variant recall.
- [ ] Task validity with real-versus-synthetic optimiser rank correlation.
- [ ] Uncertainty validity with posterior predictive coverage.

## Controlled phase diagrams

Generate paired landscapes where one biological factor changes and all other inputs remain fixed.

- [ ] Vary stability margin, constraint balance, contact locality, active fraction, and assay noise.
- [ ] Map where random search, directed evolution, ALDE, AdaLead, and EVOLVEpro change rank.
- [ ] Train a landscape-to-strategy recommender from measured landscape features.
- [ ] Validate its recommendation on held-out real proteins.

## Publication target

A strong paper should make one central claim:

> Lower-order experimental evidence can define uncertainty-aware synthetic completions that preserve higher-order topology and search-method behaviour.

The core figures should compare real and synthetic higher-order landscapes, classifier detectability, posterior coverage, and optimiser rank fidelity.
