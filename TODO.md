# Mutascape Research Direction

## Research aim

Build an uncertainty-aware generator of whole-protein combinatorial fitness oracles from sparse DMS.

The main research question is:

> Can sparse single-mutant or double-mutant DMS support a realistic, complete whole-protein oracle with calibrated uncertainty?

Mutascape should define fitness for every sequence in its chosen mutation domain. It should materialise small libraries and query larger domains lazily.

## Why ground landscapes in DMS

Real single-mutant measurements preserve the protein context, mutation vocabulary, assay scale, inactive fraction, and censoring. They also support direct tests against hidden combinations.

Singles cannot identify higher-order interactions. DMS also contains bias and assay noise. Evaluation must therefore hold out mutation orders and whole proteins.

## Immediate benchmark

- [x] Import and deduplicate the SSMuLA and CombinGym landscapes.
- [ ] Record source, assay, wild type, mutable positions, structure, replicates, and measurement floors.
- [x] Fit from wild type and singles, then test on all higher mutation orders.
- [x] Add fixed observation budgets of 96 and 384 variants.
- [x] Hold out whole proteins when learning interaction priors.
- [ ] Compare additive, global-epistasis, RMF, NK, current Mutascape, ridge, and pairwise baselines.

## Probabilistic completion

- [x] Learn empirical priors over pairwise and higher-order interaction scales.
- [x] Condition each target oracle on observed singles and optional doubles.
- [x] Sample stable whole-protein oracle realisations.
- [x] Report uncertainty by mutation depth.
- [ ] Separate latent biological variation from repeated-measurement noise.
- [ ] Rename genotype-stable `noise_sigma` as latent micro-ruggedness.
- [ ] Treat structural benefit as an open hypothesis.

Replace the single latent phenotype with simple folding and activity constraints. Add condition-specific fitness maps after the first completion model passes held-out tests.

- [ ] Add structure-based contact triangles or communities as higher-order terms.
- [ ] Compare real contact graphs with degree-matched shuffled graphs.

## Evaluation

- [x] Measure held-out correlation, error, NDCG, and top-variant recall.
- [x] Measure interval coverage, width, CRPS, and posterior ranks.
- [ ] Compare Walsh spectra, graph ruggedness, optima, basins, and accessible paths.
- [x] Add C2ST and MMD with grouped genotype splits.
- [ ] Add real-versus-real controls for each assay.
- [ ] Compare optimiser rankings on real and synthetic landscapes.
- [x] Export tidy tables, static figures, and stable JSON for the web app.

Chance C2ST performance supports distributional similarity within the classifier's power. It does not prove biological realism. Pair C2ST with topology, calibration, and search-transfer tests.

## Experimental observation model

- [ ] Model repeated observations with genotype-independent and genotype-dependent variance.
- [ ] Support count-based enrichment, detection floors, and batch effects.
- [ ] Preserve raw counts and replicates when source datasets provide them.
- [ ] Compare with the Bayesian observation model in [FLIGHTED](https://pmc.ncbi.nlm.nih.gov/articles/PMC10996587/).

## Web studio

- [ ] Ship a versioned empirical prior with source provenance.
- [ ] Accept both single-site and variant-level DMS CSV files.
- [ ] Show posterior intervals and a stable sampled oracle for exact queries.
- [ ] Show uncertainty by mutation depth and across genotype neighbourhoods.
- [ ] Keep benchmark evidence outside the first query-focused release.

## Integrations

- [ ] Add GraphFLA as an optional evaluation adapter.
- [ ] Add a thin FLEXS adapter after the native search protocol validates search transfer.
- [ ] Revisit a Proto constraint adapter after native tests validate the landscape.

## Longer-term tests

- [ ] Vary stability margin, constraint balance, contact locality, active fraction, and assay noise.
- [ ] Map where random search, directed evolution, ALDE, AdaLead, and EVOLVEpro change rank.
- [ ] Train a landscape-to-strategy recommender from measured landscape features.
- [ ] Validate its recommendation on held-out real proteins.

## Publication target

The central claim to test is:

> Empirical interaction priors can turn sparse DMS into complete synthetic oracles that preserve hidden fitness, topology, and search behaviour.
