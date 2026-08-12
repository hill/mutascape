# Mutascape Research Direction

## Research aim

Build an uncertainty-aware generator of whole-protein combinatorial fitness oracles from sparse DMS.

The main research question is:

> Can sparse single-mutant or double-mutant DMS support a realistic, combinatorially complete whole-protein oracle, with calibrated uncertainty as sequence distance increases?

Mutascape should define fitness for every sequence in the chosen whole-protein domain. It should materialise tractable sublibraries and sample larger domains.

## Immediate benchmark

- [x] Import and deduplicate the SSMuLA and CombinGym landscapes.
- [ ] Record source, assay, wild type, mutable positions, structure, replicates, and measurement floors.
- [x] Fit from wild type and singles, then test on all higher mutation orders.
- [x] Add fixed observation budgets of 96 and 384 variants.
- [x] Hold out whole proteins when learning interaction priors.
- [ ] Compare additive, global-epistasis, RMF, NK, current Mutascape, ridge, and pairwise baselines.

## Probabilistic completion

- [x] Learn empirical priors over pairwise and higher-order interaction scales.
- [x] Condition each target oracle on its observed singles and optional doubles.
- [x] Sample stable whole-protein oracle realisations.
- [x] Report uncertainty by mutation depth.
- [ ] Separate latent biological variation from repeated-measurement noise.

## Evaluation

- [x] Measure held-out correlation, error, NDCG, and top-variant recall.
- [x] Measure interval coverage, width, CRPS, and posterior ranks.
- [ ] Compare Walsh spectra, graph ruggedness, optima, basins, and accessible paths.
- [x] Add C2ST and MMD with grouped genotype splits.
- [ ] Add real-versus-real controls for each assay.
- [ ] Compare optimiser rankings on real and synthetic landscapes.
- [x] Export tidy tables, static figures, and stable JSON for the web app.

## Integrations

- [ ] Add GraphFLA as an optional evaluation adapter.
- [ ] Add a thin FLEXS adapter after the native search protocol validates search transfer.
- [ ] Revisit a Proto constraint adapter after native tests validate the landscape.

## Publication target

The central claim to test is:

> Empirical interaction priors can turn sparse DMS into complete synthetic oracles that preserve hidden fitness, topology, and search behaviour.
