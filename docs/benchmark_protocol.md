# Sparse-to-Complete Benchmark Protocol

## Research question

Can sparse single-mutant or double-mutant DMS support a realistic, combinatorially complete whole-protein oracle?

Mutascape defines fitness for every sequence in a chosen whole-protein mutation domain. It materialises small libraries and queries larger domains lazily.

## Data units

The benchmark combines 16 SSMuLA landscapes and 14 CombinGym assay landscapes. Related assays remain grouped by protein. This rule keeps TrpB libraries, antibody conditions, and fluorescent readouts within one fold.

Each normalised table contains these fields.

- `sequence` stores the full sequence or mutable-site sequence.
- `fitness` preserves the source assay value.
- `depth` stores Hamming distance from the library wild type.
- `mutations` maps one-based positions in the stored sequence to mutant amino acids.

Each manifest entry records the source URL, source revision, phenotype, schema, positions, and wild type.

## Evaluation splits

The outer split holds out one protein. Mutascape learns its interaction prior from every other protein.

The primary target split exposes wild type and all single mutants. Every supported higher-order variant remains hidden. A secondary split adds a fixed total budget of 96 or 384 observations.

A target variant enters evaluation only after the benchmark measures each constituent single mutation. Reports state this support fraction.

## Completion model

The first model uses an empirical Bayes random-effects decomposition.

1. Measured WT and single effects define the additive component.
2. Measured doubles or triples fix exact Möbius interaction coefficients.
3. Training proteins define interaction scales for each order.
4. Training proteins also define residual location and variance by mutation depth.
5. A seed fixes all missing interaction draws for one oracle realisation.

The random component shares coefficients across related genotypes. This choice creates local correlation in the completed landscape. Depth-specific scaling prevents raw interaction counts from causing uncontrolled variance.

Latent fitness remains separate from assay censoring. Users can request an assay-range-clipped oracle when the assay model supports that choice.

## Metrics

Prediction uses Spearman correlation, absolute error, Wasserstein distance, NDCG, and top-variant recall.

Uncertainty uses CRPS, posterior ranks, interval width, and coverage at 50, 80, and 95 per cent.

Topology uses Walsh or multiallelic ANOVA spectra, local optima, basins, accessible paths, roughness-to-slope, and graph diffusion ruggedness.

Distribution tests use C2ST and MMD on local graph-patch features. C2ST splits by protein or disconnected sequence regions. Real replicate comparisons define the empirical lower bound.

Search transfer uses matched query budgets for random, greedy, and ridge-guided search. Later work will add established ALDE methods. Kendall correlation compares optimiser ranks on real and synthetic oracles.

## Pilot status

A bounded pilot used 200 hidden variants from each of 30 landscapes. It held out 14 protein groups and drew 100 posterior samples.

The posterior reached median Spearman 0.144 and NDCG 0.912. Additive ridge reached 0.238 and 0.888. Median 80 per cent coverage was 0.915.

Grouped C2ST reached median AUC 0.895. Median MMD squared was 0.353. The classifier can therefore separate current posterior samples from measured variants.

The posterior improved design ranking but lost point accuracy and produced broad intervals. These results guide model changes. They do not support the final claim.

Generated pilots live under `notebooks/validation/output_completion_*` and remain outside Git.
