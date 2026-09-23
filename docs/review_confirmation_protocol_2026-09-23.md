# Confirmation protocol

Recorded before scoring new GraphFLA targets on 2026-09-23.

## Frozen choices

Use the current singles-only generator with coupling 0.1 and the fitted readout.
Do not tune parameters or select a reporting budget after seeing confirmation results.
Test all 12 search methods at 48, 96, 192, and 384 queries.
Use ten search trials, three generator draws, and seed 20260923.
Keep candidate sets and search seeds identical across generators.
Include additive, Potts, RMF, and a matched model without the readout.
Report each target separately and give each protein family equal weight.
Report pair agreement, Kendall correlation, eligible comparison counts, and measured repeatability.
Include a fixed-ranking null fitted only on the existing development proteins.

## Selection rule

Inspect the pinned local GraphFLA collection.
Use metadata and sequence coverage to identify unused protein families.
Exclude every family used in prior fitting, model development, or the existing benchmark.
Exclude RNA, organism-level combinations, communities, and non-sequence datasets.
Require a complete Cartesian protein landscape with 384 to 200,000 candidates.
Require a documented reference genotype and informative measured singles.
Choose one assay per family in filename order, up to 20 families.
Report every exclusion and the number that actually qualifies.
Do not substitute related families or lower the size threshold to reach 20.
Freeze the selected manifest and its source hashes before running search or classifier comparisons.
If fewer than 20 families qualify, use all qualifying families and report the actual number.

## Interpretation

Prior dataset availability does not prove that nobody has inspected it.
These targets can support a prospective test under this recorded protocol.
They cannot establish an externally registered or blinded study.
Keep any model developed in response to this review separate from the frozen confirmation model.

## Classifier and shape checks

Use 2,000 fixed multi-mutant genotypes, seed 20260923, and the existing paired classifier features.
Run logistic regression and histogram gradient boosting with five grouped folds.
Use training-fold scaling for logistic regression and no permutation tests for these descriptive checks.
Report raw Wasserstein distance and categorical interaction-order spectra for all three generator draws.
