# Singles-only study details

The initial experiment protocol remains unchanged.
Its saved hashes precede development screening and confirmation scoring.
This addendum records a further control and clarifies the analysis.

## Main generator control

The main-branch singles generator supplies an extra comparison.
Its implementation and default coupling remain unchanged.
The study refits its empirical prior on development proteins only.
Each development target excludes itself from prior fitting.
All four confirmation targets share the five-protein development prior.
No target combination enters its prior or conditioning step.
This avoids overlap between the target and the bundled prior's training data.

The control therefore tests the main generator with new donor priors.
It does not reproduce the bundled prior's exact published numbers.
The vectorised calculation must match the existing public oracle on 100 queries per sample.
These checks use an absolute tolerance of 1e-12.

## Local spectra

The Walsh calculation pools squared coefficients across subcubes before normalisation.
It does not average separately normalised subcube spectra.
Each subcube contains three sites with two alleles per site.
The other sites retain a measured background.
All eight corners must have measurements.
Constant subcubes add zero variance.
Only orders one, two and three enter this local spectrum.
It cannot describe higher interaction orders across the full library.

The sign-switch diagnostic asks whether one substitution changes sign across the four backgrounds within its subcube.
It averages this indicator across axes and subcubes.
Numerical ties do not count as sign changes.

## Search and uncertainty

The search test starts each method without the supplied single-mutant fitness values.
Only the generator receives these values.
This matches the existing benchmark machinery.
It tests transfer of method comparisons under that protocol.
It does not directly test a campaign that trains every method on the DMS before search.

A further analysis gives each search method WT and single-mutant observations at no new query cost.
The four confirmation proteins retain the previously selected generator settings.
The additive, linear, sigmoid and main singles models supply the comparisons.
Measured searches on all five development proteins supply the fixed ranking.
Local search starts at the best known variant.
Random search samples combinations only.
The other methods retain their existing algorithms and access all known observations.
The regret calculation includes known variants among the available outcomes.
Budgets count new combinations only.
This analysis followed inspection of the first search results.
It supplies a separate test of the intended DMS-led workflow.

Synthetic method means average search trials and landscape samples.
Each protein has equal weight in collection summaries.
The fixed ranking uses mean normalised regret on development proteins.
A development target excludes itself from this calculation.
Confirmation targets use all five development proteins.

Bootstrap intervals resample proteins as the independent units.
Intervals from four confirmation proteins give limited precision.
The search trials and landscape samples are repeated measurements within proteins.
They do not increase the number of independent proteins.
Pair eligibility uses paired search trials and the existing 95% bootstrap rule.
The many pair comparisons do not supply independent hypothesis tests.

## Structure-only comparison

The structure-only control samples random single effects in arbitrary units.
It uses the target's mutation library and verified contact map.
It receives no measured target fitness.
Its distribution comparison uses separate standardisation.
Raw Wasserstein distances have no assay interpretation for this control.
The report omits those distances and C2ST scores.
This control tests a structure-constrained synthetic environment only.
