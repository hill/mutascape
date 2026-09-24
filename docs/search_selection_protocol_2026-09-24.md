# Search-selection validation protocol

Recorded before scoring the new evaluation searches on 24 September 2026.

## Question

Does choosing a search method through Mutascape improve its measured performance over simpler choices?
The generator and its default parameters remain fixed.
The primary outcome uses measured search performance, without a pair-eligibility filter.
This experiment does not tune the generator.

## Data

The development proteins remain DHFR, eqFP611, ParB-Noc, ParD2 and CreiLOV.
They train the fixed method ranking and cannot enter the test mean.
GB1, TrpB4, T7 and TEV form a separate repeat-evaluation panel.
Their earlier results are already known.

The new candidates are Pab1, Ube4b, Bgl3 and avGFP from the authors' NN4DMS archive.
The source revision is `434e354828f306cbb6796033ef1169cbe5207350`.
The inventory records original studies, file hashes and reference sequences.
A search of project records checks earlier use before any claim of new protein validation.
avGFP shares a broad structural family with earlier fluorescent proteins, so that relationship remains explicit.

Candidate selection uses measurement coverage rather than fitness values.
Keep substitutions with a same-assay single-mutant measurement.
Exclude deletions, insertions, stop codons and unsupported substitutions.
Use the source's documented WT score or WT normalisation convention.
Exclude a dataset if its WT value cannot be established.

Rank positions by frequency among supported measured combinations, with lower position as the tie-breaker.
Retain the sixteen most frequent positions, or all positions when fewer are available.
Keep WT, singles and measured combinations restricted to those positions.
This limits computation and defines a sparse measured candidate library.
It does not create measurements for absent genotypes.
All methods and generators search exactly the same retained genotypes.

Keep a query budget only when it uses at most 10% of the retained combinations.
Record every candidate and exclusion.
A target whose known singles already contain its measured optimum remains in the performance analysis.
Its ranking correlation is undefined and must remain visible.
The source table's measurement uncertainty does not enter this computational search evaluation.

## Search design

Use the existing twelve methods and their fixed settings.
The primary budget is 96 new queries, matching the manuscript's headline correlation.
The 48-query results are secondary and address the T7 comparison.
Each method starts with all retained WT and single-mutant observations.
New query counts include combinations only.

Run fifty fresh measured-search repeats per method, target and budget.
Use fifty measured repeats on each development protein to construct the fixed ranking.
Use five new Mutascape landscape samples and twenty search repeats per sample.
Run one hundred search repeats on each deterministic control.
Keep source seeds separate for development searches, synthetic selection and measured evaluation.
Pair search seeds across methods within each source.
Pair synthetic search seeds across models where their repeat indices overlap.
No old search outcome enters a new target's measured performance estimate.

Mutascape uses its default sigmoid, 5% margin and pair strength 0.3.
The additive control sums measured single effects without a nonlinear map or pair effects.
The zero-pair sigmoid control uses Mutascape's map and margin, with pair strength zero.
These two controls separate the value of the nonlinear map from the random interaction field.

Large libraries may require an equivalent dual calculation for the pairwise ridge surrogate.
Tests must compare its predictions against explicit one-hot and pair-indicator regression before use.
The frozen run manifest records the actual search implementation and all relevant source hashes.

## Method selection

Average normalised regret over search repeats and landscape samples.
Each selector chooses the method with lowest mean regret.
Means use twelve decimal places to retain numerical ties.
An exact tie defines a uniform choice over the tied methods.
Its measured performance is their average measured regret.

The fixed selector uses mean measured regret on the five development proteins, with equal protein weight.
It receives no measured combination results from any test protein.
The additive and zero-pair selectors use the target's WT and singles only.
The measured hindsight-best method supplies a descriptive reference, rather than a deployable control.

## Outcomes and uncertainty

The primary metric is the selected method's measured normalised simple regret at 96 queries.
Lower regret is better.
Report paired differences from the fixed selector and additive selector.
A supported improvement requires both 95% intervals to lie below zero on the new-protein panel.
Report a mixed or inconclusive result when this condition fails.
Never combine repeat-evaluation proteins with new proteins to strengthen that claim.

Report each protein separately and average proteins equally.
Report the fraction of remaining improvement missed as a secondary outcome when the starting genotype is suboptimal.
The within-ML analysis restricts selection to the eight model-based methods and remains secondary.
Keep all twelve methods in the primary analysis.

Bootstrap proteins as the independent units.
Within each protein, resample paired measured repeats and the synthetic search evidence used for selection.
Resample synthetic landscape samples and search repeats separately.
Also resample development proteins and their measured repeats when assessing uncertainty in the fixed selector.
Use 5,000 bootstrap replicates with fixed analysis seeds.
With few new proteins, intervals remain descriptive and cannot establish broad transfer across protein families.

Report Kendall correlation and all 66 measured pair differences as secondary diagnostics.
Give each pair its mean difference and pointwise 95% bootstrap interval.
No interval threshold determines whether a pair enters the primary outcome.
A descriptive equivalence margin is 0.01 of the measured fitness range.
An interval wholly inside [-0.01, 0.01] supports equivalence under that convention.
An interval crossing those limits remains uncertain.
This convention is not a universal biological threshold.

Report weighted pair disagreement without selecting pairs by significance.
Weight each pair by its absolute measured mean-regret difference.
An incorrect predicted ordering costs one and a predicted tie costs one half.
Divide the weighted cost by the total weight.
Measured ties have zero weight.
Report an undefined value when every measured method mean is equal.

## Reporting

Preserve the previous experiments and manuscript.
Save trial-level regret, query counts, source hashes and the selection made by each model.
Report unsuccessful outcomes and undefined metrics.
Document any protocol correction before running the affected searches.
Only code or data errors justify a correction after the manifest is frozen.
Changes motivated by outcomes require a separate experiment.
