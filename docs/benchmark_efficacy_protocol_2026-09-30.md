# Benchmark efficacy experiment

This exploratory experiment follows the code audit and the methods of Maheshwari et al. (2024).
The study compares absolute performance discrepancy, method ranks and the consequence of selecting a method.
It does not replace the earlier study retrospectively.

Use GB1, TrpB4, CreiLOV and eqFP611.
Construct one connected library of 4,096 measured genotypes per protein.
Start from WT and every supplied single, then add available neighbours through a seeded breadth-first traversal.
Use genotype coverage only, without fitness values, to choose the library.
Keep supplied singles available before search.
These connected libraries describe restricted search domains, rather than the full measured libraries.

Use the corrected forest ranks and search protocol `new-observations-v2`.
Model methods start with 32 new combinations and select further candidates in batches of at most 32.
Random search samples unknown candidates only.
If a search stops early, fill its remaining budget uniformly and record the number of fallback queries.
Run all twelve existing methods at budgets of 48 and 96 additional evaluations.

Use five landscape seeds and six paired search repeats per seed for each synthetic condition.
Run thirty measured campaigns per method and budget.
The five conditions are the default sigmoid model, zero-pair sigmoid, additive control, linear model and pooling model.
Keep the existing model-selection settings of strength one for the linear model and 0.3 for sigmoid and pooling models.
The ensemble averages method regret over sigmoid, linear and pooling conditions with equal weight.
No result from this experiment changes these conditions or weights.

The primary outcome is measured regret after method selection at 96 evaluations.
Compare the ensemble against the default model and against a fixed method chosen from the other three proteins.
Give equal weight to tied choices and to proteins.
The fixed control averages normalised measured regret across the other proteins.
The 48-evaluation outcome is secondary.

Report mean squared performance difference, Spearman correlation and Kendall correlation across method means.
Performance means normalised regret, rather than absolute assay units.
Use the same candidate library for measured and generated searches.
Normalise regret by each landscape's own fitness range.
Also report the remaining fraction of available improvement to expose differences hidden by this normalisation.
If a landscape has no available improvement, keep that second metric undefined.

Bootstrap measured search repeats, synthetic landscape samples and paired search repeats.
Retain common resampling indices across generator conditions.
Aggregate contrasts also resample proteins.
Repeat method selection inside each bootstrap sample.
Report undefined correlation counts and avoid replacing them with zero.

Report centred method performance across generator families as a structural preference diagnostic.
The paper's self-generation bias factor has no direct counterpart because Mutascape does not itself perform the search.
Do not label the centred diagnostic as that bias factor.

Record graph connectivity, available improvement, fallback queries and all source hashes.
Keep the original studies and the new outputs in separate directories.
The publication figures must use the corrected study and identify its exploratory scope.
