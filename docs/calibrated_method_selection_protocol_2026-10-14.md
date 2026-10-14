# Combination calibration and search-method selection

## Question and outcomes

Does Combination-calibrated Mutascape recommend a search method that performs better on experimental fitness?
The primary outcome is experimental normalised simple regret after choosing a method from synthetic campaigns, at 96 further evaluations.
Lower regret is better. The 48-evaluation result and Kendall rank correlation are secondary outcomes.
Keep all results and keep the existing generator and search hyperparameters fixed.
Do not edit the manuscript.

## Data and independent samples

Use the four manuscript evaluation proteins, GB1, TrpB4, T7 and TEV.
Use every experimentally covered genotype supported by the same-assay WT and single-site measurements.
Retain the complete experimental search domain rather than selecting a smaller region.
For each protein, independently sample ten sets of 96 multi-mutant genotypes uniformly without replacement within each set.
Different sets can overlap, as expected under independent sampling from a finite dataset.
Selection uses genotype identities and a seed, without selecting on fitness.
Refit the existing public QuantileCalibratedLandscape for each sample.
Generate three landscapes per fit and run six search repeats per landscape.
Run eighteen independent experimental campaign repeats per calibration set.
Use separate seeds for calibration, synthetic landscapes, synthetic campaigns and experimental campaigns.

## Shared observations and budgets

Each calibrated campaign starts with WT, single-site measurements and the 96 calibration observations.
Every search method receives the same initial observations.
Initial observations retain their experimental values in synthetic simulations, including calibration observations that the fitted generator does not reproduce exactly.
Only unobserved genotypes receive generated values.
This conditions each simulated campaign on the data already collected.
The package generator remains unchanged.

Run all twelve existing methods with 48 and 96 further evaluations.
Count total additional experimental cost as 144 and 192 measurements beyond WT and single-site data.
Include the best calibration observation when calculating best observed fitness.
Record every queried genotype, assert that queries are distinct, and verify each budget.
Use the existing new-observations-v2 search implementation without changes.

## Comparators

Compare calibration with the single-site-only Mutascape generator and Mutascape additive.
Both controls receive the same initial experimental observations during campaigns but do not use multi-mutant data to fit their generators.
Use three landscapes and six repeats for the stochastic single-site generator.
Use eighteen repeats for deterministic Mutascape additive.
This separates the benefit of generator fitting from access to the calibration observations.

Learn a fixed method recommendation from DHFR, eqFP611, ParB-Noc, ParD2 and CreiLOV.
Those proteins do not enter the evaluation average.
For each of these five proteins, run three independent calibration sets with six experimental search repeats at each further-evaluation budget.
Give each training protein equal weight when averaging method regret.
Also report uniform random method choice and the random-search method with the same initial data and budget.

For the cost comparison, run eighteen experimental repeats without calibration at total budgets of 144 and 192.
These searches start with WT and single-site observations only.
Learn the corresponding fixed recommendation from the same five training proteins at the same total budgets.
This tests whether reserving 96 measurements for calibration helps compared with spending them during search.

## Selection and uncertainty

For each calibration set, choose methods by mean synthetic normalised regret across all landscape samples and search repeats.
Give equal selection probability to means tied at twelve decimal places.
Evaluate those recommendations using separate experimental campaign seeds.
The primary contrasts compare selected experimental regret with the fixed recommendation and the single-site generator under identical initial observations.
Average calibration sets within each protein, then give the four proteins equal weight.
Report each protein separately.

Use paired hierarchical bootstrap intervals that repeat method selection.
Resample calibration sets, then synthetic landscape samples and campaign repeats, and independent experimental repeats.
Resample training proteins and their calibration sets when estimating uncertainty in the fixed recommendation.
Report a fixed-four-protein interval and an interval that also resamples evaluation proteins.
These intervals describe computational variation and the limited protein panel, not uncertainty in assay measurements.
Report all twelve methods and the eight model-guided methods separately.
Use 5,000 bootstrap replicates.

## Records and checks

Freeze source hashes, input hashes, seeds, software versions and this protocol before production search.
Save fitted parameters, calibration identities, generated fitness, campaign results and query traces.
Record each dataset's starting fitness, available improvement and synthetic headroom.
Check supplied values and calibration exclusions directly.
Keep independent-sample results even when they disagree with the earlier distribution results.
Do not adjust the generator in response to this experiment.
