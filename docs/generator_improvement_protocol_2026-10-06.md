# Generator improvement experiments

This study tests changes to Mutascape after inspecting the external generator comparison.
The three existing proteins therefore form an exploratory panel.
No result from this panel can establish success on an untouched protein.
The manuscript and public generator remain unchanged.

## Candidate mechanisms

1. Widen the sigmoid limits from 5% to 100% of the measured single-site range.
   This tests whether the current ceiling makes synthetic campaigns too easy.
2. Scale random pair effects by the geometric mean of the two absolute single-site effects.
   This tests whether weak mutations should have weaker interactions.
3. Replace random pair effects with MSA-derived Potts interactions, while preserving measured single-site values.
   This model needs an MSA as well as DMS measurements.

The first two candidates use DMS measurements alone.
The third candidate uses the existing SLIP horizon-four interactions and converts their scale using the standard deviations of predicted and measured single-site effects.
The conversion uses the complete declared single-site set, without measured combinations.
The raw Potts fit with a non-negative regression scale and SLIP horizon eight provide sensitivity checks.
Regression uses an intercept across the single-site effects, but only its slope scales the interaction residual.

Two further DMS-only checks add negative interactions between beneficial single-site effects with an assumed epistatic horizon of four or eight.
They also retain random pair effects at strength 0.3.
The horizon controls a quadratic mean interaction and is not fitted to combinations.
A power-pooling exponent of 1.5 tests an intermediate choice between additive fitness and the existing exponent-two pooling model.
These are alternative assumptions, rather than interactions inferred from single-site measurements.

## Evaluation

Use the frozen GB1, CreiLOV and eqFP611 search spaces from the external generator comparison.
Keep its twelve methods, evaluation budgets, landscape seeds and campaign seeds.
Run five landscape samples with six campaign repeats each.
Fixed landscapes receive thirty campaign repeats.
Reuse the original measured campaigns and external controls without changing their values.

The primary outcome is protein-balanced measured regret after method selection at 96 evaluations.
Report 48 evaluations and method-rank correlation as secondary outcomes.
Report assay-scale C2ST and local Walsh spectra separately from method-selection utility.
Keep every candidate result, including failures.

Compare each candidate with the current default, additive control, SLIP, NK and Rough Mount Fuji.
Use paired hierarchical bootstrap intervals with selection repeated inside each sample.
Treat intervals as descriptive because the panel has already influenced model design.
Also select the DMS-only configuration using the other two proteins, then evaluate its recommendation on the excluded protein.
Repeat that entire selection process within the bootstrap.
This checks transfer of hyperparameter choice within this exploratory panel.
It does not make these proteins an untouched test set.

## Literature

[Otwinowski et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6094095/) motivate separating an additive latent trait from its nonlinear measurement map.
Their work does not identify the map from single-site measurements alone.
[Hsu et al.](https://www.nature.com/articles/s41587-021-01146-5) combine evolutionary and assay-labelled information for protein fitness prediction.
Our exact single-site correction is a related experiment, not a reproduction of their augmented regression model.
[Maheshwari et al.](https://arxiv.org/abs/2409.11968) assess synthetic benchmarks through downstream model performance and motivate checking multiple generators.
