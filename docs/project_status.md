# Project status

Updated 2026-09-23.

## Publication model

The manuscript presents `QuantileCalibratedLandscape` as Mutascape.
It creates fixed synthetic landscapes for combinatorial ML-assisted directed evolution benchmarks.
It does not predict the true fitness of unmeasured combinations or replace measured search benchmarks.

Inputs comprise a wild-type sequence, same-assay wild-type and single-mutant fitness, and a random sample of measured combinations.
The model fits mutation-depth terms and a shared random pair field.
It maps generated combination percentiles to the pilot fitness distribution.
Wild type and singles retain their measured values.
Pilot combinations fit the model but need not retain their measured values.
A seed fixes each generated landscape.

The Python import remains `mutascape.experimental.QuantileCalibratedLandscape` for compatibility.
The `from_dms` interface and browser studio retain their singles-only generator.
Exact enumeration limits the new model to one million declared variants by default.

## Five-protein evaluation

The evaluation includes DHFR, eqFP611, ParB-Noc, ParD2 and CreiLOV.
Each protein supplies one pilot of 192 measured combinations and 1,000 separate evaluation genotypes.
Three generator draws share that pilot.
A nested 48-combination pilot provides a measurement-budget sensitivity check.

At 192 pilot combinations, the mean of protein-median boosted C2ST AUCs is 0.679.
Adding sequence indicators gives 0.739.
Mean Wasserstein distance is 0.107 measured standard deviations.
The classifiers still detect differences between generated and measured examples.

Search comparisons use 12 methods, ten seeds, and budgets of 48 or 96 queries.
Candidates exclude wild type, singles and pilot combinations.
Mean pair agreement is 94.3 per cent, compared with 88.9 per cent for fixed rankings from other proteins.
Mean Kendall correlation is 0.653.
These percentages describe eligible method pairs and do not give the chance of selecting the best method.

Standard-generator controls use the same candidates, budgets and search seeds.
Mutascape has the largest mean Kendall correlation.
Potts has slightly higher mean pair agreement, at 94.7 per cent.
NK with K=1 reaches 93.4 per cent agreement and Kendall correlation 0.555.
NK with K=2 reaches 71.4 per cent and 0.314.
No model leads every protein, budget and metric.

The controls have different input requirements.
Additive, Potts and RMF receive singles only, while NK receives sequence alphabets without target fitness.
Mutascape also receives the measured pilot.
The comparison does not isolate model form from measurement cost.

## Structural limits

Spectrum distances are 0.688 for DHFR, 0.211 for eqFP611 and 0.436 for ParB-Noc.
CreiLOV's projected binary spectrum has distance 0.033.
Its signed pair-coefficient cosine similarity is only 0.089.
A close variance spectrum does not identify the correct interacting substitutions.
ParD2 has no reported spectrum because its library is incomplete.

Spectral calculations use full available libraries, including calibration genotypes.
C2ST and distribution distance use separate evaluation genotypes.
All five proteins appeared in earlier project work, so these tests do not establish transfer to previously unstudied families.
One pilot per protein leaves uncertainty across new calibration samples unmeasured.

## Manuscript and result sources

The manuscript retains all four figures and the full 31-assay development table.
Its earlier generator table now labels the singles-only reference explicitly.
A separate table compares the final model with Additive, Potts, RMF and both NK settings.
Bold values mark the largest displayed value for each metric within a row, including ties.
The search and CreiLOV figures include error bars with their interpretation in the captions.

The versioned summary is `src/mutascape/data/validation/quantile-confirmation-v1.json`.
Local trial records remain under `results/quantile_confirmation_20260923` and `results/quantile_generator_comparison_20260923`.
The [result report](quantile_calibration_2026-09-23.md) gives the development history and reproduction commands.
The [input guide](quantile_generator.md) describes the Python API.

Run `uv run python -m scripts.rebuild_preprint` to rebuild manuscript tables, figures and the PDF from saved results.

## Next research

1. Evaluate new protein families under a fixed protocol.
2. Repeat pilot selection to estimate variation across calibration samples.
3. Compare pilot measurements with direct optimisation under an equal total experimental budget.
