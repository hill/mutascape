# Calibration confidence follow-up

This protocol was fixed before the new campaign results were inspected on 19 November 2026.
The existing results motivated this follow-up. They remain separate from its primary analysis.

## Question

Does combination calibration improve agreement with experimental search-method ordering across the four fixed test datasets?
The comparison uses GB1, TrpB4, RBD-S309 and PhoQ with equal protein weights.

## Design

Draw 40 fresh calibration samples per protein, each containing 96 multi-mutants.
Use the original eligible rows and exclude the original spectral cubes.
Freeze the selected single-site model and all search settings from the manuscript experiment.
Use all 12 methods at budgets of 48 and 96 new measurements.
Run 50 experimental repeats and 10 repeats on each of three landscapes per configuration.
Include the additive control with 10 repeats and the existing fixed ranking from the five model-selection datasets.

Use fresh random streams with root seed 2026111901.
Pair search seeds and starting observations across configurations.
Draw fresh single-site landscapes for each calibration sample, rather than sharing three landscapes across all samples.
This change includes landscape variation in the uncertainty estimate for both configurations.
Samples may share genotypes because each draw samples the same finite dataset independently.

Run 460,800 search campaigns. Do not stop early based on effect size or significance.
Do not tune generators or search methods on the test results.
Reuse prepared experimental data and frozen model-selection outputs with recorded SHA-256 hashes.
This follow-up repeats the complete search comparison. It does not repeat unrelated classifier or spectral experiments.

## Analysis

Use pair agreement at both budgets as the primary endpoints.
Use matched Kendall correlation as a secondary endpoint.
Retain the existing four-test Holm adjustment across these endpoints.
Calculate eligible experimental method pairs with the original 2,000 paired bootstrap draws.
Calculate configuration contrasts with 100,000 paired bootstrap draws and paired sign-flip tests.
Report all protein-specific effects, with the existing 16-test Holm adjustment.
Report undefined rankings and the number of matched observations.
If no experimental method pairs qualify, pair agreement is undefined. Exclude that sample from both configurations for this endpoint.
Keep valid zero agreement values. This missing-data rule was recorded before inspecting the new effect estimates.
Keep the original ten samples out of the primary analysis.

These tests concern repeated calibration and simulation on four fixed datasets.
Additional calibration samples do not add independent proteins or experimental measurements.
Protein-level sensitivity tests assess whether the four protein effects support a broader claim.
Report mixed or negative results without changing the endpoints.

## Reproduction

```sh
.venv/bin/python -m scripts.calibration_confidence prepare
.venv/bin/python -m scripts.calibration_confidence run --jobs 6
.venv/bin/python -m scripts.report_calibration_confidence
```

The reporting command audits every campaign and replays two fixed batches before calculating the contrasts.
Generated data, tables and figures stay under the ignored results directory.
