# Mutascape preprint handover

Date: 2026-09-23

## Objective

Prepare a simple, reproducible preprint about synthetic landscapes for MLDE and ALDE benchmarks.
Keep the current singles-only model as the main contribution.
Judge usefulness through transfer of search-method comparisons, with C2ST and landscape shape as diagnostics.
Do not promise accurate fitness predictions or indistinguishable synthetic data.

## Locations and state

| Item | Location or reference |
|---|---|
| Main checkout | `/Users/tomhill/Uni/research-project/mutascape` |
| Current branch and HEAD | `codex/creilov-adversarial-model`, `6cc2f46` |
| Older research worktree | `.mutascape/worktrees/posterior-landscapes-v1`, branch `research/benchmark-validity-v1` |
| Current manuscript | `/Users/tomhill/Uni/research-project/mutascape_manu_v1/main.typ` |
| Original short manuscript | `/Users/tomhill/Uni/research-project/mutascape_manu_v1/OLD_1000w_technical_note/main.typ` |
| ESM completion | `docs/esm_completion.md`, commit `4d14320` |
| C2ST protocol and replicate controls | `docs/c2st_audit.md`, `scripts/audit_creilov_c2st.py` |
| Diagnosis and pilot results | `docs/creilov_model_diagnosis.md` |
| Pilot confirmation artefacts | `results/creilov_pilot_readout_burden_confirmation/` |
| SLIP paper | `/Users/tomhill/Papers/SLIP.pdf` |

Resolve relative paths against the main checkout.
The checkout contains another agent's refactor, staged deletions, and uncommitted experiments.
Preserve all existing work. Do not reset, move the manuscript, or commit unrelated changes together.
The status notes report 543 Python tests and 34 web tests passing previously. This handover did not rerun them.

## Current evidence

- The public model preserves measured anchors, adds seeded pair interactions, and applies a saturating readout informed by other proteins.
- Optional calibrated ESM-2 completion already exists. It fills missing singles, not missing combinatorial measurements.
- At 48 queries, reported method-pair ordering agreement is 89.5% for SSMuLA, 69.1% for CombinGym, and 73.5% for GraphFLA.
- These percentages concern statistically distinguishable pairs. They do not mean the best method wins that often.
- Later budgets give mixed results. The collections overlap in proteins and have informed development.
- CreiLOV standardised Wasserstein distance is 0.047. Projected Walsh distance is 0.057, versus 0.254 without pair interactions.
- Raw CreiLOV error remains 3.94 measured standard deviations. Raw C2ST AUC exceeds 0.999.
- Input assays disagree. Shared singles have correlation 0.131, and the model misses the negative combinatorial trend.
- Corrected replicate AUCs are about 0.51 for replicates 1/2 and 0.65 to 0.67 for comparisons involving replicate 3.
- Neither the old 0.699 claim nor the proxy median 0.820 defines a valid acceptance threshold.

A separate prototype uses 48 measured combinations to fit burden terms and a bounded readout.
Its confirmation medians are C2ST AUC 0.595/0.619, raw Wasserstein 0.062, and method-pair agreement 87.2%.
These results remain within CreiLOV. They cannot support a singles-only claim or independent-protein validation.
Keep this prototype out of the public default and main headline.

## Evaluation metrics

Keep this metric set fixed for the next assessment.

| Role | Metric | Meaning and interpretation |
|---|---|---|
| Primary benchmark usefulness | Method-pair ordering agreement | Fraction of statistically distinguishable measured method pairs whose ordering the synthetic benchmark preserves. Higher is better. Report eligible pair counts. |
| Supporting benchmark usefulness | Kendall rank correlation | Agreement between method rankings on measured and synthetic landscapes. Higher is better. |
| Underlying search outcome | Normalised regret at each query budget | Gap between the best fitness found and the candidate-domain optimum, under the protocol's normalisation. Lower is better. |
| Adversarial diagnostic | C2ST ROC AUC and balanced accuracy | Held-out discrimination between measured and synthetic samples. Report logistic and boosted-tree classifiers separately. Chance is 0.5. |
| Assay-scale diagnostic | Raw Wasserstein-1 distance | Difference between fitness distributions, including location and scale. Report fitness units and measured-standard-deviation units. Lower is better. |
| Distribution-shape diagnostic | Standardised Wasserstein-1 distance | Difference after separate location and scale standardisation. Lower is better, but this removes assay-scale errors. |
| Interaction-order diagnostic | Projected Walsh-spectrum L1 distance | Difference between normalised variance fractions across interaction orders. Lower is better. This does not test genotype-level coefficient agreement. |

Compare all 12 search methods at 48, 96, 192, and 384 queries on matched candidate sets.
Give each protein equal weight within its collection.
Define distinguishable pairs through paired bootstrap intervals for measured method differences that exclude zero.
Report uncertainty and sensitivity to search seeds and oracle draws. Repeated pairs across draws are not independent observations.

For C2ST, keep each genotype's measured and synthetic observations in the same fold.
Fit preprocessing on training folds only. Use within-pair label swaps for permutation tests.
Report discrimination effect sizes alongside permutation p-values and identical-copy and real-replicate controls.
A score near 0.5 does not prove equivalence. Replicate scores do not define a universal acceptance threshold.

For CreiLOV Walsh analysis, average amino-acid variants within each mutated-site pattern before the transform.
This projection describes site-pattern structure, rather than the full amino-acid landscape.
Keep local-optimum counts and new neighbourhood or trajectory classifiers outside the required headline set.
Assess optional ESM completion separately with hidden singles and hidden positions against mean-based controls.

## Next steps, smallest first

1. Read the manuscript, local instructions, TODO, and linked audit notes. Inventory staged and unstaged changes before editing.
2. Replay the saved C2ST audit. Check paired folds, training-only scaling, permutations, and replicate normalisation against saved metadata.
3. Trace every headline manuscript number to its script and result file. Add one command to rebuild current publication outputs.
4. Run `just test`, `just lint`, and `just web-check`. Separate refactor failures from scientific changes.
5. Freeze the model, classifiers, budgets, and reporting rules. Select one unused protein for confirmation before viewing its results.
6. Compare method ordering against additive and existing random baselines. Report every fixed budget and uncertainty across proteins where feasible.
7. Revise the manuscript around those results. Retain CreiLOV's failure, assay mismatch, and development-set limitations.
8. Update TODO and documentation to match the paper. Produce a short review report with commands, artefacts, and unresolved issues.

## Scope and stopping rule

SLIP tests tunable prediction difficulty using synthetic labels. It does not directly validate agreement with measured combinatorial fitness landscapes.
Our random Potts baseline is not an implementation of alignment-fitted SLIP. Make no superiority claim over SLIP.

Defer new model families, extra adversarial metrics, and a public pilot-calibration API.
Do not tune repeatedly against confirmation data until results look favourable.
Finish when the frozen model and reported claims reproduce, tests pass, and the manuscript states the limits plainly.
If confirmation fails, narrow the claim and report the failure.
