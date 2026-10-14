# Search-method selection with independent calibration samples

Combination-calibrated Mutascape does not show better method selection in this experiment.
Its average experimental regret is higher than the single-site generator, Mutascape additive and a fixed recommendation from other proteins.
The comparisons with the fixed recommendation and single-site generator remain uncertain across four proteins.
The calibrated result is close to choosing a method at random.

The experiment also identifies a specific failure in the current model.
In 31 of 40 calibration fits, the quantile map cannot produce a multi-mutant score above the best observation already available.
Every search method therefore receives the same predicted best observed fitness.
The experimental data contain better genotypes in every one of those cases.

## Experiment

Each of GB1, TrpB4, T7 and TEV supplied ten independently sampled sets of 96 multi-mutant measurements.
Each fit produced three landscapes, with six search repeats per landscape and all twelve search methods.
Experimental evaluation used eighteen independent campaign repeats per calibration set.
The search domain contained all experimentally covered genotypes supported by the supplied single-site measurements.

Searches started with WT, single-site measurements and the 96 calibration observations.
They then evaluated 48 or 96 further genotypes.
Every comparator received the same starting observations.
The single-site and additive generators used those observations during simulation but did not fit their generators to the multi-mutant values.
Only unobserved genotypes received synthetic fitness.
This comparison tests the value of fitting the generator, given the same experimental information.

The fixed recommendation used experimental campaigns on DHFR, eqFP611, ParB-Noc, ParD2 and CreiLOV.
These five training proteins did not enter the evaluation average.
Separate campaigns tested spending the entire budget directly on search, with no calibration stage.
Total budgets were 144 and 192 multi-mutant measurements beyond the existing WT and single-site data.

The completed study contains 75,168 campaigns and 5,785,344 newly evaluated genotypes across those campaigns.
The audit found no repeated new queries, unspent budgets, fallback queries or search warnings.
Calibration observations retained their experimental values in every simulated campaign.

## Experimental outcomes after choosing a method

Normalised simple regret measures the gap between the best fitness found and the best experimental fitness in the search domain.
The denominator is the experimental fitness range.
Zero means the campaign found the best available genotype.
Lower values are better.

| Recommendation | 48 further evaluations | 96 further evaluations |
|---|---:|---:|
| Combination-calibrated Mutascape | 0.305 | 0.279 |
| Single-site Mutascape with the same starting observations | 0.288 | 0.264 |
| Mutascape additive with the same starting observations | 0.288 | 0.262 |
| Fixed recommendation from five other proteins | 0.282 | 0.265 |
| Uniform random method choice | 0.308 | 0.284 |
| Random search | 0.347 | 0.343 |

Values average calibration sets within each protein, then give the four proteins equal weight.
Tied synthetic recommendations receive equal probability, without consulting experimental outcomes.
Random method choice differs from random search. It chooses uniformly among the twelve search methods, including the model-guided methods.

At 96 further evaluations, the calibrated model's regret exceeds the fixed recommendation by 0.0145.
The 95% hierarchical bootstrap interval is [-0.0073, 0.0405].
Its difference from the single-site generator is 0.0157, with interval [-0.0003, 0.0352].
Neither contrast supports an improvement from calibration.
The difference from Mutascape additive is 0.0174, with interval [0.0009, 0.0360], which favours the additive comparator in this study.

The difference from random method choice is -0.0041, with interval [-0.0204, 0.0015].
Beating the random-search method alone therefore does not establish useful method selection.
The same interpretation holds at 48 further evaluations.

| Protein | Calibrated | Single-site | Additive | Fixed recommendation |
|---|---:|---:|---:|---:|
| GB1 | 0.350 | 0.305 | 0.301 | 0.306 |
| TrpB4 | 0.209 | 0.193 | 0.190 | 0.198 |
| T7 | 0.235 | 0.233 | 0.232 | 0.230 |
| TEV | 0.324 | 0.324 | 0.324 | 0.325 |

This table uses 96 further evaluations and the same starting observations for every column.
Restricting selection to the eight model-guided methods does not change the conclusion.
Their calibrated mean regret is 0.269, compared with 0.263 for the single-site generator and 0.265 for the fixed recommendation.

## Accounting for the calibration cost

At a total cost of 192 measurements, calibration followed by search gives mean regret 0.279.
The fixed recommendation trained for 192 direct search evaluations gives 0.264 without collecting calibration data first.
The difference is 0.0151, with a 95% interval of [-0.0174, 0.0857].
At a total cost of 144 measurements, the corresponding values are 0.305 and 0.253.
These results do not support reserving 96 measurements for calibration as a way to improve method selection.

## Why distribution agreement did not translate into useful recommendations

| Protein | Fits predicting no improvement beyond starting observations | Fits with better experimental genotypes available |
|---|---:|---:|
| GB1 | 8/10 | 10/10 |
| TrpB4 | 7/10 | 10/10 |
| T7 | 8/10 | 10/10 |
| TEV | 8/10 | 10/10 |

The quantile map places synthetic combinations on the fitness distribution of the calibration sample.
Its linear upper tail extends that sample's upper quantiles, but the extension can still stop below a strong supplied single-site observation.
The supplied observation remains exact, so no generated combination can improve on it.
This limitation follows from the fitted map itself. It persists across random interaction seeds.

The audit evaluated the map at percentile one, its largest possible percentile input.
In exactly these 31 fits, that value was no greater than the best starting observation.
The three generated landscapes also showed zero available improvement in every such fit.
All twelve methods tied in predicted performance at both evaluation budgets.

Kendall correlation is undefined for those flat method rankings.
The remaining nine fits have a protein-balanced mean correlation of 0.222 at 96 evaluations.
That value describes only the minority of fits with a ranking, so it should not be presented as the model's overall ranking performance.

Distribution agreement nevertheless improved across the independent samples.
Mean scaled Wasserstein distances were 0.097, 0.121, 0.074 and 0.178 for GB1, TrpB4, T7 and TEV.
The corresponding single-site values were 0.590, 0.832, 2.190 and 0.916.
These comparisons exclude the calibration genotypes.
C2ST and Walsh spectra were not rerun in this experiment.
Matching the bulk of the fitness distribution did not retain realistic opportunities for further fitness improvement.

## Interpretation and next model change

The current calibrated model improves distributional agreement, but this experiment does not validate it as a method recommender.
Increasing the number of interaction seeds cannot repair a fitness map that rules out further improvement.

The next model revision should represent uncertainty in high-fitness combinations beyond the observed calibration range.
Its validation should include the fraction of combinations above the best supplied observation, alongside distribution and spectral measures.
That change needs a new method-selection experiment with independent samples and the same total-cost controls.
No generator changes or outcome-driven retuning were made during this experiment.

The protein panel is small and reuses the four manuscript evaluation datasets.
Independent calibration samples measure sensitivity to sampling within these datasets, rather than adding new biological replicates.
Bootstrap intervals repeat method selection and resample training proteins, evaluation proteins, calibration sets, landscapes and campaign repeats.
They do not measure assay uncertainty.
Separate intervals conditional on these four evaluation proteins are included in the CSV outputs.

## Files and reproduction

The study did not change the manuscript.
The commands below generate the linked result files locally.
Git tracks the code and protocol, while fitted models, source snapshots and experiment outputs remain local.

- [Protocol](calibrated_method_selection_protocol_2026-10-14.md)
- [Search results and uncertainty](../results/calibrated_method_selection_20261014/intervals.csv)
- [Paired comparisons](../results/calibrated_method_selection_20261014/contrasts.csv)
- [Individual recommendations](../results/calibrated_method_selection_20261014/selections.csv)
- [Quantile-map diagnosis](../results/calibrated_method_selection_20261014/headroom.csv)
- [Query audit](../results/calibrated_method_selection_20261014/query_audit.json)
- [Main comparison figure](../results/calibrated_method_selection_20261014/figures/method_selection_summary.png)
- [Equal-cost comparison figure](../results/calibrated_method_selection_20261014/figures/calibration_cost_comparison.png)
- [Recommendation stability figure](../results/calibrated_method_selection_20261014/figures/recommendation_stability.png)
- [Predicted improvement figure](../results/calibrated_method_selection_20261014/figures/no_predicted_improvement.png)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ./.venv/bin/python -m scripts.calibrated_method_selection prepare --jobs 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ./.venv/bin/python -m scripts.calibrated_method_selection run --jobs 6
./.venv/bin/python -m scripts.analyse_calibrated_method_selection
./.venv/bin/python -m scripts.audit_calibrated_fits
./.venv/bin/python -m scripts.plot_calibrated_method_selection
```

The full test suite passed with 648 tests, followed by 18 focused experiment tests after the final analysis test was added.
Repository lint and web checks passed.
An end-to-end analysis check confirmed that deliberately reversed synthetic recommendations produce worse experimental outcomes.
