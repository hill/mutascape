# Mutascape, SLIP and NK comparison

The experiment completed 70,560 simulated directed evolution campaigns on GB1, CreiLOV and eqFP611.
The results support using several Mutascape configurations to identify useful search methods in these search spaces.
They do not establish that the current sigmoid default outperforms additive landscapes, SLIP or NK.

The strongest positive result concerns the existing configuration average.
It averages each search method's performance across sigmoid, linear and pooling landscapes with equal weights.
This procedure selected methods with lower measured regret than random method choice at both evaluation budgets.
Both paired 95% bootstrap intervals exclude zero.
Its comparisons with additive landscapes, SLIP, NK and a fixed recommendation from other proteins remain inconclusive.

The public generator and manuscript remain unchanged.
The experiment code and outputs are on `codex/slip-nk-comparison`, without new commits.

## Experimental design

Each generator supplied fitness values for exactly the same genotypes within a search space.
All twelve search methods received the same starting genotype identities and evaluation budget.
Each search used its own landscape's fitness at those identities.
Measured multi-mutant fitness was reserved for reference campaigns and evaluation.

GB1 and CreiLOV each supplied three connected search spaces containing 4,096 genotypes.
The eligible eqFP611 search space contained 2,048 genotypes, and we used it in full.
The starting observations were wild type and the available single-site variants.
Every method received 48 or 96 additional evaluations and 30 campaign repeats.
Stochastic generators used five landscape samples with six repeats each.
Fixed models used 30 repeats of the same fitted landscape.

The [protocol](external_generators_protocol_2026-10-06.md) specifies the settings and primary outcome.
The primary outcome is measured regret after choosing a method using synthetic campaigns at 96 evaluations.
Proteins receive equal weight.
The confidence intervals resample proteins, search spaces and paired campaign blocks, with method selection repeated inside each bootstrap sample.
They exclude experimental measurement uncertainty and uncertainty from fitting another alignment model.

![Comparison design](../results/external_generators_20261006/figures/comparison_design.png)

## Search outcomes

Regret measures the gap to the highest measured fitness, divided by the fitness range of the measured search space.
Lower values mean that the recommended method finds better measured variants.
The separate improvement measure divides the same gap by the improvement available beyond the starting observations.
These measures give different relative weights to the proteins, so both appear below.

| Benchmark | Regret at 48 | Regret at 96 | Improvement missed at 96 | Kendall correlation at 96 |
|---|---:|---:|---:|---:|
| Mutascape | 0.0429 | 0.0114 | 5.40% | 0.375 |
| Mutascape additive | 0.0403 | 0.0074 | 3.56% | 0.647 |
| Mutascape zero-pair sigmoid | 0.0392 | 0.0092 | 4.37% | 0.393 |
| Mutascape linear + pairs | 0.0334 | 0.0057 | 3.39% | 0.569 |
| Mutascape pooling | 0.0278 | 0.0064 | 3.81% | 0.609 |
| Mutascape configuration average | 0.0330 | 0.0061 | 4.22% | 0.574 |
| SLIP, horizon 4 | 0.0365 | 0.0073 | 3.51% | 0.713 |
| NK, K = 1 | 0.0273 | 0.0056 | 3.62% | 0.626 |
| Rough Mount Fuji | 0.0972 | 0.0288 | 9.53% | 0.310 |
| Fixed method from other proteins | 0.0803 | 0.0089 | 4.48% | 0.576 |
| Random method choice | 0.0781 | 0.0316 | 13.93% | Not defined |

The configuration average reduces mean regret from 0.0316 for random method choice to 0.0061 at 96 evaluations.
The paired difference is -0.0256, with a 95% interval of [-0.0573, -0.0013].
Against the fixed recommendation, the difference is -0.0028, with an interval of [-0.0122, 0.0098].
That comparison does not establish an advantage.

At 48 evaluations, the corresponding mean regrets are 0.0330 for the configuration average and 0.0781 for random method choice.
Their difference is -0.0451, with an interval of [-0.0848, -0.0031].

Mutascape linear with pairs has the lowest mean selected-method regret among the individual Mutascape configurations at 96 evaluations.
Its value is close to NK with K equal to one.
This makes the linear configuration a candidate for further validation, but these three proteins do not justify declaring it a general winner.

![Search comparison](../results/external_generators_20261006/figures/search_comparison_96.png)

The separate [48-evaluation figure](../results/external_generators_20261006/figures/search_comparison_48.svg) reports the secondary budget.
The [protein-level figure](../results/external_generators_20261006/figures/selection_by_protein.svg) shows where the averages differ.
All search error bars show 95% bootstrap intervals.

## What the recommendation means

On eqFP611, the default Mutascape configuration recommends random forest search.
The configuration average recommends GP UCB.
Their measured campaigns miss 3.13% and 1.35% of the available improvement, respectively.
These values illustrate one protein and should not be interpreted as a universal ordering of those search methods.

![Method selection example](../results/external_generators_20261006/figures/method_selection_example.png)

## Method rankings

Tuned SLIP gives the highest mean Kendall correlation among the primary external comparators.
Its value is 0.713, compared with 0.375 for default Mutascape, 0.647 for Mutascape additive and 0.626 for NK with K equal to one.
SLIP with horizon eight, a sensitivity setting, reaches 0.743.

Agreement weakens when the comparison contains only the eight model-guided search methods.
Mean Kendall correlation then becomes 0.163 for default Mutascape, 0.284 for Mutascape additive, 0.429 for SLIP and 0.360 for NK.
Some of the agreement across all twelve methods therefore reflects broad differences between search families.
The [method-rank figure](../results/external_generators_20261006/figures/method_rankings.svg) shows the full pattern, including tied methods.

## Fitness distributions and Walsh spectra

Default Mutascape gives a mean local Walsh distance of 0.163.
The corresponding distances are 0.383 for Mutascape additive, 0.361 for SLIP, 0.177 for NK and 0.398 for Rough Mount Fuji.
Lower values indicate closer agreement in the fraction of variance assigned to each interaction order.
Mutascape zero-pair sigmoid reaches 0.165, so the random pair terms add little to this aggregate spectral result.

Distribution agreement remains poor.
Mean C2ST AUC is 0.958 for default Mutascape, 0.964 for Mutascape additive, 0.910 for Mutascape linear with pairs and 0.887 for Mutascape pooling.
The classifier can therefore distinguish the measured and generated examples readily.
This C2ST uses fitness and mutation depth, five grouped folds, and all multi-mutant genotypes in each evaluated search space.
The measured and generated observations for each genotype stay together in one fold.
These values describe the restricted search spaces in this study and are not directly interchangeable with earlier C2ST results.

Native SLIP, NK and Rough Mount Fuji have arbitrary score units.
Their raw assay-scale C2ST would largely test a units mismatch, so we compare their topology and search outcomes instead.

Similar Walsh spectra can coexist with different fitness distributions.
The normalised spectrum omits the mean and absolute variance, whereas C2ST can detect those differences.
The spectra below use 256 complete three-site subcubes per search space.
They describe local interactions and do not establish agreement across the full protein landscape.

![Fitness distributions and local spectra](../results/external_generators_20261006/figures/fitness_and_walsh.png)

The upper panels show cumulative fitness distributions for search space zero.
Shading shows the range across landscape samples.
The lower panels average local spectra over search spaces and landscape samples.
Their error bars show the full observed range, rather than confidence intervals.

## SLIP implementation and alignments

We fitted the authors' [Mogwai Gremlin model](https://github.com/songlab-cal/mogwai) and checked scores against the [official SLIP implementation](https://github.com/google-research/slip).
The full aligned background remains in the score calculation when only a subset of sites can mutate.
Scores agree with the reference implementation within 0.0000034 on the checked genotypes.
The small difference comes from the reference implementation's float32 offset calculations.

| Protein | Alignment sequences | Aligned positions | Experimental sites retained | Experimental sites excluded |
|---|---:|---:|---:|---|
| GB1 | 387 | 55 | 4 | None |
| CreiLOV | 51,241 | 106 | 11 | 3, 4, 5, 113 |
| eqFP611 | 1,093 | 211 | 11 | 172, 231 |

These alignments were already available in the local CombinGym data.
The larger GB1 alignments omit experimental position 54, so we used the smaller alignment that covers all four experimental sites.
Excluded positions remain wild type for every generator and the measured comparison.

The initial zero-offset tuning rule fails on the GB1 fit because only one single-site substitution is beneficial.
It also reverses coupling signs on eqFP611.
Before production search runs, we changed to SLIP's published shift controls for every protein.
The settings target 10% beneficial single-site substitutions, 80% negative pair effects among beneficial substitutions, and an epistatic horizon of four.
They use the fitted alignment model and do not inspect experimental multi-mutant fitness.
We also retained the untuned model and horizon eight.
The untuned model offers no improvement beyond the starting observations on GB1 or CreiLOV, so its method-rank correlations are undefined there.

The approach follows [Tuned Fitness Landscapes for Benchmarking Model-Guided Protein Design](https://doi.org/10.1101/2022.10.28.514293).
Our proteins and alignments differ from the authors' original evaluation panel.

Your supervisor's AlphaFold suggestion is correct.
[AlphaFold Server](https://alphafoldserver.com/) now includes generated MSAs in its downloadable results.
Those alignments can supply the input to SLIP after residue mapping and coverage checks.
Structure prediction is not required for this comparison, and no AlphaFold jobs were submitted.

## Interpretation and next decision

These results support controlled simulated campaigns under explicit landscape assumptions.
They also provide limited positive evidence for choosing useful search methods by averaging Mutascape configurations.
They do not support superiority of the sigmoid default or reliable selection among closely performing model-guided methods on a new protein.

Keep the additive comparator in the manuscript and retain SLIP and NK in the evaluation.
Validate the linear and pooling configurations on additional proteins before changing the public default.
The present comparison gives a practical reason to expose those configurations, while its C2ST results argue against claiming realistic assay distributions.

These exploratory results cover only three proteins.
The GB1 and CreiLOV search spaces overlap and favour variants near wild type.
The three CreiLOV spaces share about 90% of their genotypes, so they provide limited evidence about distinct regions.
MSA coverage also removes experimental sites from two proteins.
These are matched comparisons within the stated search spaces, rather than a validation across entire proteins.
SLIP receives an alignment, whereas Mutascape receives measured single-site fitness.
The comparison therefore evaluates practical input configurations rather than methods with identical information.

## Verification and files

All 70,560 campaigns spent their full budgets without fallback queries or search warnings.
An independent audit recalculated all outcomes from 5,080,320 saved query identities.
It found no duplicate charged queries, charged starting observations or score discrepancies.
All 612 Python tests pass.
Python and browser lint checks pass, along with all 34 browser tests and the browser build.

The [result directory](../results/external_generators_20261006/) contains the following files.

| File | Contents |
|---|---|
| `manifest.json`, `run_manifest.json` | Frozen inputs, settings, source hashes and the original execution manifest |
| `analysis_audit.json`, `query_audit.json` | Campaign completeness and independent outcome checks |
| `trials.csv` | Every campaign outcome |
| `intervals.csv`, `contrasts.csv` | Bootstrap intervals and paired comparisons |
| `per_search_space.csv`, `ml_only.csv` | Individual search spaces and the model-guided sensitivity analysis |
| `properties.csv` | Distribution, graph and spectral metrics |
| `figures/` | Seven figures and diagrams, each in PNG and SVG |

A formatting-only change after execution has an identical Python syntax tree.
The original source and manifest remain alongside the amendment record.
Model weights and upstream repositories remain in the ignored local experiment cache.

Reproduction commands appear in `docs/external_generators_reproduction_2026-10-06.md`.
