# Search selection on new proteins

## Result

Mutascape did not pass the predeclared test for better method selection.
The experiment completed 56,400 search runs.
Every run spent its full query budget, and no search emitted a warning.
The primary comparison uses two new proteins, Pab1 and Ube4b, at 96 queries.

| Selector | Measured regret at 48 queries | Measured regret at 96 queries |
|---|---:|---:|
| Mutascape | 0.0640 | 0.0479 |
| Additive | 0.0901 | 0.0453 |
| Zero-pair sigmoid | 0.0901 | 0.0453 |
| Fixed development selector | 0.0917 | 0.0409 |

These values average the two proteins equally.
They measure missed fitness as a fraction of each measured fitness range.
Lower values are better.

At 96 queries, Mutascape's difference from the fixed selector is +0.0070, with a 95% interval of [-0.0536, 0.0170].
Its difference from additive selection is +0.0025, with interval [-0.0006, 0.0055].
The protocol requires both intervals below zero.
Neither comparison passes.
These results do not establish that Mutascape is worse either.
They establish no supported advantage under this experiment.

The secondary 48-query result favours Mutascape numerically.
Its differences from fixed and additive selection are -0.0276 and -0.0260.
Their intervals are [-0.0664, 0.0039] and [-0.0584, 0.0000].
This result also fails the strict improvement criterion.

![Measured performance after method selection](../results/search_selection_20260924/selection_regret.png)

The points show Mutascape's measured regret minus each control's regret.
Negative differences favour Mutascape.
Lines show 95% bootstrap intervals that include uncertainty in method selection.
The new and repeat-evaluation panels remain separate.

### Where it helps

On Ube4b at 48 queries, Mutascape selects ridge-greedy, which also has the best measured mean performance.
Its measured regret is 0.0366, compared with 0.0937 for the fixed selector.
The difference is -0.0571, with interval [-0.0734, -0.0026].
This is a positive result for one protein and budget.
Against additive selection, the interval reaches zero.

At 96 queries, that advantage disappears.
Mutascape ties four methods and has expected measured regret 0.0180.
The fixed selector chooses GP-UCB and reaches 0.0040.
Pab1 shows little difference between selectors at either budget.
Its measured method ordering also remains unstable across repeat subsets.

Restricting selection to eight ML methods gives a small numerical advantage at 96 queries.
On the new proteins, Mutascape reaches regret 0.0398, compared with 0.0409 for fixed selection and 0.0401 for additive selection.
Both difference intervals cross zero.
The ML-only analysis therefore does not change the main conclusion.

### Ranking agreement

| Panel | Queries | Mutascape Kendall tau | Additive | Fixed development ranking |
|---|---:|---:|---:|---:|
| New proteins | 48 | 0.046 | 0.099 | 0.121 |
| New proteins | 96 | 0.230 | 0.241 | 0.320 |
| Repeat evaluation | 48 | 0.171 | 0.164 | 0.029 |
| Repeat evaluation | 96 | 0.334 | 0.304 | 0.337 |

At 96 queries, Mutascape reaches tau 0.779 on GB1 and 0.687 on TrpB4.
It reaches 0.185 on T7 and -0.313 on TEV.
The new proteins also differ, with -0.224 on Pab1 and 0.683 on Ube4b.
Positive agreement on some proteins does not establish a reliable selection advantage across proteins.

The repeat-panel mean of 0.334 is below the earlier ten-repeat estimate of 0.582 at 96 queries.
These are fresh search repeats and five synthetic samples.
The change exposes sensitivity to search and landscape sampling.
Its recorded arithmetic remains correct.

### T7 and the uncertainty rule

More repeats resolve the original lack of eligible T7 comparisons without weakening the rule.
Using pointwise 95% intervals, 21 of 66 T7 pairs exclude zero at 48 queries.
Mutascape orders all 21 in the opposite direction.
At 96 queries, 24 pairs exclude zero and Mutascape orders 12 correctly.
These pair counts remain descriptive because the intervals lack multiple-testing correction.
The primary selection outcome uses every method and no eligibility filter.

T7's median split-half tau is 0.304 at 48 queries and 0.200 at 96.
These values compare random groups of twenty-five repeats.
More repeats expose a mismatch at 48 queries, rather than supporting looser evaluation rules.
TEV also separates some methods after fifty repeats.
Its Mutascape ranking is negative at both budgets.

The weighted pair disagreement uses every real pair difference.
At 96 queries, its new-protein mean is 0.337 for Mutascape, 0.341 for additive selection and 0.320 for the fixed ranking.
Zero means no weighted ordering error and one means complete reversal.
This diagnostic also provides no clear advantage for Mutascape.

### Recommendation

Do not use reliable transfer of search rankings as the paper's headline claim.
The new experiment supports isolated successes and mixed transfer.
Keep the claim that Mutascape supplies repeatable synthetic benchmarks under explicit landscape assumptions.
State that choosing methods for a new measured protein still needs validation.
A broader reliability claim needs a new generator hypothesis and further protein-level tests.
Changing the significance cutoff would not address the observed reversals.

## Experiment

The [protocol](search_selection_protocol_2026-09-24.md) fixes the models, controls and primary comparison before new searches.
The main question concerns measured performance after choosing a method through Mutascape.
The generator receives the target WT and measured singles.
Its parameters remain unchanged.
No measured target combinations enter its fit or method choice.

The experiment uses twelve existing search methods at 48 and 96 combination queries.
Each method starts with the same WT and single-mutant observations.
Those known measurements cost no new queries.
The primary budget is 96.
The eight-method ML subset supplies a secondary analysis.

Each measured landscape has fifty fresh search repeats per method and budget.
Mutascape has five landscape samples, each with twenty search repeats.
Each deterministic control has one hundred search repeats.
Development, synthetic selection and measured evaluation use separate seed sources.
Methods within each source share trial seeds.

The additive control sums measured single effects on the assay scale.
The zero-pair sigmoid uses Mutascape's nonlinear map with interaction strength zero.
The fixed selector learns one method ordering from DHFR, eqFP611, ParB-Noc, ParD2 and CreiLOV.
Each development protein receives equal weight.
The fixed selector sees no measured combinations from any test protein.

Each selector chooses the method with lowest mean normalised regret.
An exact tie gives each tied method equal probability.
Evaluation averages their measured performance.
Method means use twelve decimal places before tie handling.

Normalised regret is the gap between the best measured candidate and the best queried candidate, divided by the measured fitness range.
Lower values mean less missed fitness.
The secondary measure divides that gap by the improvement available beyond the best supplied WT or single mutant.
This second measure remains undefined when the supplied observations already contain the measured optimum.

## Protein selection and data

The new datasets come from the [NN4DMS author archive](https://github.com/gitter-lab/nn4dms/tree/434e354828f306cbb6796033ef1169cbe5207350).
The pinned revision is `434e354828f306cbb6796033ef1169cbe5207350`.
Dataset selection uses genotype coverage without comparing fitness outcomes.
Each retained substitution requires a same-assay single-mutant measurement.
The sixteen positions with most supported measured combinations define each candidate library.
All methods and generators query the same measured genotypes.

| Candidate | Measured singles | Measured combinations | Included budgets | Decision |
|---|---:|---:|---|---|
| Pab1 | 275 | 2,856 | 48, 96 | Include |
| Ube4b | 193 | 3,556 | 48, 96 | Include |
| Bgl3 | 141 | 316 | None | Insufficient combinations |
| avGFP | 109 | 290 | None | Insufficient combinations |

The rule requires at least ten candidate combinations per query.
These two new proteins supply two independent protein units.
Their many search repeats reduce simulation noise without increasing that number.
The results concern sparse measured libraries.
They cannot establish transfer across every possible sixteen-site genotype or across broad protein families.

Pab1 uses the archive's log2 enrichment scores.
The [source study](https://rnajournal.cshlp.org/content/19/11/1537.full.pdf) sets WT enrichment to one, which becomes zero after this transform.
Ube4b uses the original spreadsheet's corrected log2 score for its `NA-NA` reference row.
The manifest records both WT values and the source file hashes.
Archive coordinates map to one-based positions in the supplied reference domain.
The parser checks reference residues before accepting substitutions.

GB1, TrpB4, T7 and TEV form a separate repeat-evaluation panel.
We knew their earlier search outcomes before this experiment.
They do not enter the new-protein mean.

## Uncertainty

The primary outcome uses measured performance without a pair-eligibility filter.
All 66 method pairs also receive a measured effect estimate and a pointwise 95% interval.
These intervals describe individual pairs and do not correct for multiple testing.
An interval wholly within one percentage point of the measured fitness range supports practical equivalence under the recorded convention.
The convention supplies a descriptive scale, rather than a universal biological threshold.

The analysis uses 5,000 bootstrap replicates.
It resamples proteins, paired measured repeats and the synthetic evidence used for selection.
Mutascape resampling separates landscape samples from search repeats.
Each sample travels with its matching control search-seed block.
Each selected block then resamples paired search repeats across all three generators.
This retains shared seed variation in their comparisons.
The fixed selector also resamples development proteins and their measured repeats.
Each replicate repeats method selection before calculating measured performance.
The intervals include uncertainty about which method the selector chooses.

Two new proteins give limited information about variation between proteins.
These intervals remain descriptive.
The analysis does not model experimental uncertainty in the original assay measurements.

## Implementation and reproducibility

The generator's fit and query equations did not change.
Large categorical libraries use an equivalent kernel calculation for pairwise ridge search.
The calculation counts matching sites and matching site pairs.
Tests compare its predictions against explicit indicator regression, including unseen categories and a fitted intercept.
Small designs retain the previous implementation.
Lexicographic genotype ordering avoids integer overflow from the theoretical Cartesian library size.

The experiment branch is `codex/search-selection-validation-2026-09-24`.
Protocol commit `b67a48b` precedes runner commit `9e20f0f` and analysis commit `0060a61`.
Commit `a4ca0c1` keeps matched synthetic seeds together in resampling.
That analysis correction precedes inspection of the completed search outcomes.
The frozen run manifest has SHA256 `312d7214f964d315aaf105f6270929ec1c3be33ffdd9c93c1bf120f89363b1ab`.
It records input hashes, source hashes, package versions and random seeds.
The analysis manifest records hashes for every trial file.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.search_selection_validation prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.search_selection_validation freeze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.search_selection_validation run --jobs 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.analyse_search_selection
.venv/bin/python -m scripts.plot_search_selection
```

Preparation reads the existing development and repeat-panel inputs from `results/singles_only_restart_20260923/`.
It downloads the pinned new-protein archive files when absent.
Run preparation once in an empty experiment folder.
Later run commands resume completed search chunks after verifying frozen files.
The results remain under `results/search_selection_20260924/` and outside version control.
The manuscript remains unchanged.

The full Python suite passes all 588 tests.
Ruff and the browser checks also pass.
The figure uses the saved contrast table and includes uncertainty intervals.
