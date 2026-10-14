# A generator from measured singles

`from_dms` builds a generator from a wild-type sequence and same-assay WT and single-mutant measurements.
It does not need measured combinations, a donor prior or a protein structure.

```python
import pandas as pd
from mutascape import from_dms

sequence = "AAA"
singles = pd.DataFrame({
    "mutations": [{}, "A1C", "A2D", "A3E"],
    "fitness": [1.0, 1.2, 0.8, 1.5],
})
generator = from_dms(singles, wt_sequence=sequence)
landscape = generator.draw(seed=42)
fitness = landscape["A1C+A2D+A3E"]
```

Use the assay's fitness scale consistently.
Include WT and at least two measured single mutants, with variation across the supplied fitness values.
Mutation positions start at one.
By default, every queried substitution needs a measured single effect.
Optional [ESM-2 completion](esm_completion.md) estimates missing single-mutant effects through `from_dms(..., complete="esm2")`.
The loader rejects duplicate genotypes and nonfinite fitness.
Combination measurements enter through the optional `calibration` input, described in the [combination-calibration guide](quantile_generator.md).
Use `from_dms(singles, wt_sequence=sequence, calibration=combinations)` with the same sampling and query calls.
Omitting this argument keeps the single-site-only generator described here.

The default sigmoid extends 5% beyond the range of WT-centred measurements.
Its inverse maps each measured single effect to a latent contribution.
Each query sums these contributions and fixed random pair effects, then maps the result back to fitness.
WT and singles return their measurements exactly.

The default `strength=0.3` scales pair interactions by the standard deviation of latent single effects.
Set another value with `generator.draw(seed=42, strength=...)`.
Changing the seed changes the interactions without changing measured values.
Queries share coefficients, so their order does not change the answers.
The constructor and individual queries do not enumerate the complete library.

Five model-selection datasets selected the default hyperparameters.
The [research report](singles_only_research_results.md) also tests linear and pooling rules.
These experiments use the lower-level `DMSLandscape.from_singles` constructor, which exposes `mode`, `margin` and `contacts`.
Its `contacts` argument restricts site pairs for comparison experiments.
The manuscript model uses all supported site pairs.

The same `from_dms` entry point accepts CSV and FASTA paths, using `from_dms("singles.csv", fasta="wt.fasta")`.
The browser studio retains the earlier empirical-range implementation, available as `mutascape.legacy`.

These are synthetic interaction assumptions.
Landscape samples do not represent calibrated uncertainty about true unmeasured fitness.
[Combination-calibrated Mutascape](quantile_generator.md) can improve distributional agreement if measured combinations are available.

## Evaluating search methods

Use the same candidate library and known observations for every method.
Check that local search can reach candidates through single-substitution steps.
Also check whether unknown candidates can improve on the best supplied observation.
The benchmark helpers `library_connectivity` and `fitness_headroom` report these properties.

The corrected search protocol gives model-guided methods 32 new initial combinations.
Known WT and singles do not consume that initial design or the evaluation budget.
Any uniform random fallback is counted separately.

The [benchmark efficacy study](benchmark_efficacy_results_2026-09-30.md) tests connected libraries from four proteins.
It compares absolute performance discrepancy, method ranks and measured regret after method selection.
An ensemble averages method performance over sigmoid, linear and pooling models.
It does not average genotype fitness or change the default generator.
Its lower regret estimate remains uncertain, and additive landscapes give better mean rank agreement.
Use several generator assumptions to test sensitivity, then confirm conclusions with experimentally measured combinatorial datasets.
