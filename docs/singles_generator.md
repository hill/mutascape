# A generator from measured singles

`SimpleLandscape` constructs a fixed synthetic landscape from a wild-type sequence and same-assay WT and single-mutant measurements.
It does not need measured combinations, a donor prior or a protein structure.

```python
import pandas as pd
from mutascape import SimpleLandscape

sequence = "AAA"
singles = pd.DataFrame({
    "mutations": [{}, "A1C", "A2D", "A3E"],
    "fitness": [1.0, 1.2, 0.8, 1.5],
})
landscape = SimpleLandscape.from_singles(sequence, singles, seed=42)
fitness = landscape["A1C+A2D+A3E"]
```

Use the assay's fitness scale consistently.
Include WT and at least two measured single mutants, with variation across the supplied fitness values.
Mutation positions start at one.
Every queried substitution needs a measured single effect.
The constructor rejects combination measurements, duplicate genotypes and nonfinite fitness.

The default sigmoid extends 5% beyond the range of WT-centred measurements.
Its inverse maps each measured single effect to a latent contribution.
Each query sums these contributions and fixed random pair effects, then maps the result back to fitness.
WT and singles return their measurements exactly.

The default `strength=0.3` scales pair interactions by the standard deviation of latent single effects.
Changing the seed changes the interactions without changing measured values.
Queries share coefficients, so their order does not change the answers.
The constructor and individual queries do not enumerate the complete library.

Five development proteins selected the default settings.
The [research report](singles_only_research_results.md) also tests linear and pooling rules through the `mode` argument.
The optional `contacts` argument restricts site pairs for comparison experiments.
The manuscript model uses all supported site pairs.

`from_dms` and the browser studio retain the earlier empirical-range implementation.
Their settings and numerical outputs differ from `SimpleLandscape`.
Use the import above to reproduce the manuscript model.

These are synthetic interaction assumptions.
Landscape samples do not represent calibrated uncertainty about true unmeasured fitness.
[Combination-calibrated Mutascape](quantile_generator.md) can improve distributional agreement if measured combinations are available.
