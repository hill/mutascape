# Combination-calibrated Mutascape

Combination-calibrated Mutascape uses measured combinations to fit depth effects, interaction strength and a fitness-distribution map.
Pass the combination measurements to the optional `calibration` argument of `from_dms`.
Use a uniform sample of the intended search space, with WT and single-site measurements from the same assay.

```python
from mutascape import from_dms

generator = from_dms(
    "singles.csv", fasta="wt.fasta", calibration="combinations.csv",
)
landscape = generator.draw(seed=42)
fitness = landscape["A102G+P20Y"]
```

Both inputs accept a DataFrame or CSV path.
For tables and a sequence string, use `from_dms(singles, wt_sequence=sequence, calibration=combinations)`.
The single-site file follows the `from_dms` formats.
The calibration table has `variant` and `fitness` columns, or `mutations` and `fitness` columns with mutation labels or dictionaries.
Each calibration genotype must contain at least two substitutions, and every substitution needs a single-site measurement.
Supply at least three distinct calibration genotypes with enough variation to fit both mutation-depth terms.
The loader rejects invalid measurements and duplicate genotypes.

Omitting `calibration` returns the single-site-only generator.
The older `pilot` argument remains an alias. Supplying both arguments raises an error.
Combination calibration does not support `complete="esm2"` or `maximum_depth`.

The model fits the mutation-depth terms and sigmoid used by `PilotCalibratedLandscape`.
For each landscape sample, it estimates generated fitness ranks from a fixed uniform reference sample.
By default, this contains 100,000 generated multi-mutants sampled with replacement.
If fewer supported multi-mutants exist, the generator scores all of them exactly.
It maps those ranks to quantiles of the measured calibration distribution.
Repeated measured values retain their empirical probability mass.
Linear interpolation handles other values, with linear extrapolation through the extreme intervals.

The map preserves ordering among combinations except where a repeated measured value creates ties.
WT and single-site variants retain their measured fitness exactly.
The calibration combinations fit the model but do not retain their measured fitness exactly.
Repeated queries return the same value, irrespective of query order or batch size.

There is no default limit on the number of supported genotypes.
The reference sample needs generated scores only, with no extra experimental measurements.
Each position receives a residue uniformly from its supported alphabet, and WT and single-site genotypes are rejected.
This samples combinations uniformly even when sites have different numbers of alternatives.
The reference sample stays fixed during queries. Values outside its score range use percentile zero or one.

Use `reference_size` to trade computation for percentile precision, and `reference_seed` to choose an independent reference sample.
The default reference seed is zero, separate from the seed for pair interactions.
Record both seeds and the reference size to reproduce a sampled landscape.

```python
landscape = generator.draw(seed=42, reference_size=100_000, reference_seed=0)
# Exact calculation for numerical checks and archived experiments.
exact = generator.draw(seed=42, reference_size=None)
```

Exact calculation still needs time and memory proportional to the search-space size.
The older `maximum_variants` argument remains an optional user-specified safety limit, with no limit set by default.

```python
from mutascape import QuantileCalibratedLandscape

saved = generator.to_dict()
restored = QuantileCalibratedLandscape.from_dict(saved)
landscape = restored.draw(seed=42)
```

The [matched comparison](singles_pilot_extension_results.md) tests Combination-calibrated Mutascape on the same genotypes as the singles-only manuscript model.
It uses 192 calibration combinations per protein and excludes every evaluation genotype and spectral-test corner.
Across four confirmation proteins, mean sequence-aware C2ST AUC is 0.667, compared with 0.979 for singles-only generation.
These scores describe one calibration set per protein.

The manuscript presents combination calibration as its main construction, with a single-site-only comparison.
The [reference-sampling validation](reference_sampling.md) compares the numerical approximation with the exact maps used for the reported experiments.
The [earlier combination-calibration study](quantile_calibration_2026-09-23.md) records a separate five-protein search experiment.
A close fitness distribution does not establish correct interactions or better search-method rankings.
