# Combination-calibrated Mutascape

Combination-calibrated Mutascape uses measured combinations to fit depth effects, interaction strength and a fitness-distribution map.
Its Python class is `QuantileCalibratedLandscape`.
It uses measured combinations from a uniform sample of the intended library.
The wild type and singles must come from the same assay.

```python
from mutascape import from_dms

model = from_dms("singles.csv", fasta="wt.fasta", pilot="pilot.csv")
landscape = model.draw(seed=42)
fitness = landscape["A102G+P20Y"]
```

The singles file follows the `from_dms` formats.
The pilot file has `variant` and `fitness` columns.
`QuantileCalibratedLandscape.fit(sequence, singles, pilot)` accepts DataFrames with `mutations` dictionaries instead.
The pilot contains distinct combinations with at least two substitutions.
Each substitution needs a measured single.

The model fits the mutation-depth terms and sigmoid used by `PilotCalibratedLandscape`.
For each landscape sample, it enumerates the supported combination library and records the generated fitness ranks.
It maps those ranks to quantiles of the measured pilot distribution.
Repeated measured values retain their empirical probability mass.
Linear interpolation handles other values, with linear extrapolation through the extreme intervals.

The map preserves ordering among combinations except where a repeated measured value creates ties.
Wild type and singles retain their measured fitness exactly.
The calibration combinations fit the model but do not retain their measured fitness exactly.
Repeated queries return the same value, irrespective of query order or batch size.

The default enumeration limit is one million variants.
Larger libraries raise an error before enumeration.
A caller can set `maximum_variants` on `draw` after assessing memory and runtime.
The enumeration uses generated scores and genotype identities. It does not need unknown measured fitness.

```python
saved = model.to_dict()
restored = QuantileCalibratedLandscape.from_dict(saved)
landscape = restored.draw(seed=42)
```

The [matched comparison](singles_pilot_extension_results.md) tests Combination-calibrated Mutascape on the same genotypes as the singles-only manuscript model.
It uses 192 calibration combinations per protein and excludes every evaluation genotype and spectral-test corner.
Across four confirmation proteins, mean sequence-aware C2ST AUC is 0.667, compared with 0.979 for singles-only generation.
These scores describe one calibration set per protein.

The manuscript presents `DMSLandscape` as the main generator and Combination-calibrated Mutascape as an optional model.
The [earlier combination-calibration study](quantile_calibration_2026-09-23.md) records a separate five-protein search experiment.
A close fitness distribution does not establish correct interactions or better search-method rankings.
