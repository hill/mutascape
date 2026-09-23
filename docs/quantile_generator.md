# Distribution-calibrated pilot landscapes

`QuantileCalibratedLandscape` fits the existing pilot model, then calibrates the distribution of generated combination fitness.
It uses measured combinations from a uniform sample of the intended library.
The wild type and singles must come from the same assay.

```python
from mutascape.experimental import QuantileCalibratedLandscape

model = QuantileCalibratedLandscape.fit(sequence, singles, pilot)
landscape = model.draw(seed=42)
fitness = landscape.evaluate_batch(variants)
```

The `singles` and `pilot` tables contain `mutations` dictionaries and numeric `fitness` values.
The singles table includes wild type as an empty mutation dictionary.
The pilot contains distinct combinations with at least two substitutions.
Each substitution needs a measured single.

The model fits the mutation-depth terms and sigmoid used by `PilotCalibratedLandscape`.
For each draw, it enumerates the supported combination library and records the generated fitness ranks.
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

The [confirmation report](quantile_calibration_2026-09-23.md) compares this model across five proteins.
The strongest evidence uses 192 calibration combinations.
The manuscript presents this model as Mutascape.
Its Python import remains in `mutascape.experimental` for compatibility.
The `from_dms` interface and browser studio retain the singles-only generator.
A close fitness distribution does not establish correct interaction structure or better search-method rankings.
