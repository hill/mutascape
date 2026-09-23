# Experimental pilot generator

`PilotCalibratedLandscape` needs wild type, measured singles, and measured combinations from the same assay.
It fits mutation-depth effects that singles alone cannot identify.
It remains separate from the default generator.

```python
from mutascape.experimental import PilotCalibratedLandscape

model = PilotCalibratedLandscape.fit(sequence, singles, pilot)
oracle = model.draw(seed=42)
values = oracle.evaluate_batch(variants)
saved = model.to_dict()
restored = PilotCalibratedLandscape.from_dict(saved)
```

Both input tables need `mutations` and `fitness` columns.
Each mutation map uses one-based residue positions and amino-acid letters.
The singles table must include wild type as an empty map.
Every pilot mutation must have a measured single effect.
The pilot must contain distinct multi-mutants and identify both fitted burden terms.
The fitted model rejects unsupported substitutions.

The model uses a sigmoid with bounds five per cent beyond the calibration range.
Two regularised terms describe added mutation burden and its dependence on additive latent fitness.
Pilot residuals set the scale of shared random pair effects.
Wild type and singles retain their measured values exactly.
Pilot combinations fit aggregate terms, so their generated fitness can differ from their measurements.
A saved model and seed reproduce the same query values in any order.

## Evidence and measurement cost

The implementation reproduces held-out values from nine archived CreiLOV prototype draws.
Three splits each use 48 calibration combinations and 2,000 held-out classifier genotypes.
The matched control uses the same assay's singles with the original transferred prior.

| Median measure | Singles-only control | Pilot model |
|---|---:|---:|
| Raw Wasserstein distance | 0.439 | 0.062 |
| Logistic classifier AUC | 0.999 | 0.595 |
| Boosted classifier AUC | 0.999 | 0.619 |

This evidence covers one protein and does not validate biological predictions.
The pilot model still differs from measured fitness.
Its search check preserves 348 of 399 scored comparisons across repeated draws.
These comparisons are not independent observations.
Calibration variants do not enter the search pool.
A 48-query search requires 48 calibration measurements plus 48 search measurements, beyond the singles.

Reproduce the matched comparison with `uv run python -m scripts.validate_pilot_generator`.
The script reads the local CreiLOV workbook and checks every saved prototype prediction.
