# Reference sampling for combination calibration

The generator estimates score percentiles from 100,000 uniformly sampled generated multi-mutants, with replacement.
It uses every multi-mutant exactly when fewer are supported.
This removes the previous default limit of one million supported genotypes without requiring more experimental measurements.
The reference sample remains fixed during queries. Reference seed zero is independent of the pair-interaction seed.

```python
from mutascape import from_dms

generator = from_dms(singles, wt_sequence=sequence, calibration=combinations)
landscape = generator.draw(seed=42)
fitness = landscape["A102G+P20Y"]
```

Change `reference_size` or `reference_seed` on `draw` to control numerical precision or assess sensitivity.
Set `reference_size=None` for exact enumeration.
The legacy `maximum_variants` argument applies a limit only when explicitly supplied.
Record both seeds and the reference size with a sampled landscape.

## Validation

The checks restore all nine fitted manuscript generators with their original 96 experimental calibration genotypes.
They retain the fitted parameters, interaction seeds, evaluation genotypes and Walsh subcubes.
Exact mode reproduces all 27 archived fitness arrays with zero maximum error.
Reference sampling uses no additional experimental fitness values.

The sensitivity study compares 10,000 and 100,000 reference samples using three interaction seeds and three reference seeds.
At 100,000, the largest midpoint-percentile error is 0.00533.
The largest mean absolute fitness error, divided by experimental standard deviation, falls from 0.03742 at 10,000 to 0.00794 at 100,000.
This supports the larger default as a numerical-precision choice rather than a maximum search-space size.

Maximum absolute metric changes appear below.
Wasserstein and Walsh changes cover all three reference seeds.
C2ST uses reference seed zero, all eligible measured genotypes and the manuscript's saved folds.
Exact-mode C2ST values come from verified archived arrays. A fresh GB1 classifier replay also matches the archived AUC.

| Protein | Default mode | Wasserstein change | Walsh change | Full C2ST AUC change |
|---|---|---:|---:|---:|
| CreiLOV | sampled | 0.00325 | 0.00126 | 0.00099 |
| DHFR | exact | 0.00000 | 0.00000 | 0.00000 |
| GB1 | sampled | 0.00568 | 0.00294 | 0.00119 |
| ParB-Noc | sampled | 0.00498 | 0.00132 | 0.00124 |
| ParD2 | exact | 0.00000 | 0.00000 | 0.00000 |
| T7 | exact | 0.00000 | 0.00000 | 0.00000 |
| TEV | sampled | 0.00267 | 0.00194 | 0.00167 |
| TrpB4 | sampled | 0.00440 | 0.00715 | 0.00104 |
| eqFP611 | exact | 0.00000 | 0.00000 | 0.00000 |

The four evaluation proteins give the following means after taking medians across three interaction seeds, using reference seed zero.

| Metric | Exact | Sampled |
|---|---:|---:|
| Scaled Wasserstein distance | 0.121138 | 0.121052 |
| Local Walsh distance | 0.281980 | 0.281672 |
| Sequence-aware C2ST AUC | 0.731340 | 0.731936 |

These summary changes are small, but errors vary by genotype.
The largest individual fitness difference is 0.297 experimental standard deviations, and the largest 99th-percentile difference is 0.143.
Sparse calibration tails can turn a small percentile error into a larger fitness difference.
Increase the reference size when those values matter, or use exact mode when computation permits.
This numerical check does not rerun search campaigns or test new experimental calibration samples.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m scripts.validate_reference_sampling --c2st-scope sample1000 --jobs 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m scripts.validate_reference_sampling --c2st-scope full --sizes 100000 --reference-seeds 0 --output results/reference_sampling_full_c2st_20261014
```

The first command retains a sensitivity check with the historical 1,000-genotype C2ST subset.
Those AUCs use different training sizes and folds and must not replace the full-dataset manuscript values.
The second command supplies the full-dataset C2ST values reported here.
Result directories contain source hashes, per-seed comparisons and summaries.

Tests cover unequal site alphabets, fixed queries, preservation of observations, restored models and spaces above one million genotypes.
A 70-site binary example verifies sampling without enumerating more than 10^21 genotypes.
All 676 Python tests, lint and the web checks pass, including 34 web tests.

The manuscript retains its historical exact-mode comparisons and adds this check in Supplementary Results S11.
Historical experiment scripts explicitly request exact mode to reproduce their archived outputs.
