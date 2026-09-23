# Matched test of the optional combination extension

The manuscript model uses only WT, a sequence and same-assay single-mutant fitness.
This experiment tests what 192 additional combination measurements contribute.
It does not change either model's parameters or implementation.

## Protocol

Each protein uses the same 1,000 evaluation genotypes and 256 complete three-site subcubes as the singles-only study.
Pilot selection excludes WT, singles, evaluation genotypes and every spectral subcube corner.
Sampling is uniform without replacement among the remaining measured combinations.
The selection seed is 2026092327 plus the protein index in the frozen protocol.
One pilot per protein supplies all three landscape seeds.

Five grouped C2ST folds keep each genotype's measured and generated examples together.
The classifier uses the same settings and feature construction for both models.
The sequence-aware version adds residue indicators to assay and mutation-depth features.
We report median AUC across landscape samples and weight proteins equally in the confirmation mean.
AUC near 0.5 indicates weak discrimination for this classifier.
This test uses no permutation p-values.

## Results

All four confirmation proteins improve on all three reported diagnostics.
Mean sequence-aware AUC changes from 0.979 to 0.667.
Mean scaled Wasserstein distance changes from 1.116 to 0.100.
Mean local Walsh distance changes from 0.500 to 0.234.

| Protein | C2ST, singles | C2ST, pilot | W1/SD, singles | W1/SD, pilot | Walsh, singles | Walsh, pilot |
|---|---:|---:|---:|---:|---:|---:|
| DHFR | 0.988 | 0.733 | 1.393 | 0.170 | 0.172 | 0.221 |
| eqFP611 | 0.988 | 0.798 | 0.640 | 0.106 | 0.195 | 0.056 |
| ParB-Noc | 0.953 | 0.651 | 1.133 | 0.106 | 0.575 | 0.374 |
| ParD2 | 0.796 | 0.730 | 0.324 | 0.068 | 0.069 | 0.008 |
| CreiLOV | 1.000 | 0.753 | 1.508 | 0.166 | 0.028 | 0.089 |
| GB1 | 0.994 | 0.722 | 0.567 | 0.081 | 0.358 | 0.092 |
| TrpB4 | 0.989 | 0.706 | 0.746 | 0.105 | 0.218 | 0.106 |
| T7 | 0.992 | 0.559 | 2.239 | 0.143 | 0.559 | 0.211 |
| TEV | 0.943 | 0.681 | 0.910 | 0.071 | 0.865 | 0.526 |

DHFR, eqFP611, ParB-Noc, ParD2 and CreiLOV are development proteins.
GB1, TrpB4, T7 and TEV are confirmation proteins.
The latter did not select the singles-only settings, although they appeared in earlier project work.

CreiLOV illustrates why the diagnostics differ.
Its AUC changes from 1.000 to 0.753, but local Walsh distance increases from 0.028 to 0.089.
A closer fitness distribution does not guarantee a closer spectrum.

The classifiers still detect differences after fitting the extension.
Variation between landscape seeds does not measure uncertainty from collecting a different pilot.
The experiment uses one pilot per protein and does not test search rankings for the extension.
Earlier pilot-search results use different test designs and should not be substituted here.
The cost includes 192 measured combinations per target, beyond WT and singles.

## Reproduction

Run from the Mutascape repository after preparing the singles-only study.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m scripts.assess_singles_pilot_extension prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m scripts.assess_singles_pilot_extension run --jobs 4
uv run python -m scripts.assess_singles_pilot_extension freeze
uv run python -m scripts.rebuild_preprint
```

Raw pilot identities, model states and evaluation arrays remain under `results/singles_pilot_extension_20260923`.
The compact publication record is `src/mutascape/data/validation/singles-pilot-extension-v1.json`.
It includes all pilot genotypes, per-seed metrics and the protocol recorded before scoring.
The manuscript builder checks search point estimates against the frozen singles-only results.
It treats Kendall input differences below 12 decimal places as numerical ties during bootstrap resampling.
That convention leaves all frozen point estimates unchanged.
