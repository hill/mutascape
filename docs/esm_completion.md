# Whole-protein single completion

The optional ESM-2 mode fills missing single substitutions across the wild-type sequence.
Measured observations remain exact. The default mode still requires measured singles.

```python
from mutascape import from_dms, load_oracle

generator = from_dms(
    "data/creilov/creilov.csv",
    fasta="data/creilov/creilov_wt.fasta",
    complete="esm2",
)
oracle = generator.draw(seed=42)
oracle["A29H"]                 # 0.2443804979999999, measured
oracle["A2N"]                  # about 0.095650, imputed
oracle["M1A+A29H+R60D"]         # about 0.483683, synthetic
oracle.provenance("A2N")       # "imputed"
oracle.save_manifest("oracle.json")

restored = load_oracle(
    "oracle.json",
    dms="data/creilov/creilov.csv",
    fasta="data/creilov/creilov_wt.fasta",
)
```

ESM-2 scores each candidate residue with its wild-type position masked.
The score is the log probability ratio of the candidate residue to the wild-type residue.
The default model is `esm2_t33_650M_UR50D`.
See the [ESM variant scoring instructions](https://github.com/facebookresearch/esm/tree/main/examples/variant-prediction).

Ridge regression maps this score to the assay scale, with an intercept and a fixed penalty of one.
The score scaler and regression use measured singles only.
Predictions stay within the smallest and largest measured single effects.
The readout and pair-interaction scale also use measured singles only.
Adding completion therefore preserves existing supported queries at a fixed seed.

The CreiLOV input contains 2,121 measured substitutions across 118 positions.
Completion adds 140 substitutions, including the missing position, for 2,261 substitutions across 119 positions.
Imputed singles stay fixed across oracle draws. Their spread across draws does not estimate imputation uncertainty.
The saved manifest contains their values and provenance, so restoration needs no model weights.

## Validation

Five folds hide either individual substitutions or whole positions.
Each fold fits calibration and clipping bounds on its training singles.
The score model sees the wild-type sequence but no assay labels.

| Held-out unit | Method | MAE | RMSE | Spearman correlation |
|---|---|---:|---:|---:|
| Position | ESM-2 calibration | 0.1553 | 0.2564 | 0.6340 |
| Position | Training mean | 0.1837 | 0.3042 | -0.0345 |
| Substitution | ESM-2 calibration | 0.1545 | 0.2550 | 0.6364 |
| Substitution | Training position mean | 0.1305 | 0.2465 | 0.6070 |

ESM calibration helps at positions without measurements.
Position means give lower errors when other substitutions at that position remain measured.
These results support an optional completion mode, rather than a claim that ESM gives the best estimate for every missing single.
Whole-protein query coverage does not validate high mutation depths or correct the combinatorial fitness distribution.

Reproduce the tables with the following command.

```sh
uv run python scripts/validate_single_imputation.py --output results/creilov_imputation
```

The output includes all held-out predictions, masked scores, source hashes, and fold settings.
The studio upload form offers the same ESM option and labels imputed queries.
