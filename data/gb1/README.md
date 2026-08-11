# GB1 dataset (Wu et al. 2016)

Small-site **complete combinatorial** DMS of the Streptococcus protein G domain B1 (GB1), used as a primary complete-combinatorial validation dataset in the mutascape study. GB1 covers ~93% of the 20^4 possible variants across 4 sites, which lets us compute exact Walsh-Hadamard variance decompositions, local-optima counts, and accessible-path fractions on both real and synthetic landscapes.

## Source

- **Publication**: Wu, N. C., Dai, L., Olson, C. A., Lloyd-Smith, J. O., & Sun, R. (2016). *Adaptation in protein fitness landscapes is facilitated by indirect paths.* eLife 5:e16965. [https://doi.org/10.7554/eLife.16965](https://doi.org/10.7554/eLife.16965)
- **ProteinGym identifier**: `SPG1_STRSG_Wu_2016` (ProteinGym substitutions release, [proteingym.org](https://proteingym.org), CC-BY-4.0). The single-site CSV is the source for `gb1.csv`.
- **Structure (PDB)**: [2QMT](https://www.rcsb.org/structure/2QMT) - 1.05 A X-ray structure of the GB1 monomer. Chosen over 2GI9 (GB1/Fc complex) for a cleaner single-chain contact map. Downloaded from `https://files.rcsb.org/download/2QMT.pdb` by the preparation script.

## Experiment design

Wu et al. screened every combination of 20 amino acids at four positions in the IgG-binding site of GB1 for binding fitness via an mRNA display assay:

| Position (GB1 domain) | Wild-type AA |
|---|---|
| 39 | V |
| 40 | D |
| 41 | G |
| 54 | V |

Of the 20^4 = 160,000 possible variants, 149,360 (93.4%) were measured above detection. `DMS_score` is an enrichment ratio (selected / input); WT enrichment is 1.0. After subtracting `WT_FITNESS = 1.0` we get fitness values relative to WT: `0.0` is WT, `-1.0` is completely dead, positive values are beneficial.

Depth distribution after WT-centering:

| Depth | Count |
|---|---|
| 1 | 76 |
| 2 | 2,091 |
| 3 | 26,019 |
| 4 | 121,174 |

## Numbering convention

ProteinGym numbers residues using the full 448-residue Streptococcus G precursor, which places the GB1 domain at residues 227-282. The four screened sites appear in ProteinGym as positions 265/266/267/280. The preparation script translates everything to the canonical **1-56 GB1 domain numbering** used in the Wu et al. paper and in PDB 2QMT (offset = -226). All files in this directory use domain numbering.

## Files

| File | Contents |
|---|---|
| `gb1.csv` | Single-site DMS in mutascape format (`position, wildtype, mutant, fitness`) for the 4 screened positions only. 76 rows (19 non-WT amino acids x 4 positions). |
| `gb1_combinatorial.csv` | All 149,360 variants from the Wu et al. screen, with mutation strings in domain numbering (e.g. `V39G:D40A`) and fitness relative to WT. Used as the real-data anchor for validation metrics that require the full combinatorial space. |
| `gb1_wt.fasta` | 56-residue GB1 domain wild-type sequence (residues 227-282 of the Streptococcus G precursor, extracted from the ProteinGym CSV). |
| `gb1_2qmt.pdb` | PDB 2QMT structure file (downloaded during preparation). |
| `gb1_distances.npz` | Pairwise C-beta distance matrix for the 56 domain residues, with `distances` and `positions` arrays. Loaded via `ContactMap.from_distance_matrix` with an 8.0 A threshold in the validation pipeline. |
| `prepare_gb1.py` | Preparation script. Reruns idempotently if the source CSV is available and regenerates every file in this directory. |

## How to regenerate

```bash
uv run python data/gb1/prepare_gb1.py
```

Set `PROTEINGYM_DMS_DIR` (or edit the path near the top of the script) to point at your local copy of the ProteinGym substitutions release, then run the command above. It reads `SPG1_STRSG_Wu_2016.csv`, downloads 2QMT if not already present, and rewrites all output files. No other inputs required.

## Usage in validation

The frozen publication configuration is `gb1_headline` in `src/mutascape/validation/reference_configs.py`. It loads `gb1.csv` as the single-site DMS input, loads `gb1_combinatorial.csv` as the real multi-mutant reference via `load_gb1_combinatorial`, and uses `gb1_distances.npz` + `gb1_wt.fasta` for structural context. The older exploratory notebook at `notebooks/validation/gb1_validation.ipynb` remains useful as a workbench, but the scripted reference config is the canonical pipeline.
