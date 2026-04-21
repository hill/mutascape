# GFP dataset (Sarkisyan et al. 2016)

Sparse full-protein DMS and multi-mutant fluorescence data for avGFP. This is **not** a complete factorial landscape.

## Local files

| File | Contents |
|---|---|
| `gfp.csv` | Single-site DMS table in mutascape format. 1,084 single mutants across 233 positions. |
| `gfp_wt.fasta` | Wild-type avGFP sequence used for mutation parsing. |
| `gfp_2wur.pdb` | PDB 2WUR structure used as the GFP structural prior. |
| `gfp_distances.npz` | C-beta distance matrix for 224 resolved DMS positions in 2WUR. |
| `prepare_gfp.py` | Regenerates the local single-site and structure files. |

## Multi-mutant data used for supplemental validation

The supplemental GFP rerun reads the sparse multi-mutant ProteinGym file from the adjacent `mutantopt` checkout:

`../mutantopt/experiments/datasets/proteingym/GFP_AEQVI_Sarkisyan_2016.csv`

Current local copy:

- 51,714 total variants
- 1,084 single mutants
- depths 1 through 15
- 233 observed mutated positions

This is many orders of magnitude smaller than the complete product space over the observed single-mutant alphabets, so exact Walsh-Hadamard spectra and complete-space local-optima counts are not available for GFP. Use GFP only as a sparse full-protein generalization check.
