# TrpB dataset (Johnston et al. 2024)

Small-site **complete combinatorial** DMS of the tryptophan synthase β subunit (TrpB) from *Thermotoga maritima*, used as the second cross-protein validation for mutascape. Like the GB1 dataset, TrpB covers nearly all 20^4 variants across 4 positions, letting us compute exact Walsh-Hadamard variance decompositions, local-optima counts, and accessible-path fractions on both real and synthetic landscapes.

## Source

- **Publication**: Johnston, K. E., Almhjell, P. J., Watkins-Dulaney, E. J., Nguyen, G., Nicolaou, N. J., Kim, H. J., Wang, A. A., Hameedi, M. A., Arnold, F. H. (2024). *A combinatorially complete epistatic fitness landscape in an enzyme active site.* PNAS 121(32), e2400439121. [https://doi.org/10.1073/pnas.2400439121](https://doi.org/10.1073/pnas.2400439121)
- **Data provenance**: This landscape is **not** in the main ProteinGym substitutions benchmark. It is distributed through the **SSMuLA** benchmark release (Li, X. et al., *Cell Systems*, 2025, [https://doi.org/10.1016/j.cels.2025.101236](https://doi.org/10.1016/j.cels.2025.101236)) on Zenodo.
- **Download**: SSMuLA data bundle at Zenodo record [10.5281/zenodo.13910506](https://zenodo.org/records/13910506). Direct URL for the 278 MB `data.zip`:
  ```
  https://zenodo.org/api/records/13910506/files/data.zip/content
  ```
  From the archive we use `data/TrpB/scale2max/TrpB4.csv` (stored locally as `trpb4_scaled.csv`) and `data/TrpB/fitness_landscape/TrpB4.csv` (stored as `trpb4_raw.csv`, kept for reference only).
- **Alternative upstream (raw data)**: Johnston et al. deposit raw counts + processed landscape on CaltechDATA at [10.22002/h5rah-5z170](https://data.caltech.edu/records/h5rah-5z170) (3.3 GB). Not used here; the SSMuLA release is easier to work with.
- **Structure (PDB)**: [8VHH](https://www.rcsb.org/structure/8VHH) - engineered Tm9D8* holoenzyme (homodimer, 397 aa per chain + His6 tag, PLP cofactor at residue 401). Downloaded from `https://files.rcsb.org/download/8VHH.pdb` by the preparation script.

## Experiment design

Johnston et al. screened every combination of 20 amino acids at four active-site positions of the engineered Tm9D8* TrpB parent. The screen is a growth-based selection in an *E. coli* tryptophan auxotroph, enriched for the native TrpB reaction (indole + L-Ser → L-Trp). Reads from pooled long-read sequencing give a per-variant specific growth rate, normalised so the best variant (AIKG) = 1.0.

| Position | Parent AA (Tm9D8*) | Optimum AA |
|---|---|---|
| 183 | V | A |
| 184 | F | I |
| 227 | V | K |
| 228 | S | G |

The global optimum variant is therefore **AIKG**, with the parent **VFVS** sitting at scaled fitness 0.408. All 20^4 = 160,000 theoretical variants were screened; after filtering out stop codons, 159,129 are retained.

mutascape convention requires WT / parent fitness = 0, so the preparation script subtracts the parent fitness (0.408): parent becomes 0.0, best becomes +0.592, and the lowest-observed variant sits at -0.572. This makes TrpB's fitness range (-0.572 to +0.592) considerably narrower than GB1's (-1.000 to +7.762); keep that in mind when comparing `fitness_W1` values across proteins.

Depth distribution after re-centring:

| Depth | Count |
|---|---|
| 1 | 76 |
| 2 | 2,166 |
| 3 | 27,368 |
| 4 | 129,518 |

## Contact-map caveat

The four mutated positions form the **active site** of the enzyme, but in the apo chain-A structure only two of the six possible pairs are in direct 3D contact at the 8 Å Cβ threshold:

| Pair | Cβ distance | In contact |
|---|---|---|
| 183-184 | 6.04 Å | ✓ |
| 183-227 | 20.35 Å | ✗ |
| 183-228 | 17.39 Å | ✗ |
| 184-227 | 16.16 Å | ✗ |
| 184-228 | 13.57 Å | ✗ |
| 227-228 | 5.87 Å | ✓ |

The four residues functionally couple via the shared PLP cofactor and the substrate rather than via direct residue-residue contact. When running Contact-NK on this dataset, the validation notebook (`notebooks/validation/trpb_validation.ipynb`) runs two parallel sweeps:

1. **Structural contacts (as-is)**: uses the PDB-derived distance matrix. Contact-NK gets K=1 per position (only the adjacent residue is a neighbour) and under-generates higher-order ruggedness.
2. **Active-site-augmented**: overrides the distance matrix to force all four positions into a 4-clique, reflecting the functional coupling. Contact-NK gets K=3 per position and recovers behaviour consistent with GB1.

This is flagged as a deliberate modelling choice in the paper: spatial and functional contact graphs can differ, but overrides are *not* meant to be arbitrary. The project rule is:

> Only override the structural contact graph when an independent mechanistic justification exists, such as a shared ligand/cofactor or an annotated active-site mechanism.

TrpB qualifies because the four assayed positions are coupled through the same PLP/substrate chemistry. This rule is now the default policy for future benchmark landscapes as well.

## Files

| File | Contents |
|---|---|
| `trpb.csv` | Single-site DMS in mutascape format (`position, wildtype, mutant, fitness`) for the 4 screened positions. 76 rows (19 non-parent amino acids × 4 positions). Fitness is parent-relative. |
| `trpb_combinatorial.csv` | All 159,128 multi-mutant variants (parent row dropped), mutation strings in ProteinGym style (e.g. `V183A:F184I`), fitness parent-relative, depth column. Same schema as `gb1_combinatorial.csv` — the `load_gb1_combinatorial` loader works unchanged. |
| `trpb_wt.fasta` | 390-residue Tm9D8* WT sequence, extracted from PDB 8VHH chain A. |
| `trpb_8vhh.pdb` | PDB 8VHH structure file (downloaded during preparation). |
| `trpb_distances.npz` | Pairwise Cβ distance matrix for the 389 chain-A residues, with `distances` and `positions` arrays. Used by `ContactMap.from_distance_matrix` with the 8 Å threshold. |
| `trpb4_scaled.csv` | Raw SSMuLA landscape CSV (columns `AAs, AA1..AA4, # Stop, fitness, active, muts, n_mut`). Fitness in this file is on the Johnston et al. scale-to-max convention (max = 1.0), **not** re-centred to parent. Kept for traceability. |
| `trpb4_raw.csv` | Raw fitness-landscape CSV from the SSMuLA bundle before scaling. Not used by the pipeline; kept for traceability. |
| `prepare_trpb.py` | Preparation script. Reads `trpb4_scaled.csv` and `trpb_8vhh.pdb` and regenerates every other file in this directory. |

## How to regenerate

```bash
# One-off downloads (only if the CSV and PDB aren't already in this directory):
curl -L -o data/trpb/zenodo_data.zip \
  "https://zenodo.org/api/records/13910506/files/data.zip/content"
cd data/trpb
unzip -o zenodo_data.zip 'data/TrpB/scale2max/TrpB4.csv' 'data/TrpB/fitness_landscape/TrpB4.csv'
mv data/TrpB/scale2max/TrpB4.csv trpb4_scaled.csv
mv data/TrpB/fitness_landscape/TrpB4.csv trpb4_raw.csv
rm -rf data zenodo_data.zip
curl -L -o trpb_8vhh.pdb "https://files.rcsb.org/download/8VHH.pdb"
cd ../..

# Then:
uv run python data/trpb/prepare_trpb.py
```

The preparation script is idempotent: safe to rerun without re-downloading.

## Usage in validation

The TrpB validation notebook is at `notebooks/validation/trpb_validation.ipynb`. It loads `trpb.csv` as the mutascape single-site DMS input, loads `trpb_combinatorial.csv` as the real multi-mutant reference via `load_gb1_combinatorial` (the name refers to the generic schema, not to GB1 specifically), and uses `trpb_distances.npz` + `trpb_wt.fasta` to build the contact-map-derived epistasis topology. The notebook runs both contact-map variants described above and produces a cross-protein comparison against GB1.
