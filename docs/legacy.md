# Legacy generators and archived studies

These sections cover the empirical-range generator behind the browser studio, the structure-aware `Landscape` API and the studies built on them. They do not describe the current `from_dms` generator.

## Archived empirical-range reference study

Mutascape uses task-relative validity. A useful search benchmark should preserve conclusions about which optimisation method performs better.

The development study covers 31 assay entries across 17 protein groups, with 12 search methods.
GB1 and eqFP611 recur across collections, so these entries are not independent experiments.
It tests 48, 96, 192, and 384 queries with ten trials and three generator samples.

At 48 queries, protein-balanced agreement between eligible method pairs is 89.5 per cent on SSMuLA, 69.1 on CombinGym, and 73.5 on GraphFLA.
Their median rank correlations are 0.556, 0.364, and 0.439.
A pair qualifies when its paired bootstrap interval on measured regret excludes zero.
We calculate the matching fraction within each protein, then average those fractions.

Fixed rankings learned from other proteins reach 91.9, 84.0, and 73.8 per cent at 48 queries.
The singles-only reference exceeds this control in four of twelve collection-budget comparisons.
High agreement alone does not establish added value from a target-specific landscape.

The browser retains 96 queries as its reference.
At that budget, pair agreement is 81.9, 67.8, and 71.9 per cent, respectively.
No generator leads every collection and budget.
Every test excludes all assays of the target protein from prior fitting, including the three CombinGym eqFP611 readouts for GraphFLA.

CreiLOV has a close projected Walsh spectrum but a large raw fitness mismatch.
The projection averages amino-acid variants and fills 118 missing site patterns with the observed mean.
Complete-library spectra for DHFR and eqFP611 red show excess additive variance.
Use the reported search comparisons within their tested protocol.
They do not establish matching genotype values, peaks, or molecular interactions.

A prospective ParB-Noc test exceeds the fixed-ranking control at all four budgets.
Its median classifier AUCs remain 0.900 and 0.963, and its median spectral distance is 1.033.
Search agreement does not imply matching landscape shape.

On CreiLOV, 48 calibration combinations reduce raw Wasserstein distance from 0.439 to 0.062 against a matched singles-only control.

## Reproduce the archived task-transfer result

```bash
just benchmark-data
for budget in 48 96 192 384; do
  uv run python scripts/run_search_transfer.py \
    --ssmula-root .mutascape/benchmark-data/ssmula \
    --combingym-root .mutascape/benchmark-data/combingym \
    --output "notebooks/iclr/output_validity_b${budget}_primary" \
    --budget "$budget" --seeds 10 --samples 3 \
    --generators primary --dms-pairwise-coupling 0.1
done
```

The runner writes trial-level regret, rank transfer, paired winners, and measured split-half reliability. Source manifests pin upstream revisions.

## Browser studio

The studio exposes the existing empirical-range DMS-pairwise oracle. It shows the frozen task-transfer result, fitness by mutation depth, a genotype graph, and exact queries.

Upload DMS data with either supported CSV schema. Both schemas require a matching FASTA sequence.

```csv
position,wildtype,mutant,fitness
39,V,A,-0.42
```

Variant-level files must include `WT`. They may also include measured combinations.

```csv
variant,fitness
WT,1.0
V39A,0.58
V39A:D40W,0.91
```

Run the studio and API locally with `uv sync --extra dev --extra web` and `just dev`. The local address is `http://localhost:8000`.

The API keeps its existing posterior route names for compatibility. Studio labels describe oracle samples and variation rather than biological uncertainty.

## Legacy structure-aware generator

The earlier `Landscape` API remains available for experiments with structural contacts, MSA couplings, and genotype-stable noise. These experimental models are outside the current default claim.

## Loading landscapes

### From a PDB or mmCIF structure

```python
contacts = ContactMap.from_pdb("structure.pdb", threshold=8.0, chain_id="A")
```

### From an ESMFold prediction (no PDB available)

```python
contacts = ContactMap.from_sequence("MAEQLA...", threshold=8.0)  # cached locally
```

### From a precomputed distance matrix

```python
import numpy as np

distances = np.load("distances.npy")  # shape (L, L)
contacts = ContactMap.from_distance_matrix(distances, positions=range(1, 120), threshold=8.0)
```

### Hybrid DMS with ESM2 fill-in

```python
from mutascape import ESM2Scorer

scorer = ESM2Scorer()
landscape = Landscape.from_dms(
    dms_path="partial_dms.csv",
    wt_sequence="MAEQLA...",
    contact_map=contacts,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    scorer=scorer,  # fills unscreened positions with masked-marginal ESM2 scores
)
```

### Pure pLM landscape (no DMS)

```python
landscape = Landscape.from_plm(
    sequence="MAEQLA...",
    contact_map=contacts,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    scorer=scorer,
)
```

## Querying the oracle

```python
# Index by label or by Mutant
landscape["A29H"]
landscape[Mutant("A29H")]

# Compose mutants with + (and remove with -)
m = Mutant("A29H") + Mutant("R60D")
landscape[m]
m.depth         # 2
dict(m)         # {29: 'H', 60: 'D'}

# Stream variants at a fixed mutation depth
for v in landscape.generate(n_mutations=3, n=1000, seed=42):
    print(v.mutations, v.fitness)

# Bulk sample to a pandas DataFrame across a depth range
df = landscape.to_dataframe(n_variants=10_000, n_mutations=(1, 5), seed=42)

# Compare actual against the additive prediction
actual   = landscape.evaluate({29: "H", 60: "D"})
additive = landscape.additive_fitness({29: "H", 60: "D"})

# Cheap descriptive stats
landscape.stats(n_samples=5000, n_mutations=3)
# -> {'mean_fitness': ..., 'additive_correlation': ...,
#     'mean_epistatic_deviation': ..., 'fraction_sign_epistasis': ...}
```

## Tuning ruggedness

`alpha` is a global multiplier on the pairwise term. `alpha=0` recovers the purely additive landscape. Raising it progressively introduces sign epistasis and local optima.

```python
smooth   = Landscape.from_dms(..., epistasis=RandomEpistasis(sigma=0.3, seed=42), alpha=0.0)
moderate = Landscape.from_dms(..., epistasis=RandomEpistasis(sigma=0.3, seed=42), alpha=1.0)
rugged   = Landscape.from_dms(..., epistasis=RandomEpistasis(sigma=0.3, seed=42), alpha=2.5)
```

Swap the epistasis model itself to change the *character* of the interactions, not just their magnitude:

```python
from mutascape import SignConsistentEpistasis, MSACouplingsEpistasis

# Sign-consistent: beneficial+beneficial and deleterious+deleterious cooperate
sc = Landscape.from_dms(..., epistasis=SignConsistentEpistasis(sigma=0.3, seed=42))

# Evolutionarily informed: precomputed couplings from plmDCA / EVcouplings
msa = Landscape.from_dms(
    ...,
    epistasis=MSACouplingsEpistasis(
        coupling_matrix=J,                    # shape (L, L, 20, 20)
        position_order=tuple(positions),
        scale=0.5,
    ),
)
```

Add a global-epistasis function, genotype-stable measurement noise, and an optional censored readout texture on top of the latent score:

```python
from mutascape import GlobalEpistasisSigmoid

nonlin = GlobalEpistasisSigmoid.fit_from_singles(
    landscape.single_site_effects,
    combinatorial_upside=2.0,
)

landscape = Landscape.from_dms(
    ...,
    epistasis=RandomEpistasis(sigma=0.3, seed=42),
    nonlinearity=nonlin,
    noise_sigma=0.15,
    readout_texture_sigma=0.005,   # tiny post-readout texture on the floor
    readout_texture_order=3,
    seed=42,
)
```

## Archival reference landscapes

Mutascape still ships the earlier GB1 and TrpB reference configurations. They support regression tests and comparison with the first manuscript. They are outside the new task-transfer default.

```python
from mutascape.validation.reference_configs import GB1_HEADLINE, TRPB_HEADLINE

gb1  = GB1_HEADLINE.build_landscape()
trpb = TRPB_HEADLINE.build_landscape()

gb1["V39A"]
trpb[Mutant("V183A") + Mutant("F184I")]
```

These frozen outputs preserve the earlier Walsh and local-optima results. They should not be mixed with the DMS-pairwise task-transfer claim.

## Reproduce archival outputs

```bash
./.venv/bin/python scripts/run_noise_sweep.py --protein gb1
./.venv/bin/python scripts/run_noise_sweep.py --protein trpb
./.venv/bin/python scripts/build_noise_figures.py
./.venv/bin/python scripts/build_contact_map_figure.py
./.venv/bin/python scripts/run_gfp_supplemental_validation.py
./.venv/bin/pytest -q tests/test_reference_configs.py
```

These commands regenerate the archived GB1 and TrpB CSVs and figures under `notebooks/validation/output_*`.

- [Global-epistasis design](nonlinearity_design.md)
