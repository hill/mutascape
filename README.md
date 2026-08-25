<p align="center">
  <img src="docs/logo_ai_image.png" alt="mutascape" width="500">
</p>

# mutascape

Task-relative synthetic combinatorial protein fitness oracles for ML-assisted directed evolution benchmarks.

Mutascape turns wild type and measured single mutants into a stable, lazy fitness oracle. The default generator adds baseline-corrected random pair interactions and a saturating readout. It preserves every supplied observation exactly.

Mutascape targets benchmark behaviour. It does not predict an unknown protein's combinatorial landscape.

## Installation

Requires Python 3.12+.

```bash
uv sync              # core dependencies
uv sync --extra dev  # + pytest/ruff
```

## Quick start

```python
from mutascape import from_dms

generator = from_dms(
    "dms.csv",
    fasta="wild_type.fasta",
    maximum_depth=5,
)

# One seed fixes one benchmark instance.
oracle = generator.draw(seed=42)
fitness = oracle["V39A+D40W+V54F"]
```

Mutation labels are case-insensitive. Use `+` or a colon between substitutions. Dictionaries and `Mutant` objects remain supported.

```python
oracle.score("v39a:d40w")
oracle.score({39: "A", 40: "W"})
oracle.score_many(["WT", "V39A", "V39A+D40W"])
```

The oracle accepts a genotype only when every constituent substitution has a measured single effect. A supplied multi-mutant returns its measured value. `generator.domain` reports the supported positions, substitutions, and depth limit.

Save the fixed generator version, seed, source digest, and domain with the benchmark result.

```python
oracle.save_manifest("oracle.json")
```

Recreate the same oracle from the manifest and source data.

```python
from mutascape import load_oracle

oracle = load_oracle(
    "oracle.json",
    dms="dms.csv",
    fasta="wild_type.fasta",
)
```

## Benchmarking a search method

A campaign gives a method a budget, a candidate set, and an assay. It records every measurement, counts rounds, and scores the run afterwards. The oracle stays private, so a method can only learn a fitness by spending budget on it.

```python
from mutascape import run_campaign

wanted = min(5000, generator.domain.count(depth=3))
candidates = generator.domain.sample(depth=3, n=wanted, seed=0)

def my_method(campaign):
    campaign.measure(campaign.sample(96))          # first plate
    while campaign.remaining:
        design = campaign.encode()                  # one-hot over candidates
        observed = campaign.observed                # variant, fitness, round
        picks = choose_next(design, observed)       # your model goes here
        campaign.measure(picks[: campaign.remaining])

result = run_campaign(
    my_method,
    generator.draw(seed=42),
    candidates,
    budget=384,
    batch_size=96,
    noise=0.1,
)
print(result.simple_regret, result.rounds)
```

`batch_size` defaults to 96, one microplate. `noise` adds assay error in units of the landscape's own spread, pinned to the variant so a repeated read never changes. A repeated measurement is free, because a campaign that already holds it does not buy it twice.

Scoring reads the true fitness of the variant with the best read, not the read itself, so a method that chases assay noise pays for it. `simple_regret` is the gap to the best candidate over the range of the candidate set, which is the number the frozen study reports.

## What real enough means

Mutascape uses task-relative validity. A useful search benchmark should preserve conclusions about which optimisation method performs better.

The frozen study uses 16 complete SSMuLA landscapes, seven external GraphFLA protein landscapes, and 12 search methods. It tests budgets of 48, 96, 192, and 384 queries with ten trials and three matched generator draws.

- At 48 queries, development Kendall transfer is 0.529 and reliable winner accuracy is 88.5 per cent.
- Mutascape passes the measured-reliability floor on 14 of 16 development landscapes.
- External winner accuracy is 86.4 per cent, with a reliability pass on all seven landscapes.

The browser retains 96 queries as its reference. At that budget, development Kendall transfer is 0.454 and winner accuracy is 78.7 per cent. External Kendall transfer is 0.644 and winner accuracy is 74.2 per cent.

No generator leads every metric at later budgets. High-budget external rank transfer also saturates for several generators. Treat validity as relative to the method set, budget, and decision measure.

The default fails a stricter topology test. It has fewer local optima and less Walsh non-additive variance than measured landscapes. Use it for method-ranking studies within the tested protocol. Do not use it for claims about peak counts or molecular epistasis.

## Reproduce the task-transfer result

```bash
just benchmark-data
for budget in 48 96 192 384; do
  uv run python scripts/run_search_transfer.py \
    --ssmula-root .mutascape/benchmark-data/ssmula \
    --combingym-root .mutascape/benchmark-data/combingym \
    --output "notebooks/iclr/output_validity_b${budget}_primary" \
    --budget "$budget" --seeds 10 --draws 3 \
    --generators primary --dms-pairwise-coupling 0.1
done
```

The runner writes trial-level regret, rank transfer, paired winners, and measured split-half reliability. Source manifests pin upstream revisions.

## Browser studio

The studio exposes the same DMS-pairwise oracle. It shows the frozen task-transfer result, fitness by mutation depth, a genotype graph, and exact queries.

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

The API keeps its existing posterior route names for compatibility. Studio labels describe oracle draws and variation rather than biological uncertainty.

## Legacy structure-aware generator

The earlier `Landscape` API remains available for experiments with structural contacts, MSA couplings, and genotype-stable noise. These arms are outside the current default claim.

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
./.venv/bin/python scripts/rebuild_reference_outputs.py
./.venv/bin/python scripts/run_gfp_supplemental_validation.py
./.venv/bin/pytest -q tests/test_reference_configs.py
```

The rebuild script regenerates the GB1 / TrpB sweep CSVs and the headline figures under `notebooks/validation/output_*`.

## Known limitations

- Validation depends on the method set, query budget, and candidate domain.
- The coupling was selected on the SSMuLA development panel.
- The external protein panel contains seven landscapes.
- The default underestimates measured local optima and non-additive Walsh variance.
- Oracle ensembles measure benchmark sensitivity. They are not calibrated biological intervals.
- Whole-protein queries require measured singles for every constituent substitution.

## Docs

- [Project status](docs/project_status.md)
- [Benchmark protocol](docs/benchmark_protocol.md)
- [Global-epistasis design](docs/nonlinearity_design.md)
- [Historical design document](docs/design.md)
