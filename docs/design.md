# mutascape - Design Document

> **Historical note (2026-04-21).** This design document captures the original alpha/pairwise-coupling concept. The current publication pipeline is narrower: DMS additive pre-fitness, fitted censored global-epistasis function, and Gaussian noise, with random pairwise couplings off by default (`sigma = 0`). Use `docs/project_status.md` and `src/mutascape/validation/reference_configs.py` as the canonical method spec.

## 1. Problem Statement

Machine Learning Directed Evolution (MLDE) methods are typically benchmarked on
single-site Deep Mutational Scanning (DMS) datasets, where each variant differs from
wild type by exactly one amino acid substitution. In real protein engineering campaigns,
however, beneficial variants often carry multiple simultaneous mutations. When mutations
are combined, their fitness effects can interact non-additively - a phenomenon called
**epistasis**. Epistasis makes combinatorial fitness landscapes rugged, creating local
optima and deceptive gradients that challenge ML-guided search.

The core problem: we lack benchmarks with **controlled, tuneable epistasis** based
on protein biophysics. Without such benchmarks, we cannot systematically answer
questions like "at what level of epistasis does method X outperform method Y?" or
"does encoding choice matter more in rugged vs smooth landscapes?"

mutascape originally addressed this by generating synthetic combinatorial fitness
landscapes that start from real DMS data and add structurally informed pairwise
epistatic interactions. The current validated default keeps the DMS and structure
interfaces but does not rely on random pairwise couplings for the headline results.


## 2. Prior Work and Positioning

### 2.1 SLIP - Tuned Fitness Landscapes (Thomas et al., 2022)

Thomas et al. introduced SLIP (Structured Landscapes with Informed Priors), which uses
**Potts models** fit to Multiple Sequence Alignments (MSAs) to generate synthetic
fitness landscapes with tuneable epistasis.

Their approach: given a protein family, fit an inverse Potts model to the MSA of
homologous sequences. This yields single-site fields h_i(a) and pairwise couplings
J_ij(a, b) that capture evolutionary covariation. They then introduce a "shift" and
"scale" parameterisation that controls the **epistatic horizon** K_epi - the number of
mutations from wild type at which the linear (additive) approximation breaks down.

**Key strengths**: the Potts couplings reflect real evolutionary constraints; the
epistatic horizon gives a single intuitive tunability parameter.

**Limitations for our purposes**: the single-site effects h_i come from the Potts model
(evolutionary inference) rather than actual measured DMS fitness values. The epistatic
topology is determined by evolutionary covariation, which may not correspond to direct
physical interactions. And the approach requires a high-quality MSA, which isn't always
available.

### 2.2 NK Landscapes (Sandhu et al., 2025)

Sandhu et al. used NK landscapes to study ML performance determinants across different
landscape properties. In the NK model, a protein of length N has K epistatic
interaction partners per position, drawn at random. Increasing K increases ruggedness.

**Key strengths**: NK landscapes are mathematically well-characterised and trivially
tuneable. Sandhu et al.'s experimental designs (interpolation, extrapolation, positional
extrapolation, sparsity robustness) are excellent and we can adapt them.

**Limitations**: NK landscapes have no biophysical grounding. The interacting positions
are random, not structurally determined. They used a reduced alphabet (6 amino acids)
and short sequences (length 6), which makes the landscapes tractable but distant from
real protein engineering. They note this limitation explicitly.

### 2.3 Where mutascape sits

mutascape fills the gap between these approaches:

| Property                    | SLIP          | NK (Sandhu)  | mutascape          |
|-----------------------------|---------------|--------------|---------------------|
| Single-site effects         | Potts fields  | Random       | Real DMS data       |
| Epistatic topology          | MSA covariation | Random K neighbours | Structural contacts |
| Biophysical grounding       | Evolutionary  | None         | Structural          |
| Requires MSA                | Yes           | No           | No (uses structure) |
| Tunability                  | K_epi         | K            | noise_sigma, global-epistasis function, optional couplings |
| Sequence length             | Full protein  | Short (6)    | Full protein        |
| Amino acid alphabet         | Full (20)     | Reduced (6)  | Full (20)           |

The original thesis narrative was: SLIP asks "what epistasis does evolution encode?",
Sandhu asks "what happens under abstract tuneable ruggedness?", and mutascape asks
"what epistasis does 3D structure predict?" The current paper should be phrased more
conservatively: mutascape asks whether generators that use DMS measurements and protein structure can
produce reproducible benchmark landscapes with measurable similarity to real
combinatorial landscapes.


## 3. Theoretical Background

### 3.1 Fitness Landscapes and Epistasis

A fitness landscape maps genotype to phenotype (fitness). For a protein of length L
with amino acid alphabet A (|A| = 20), the full landscape has 20^L points. In
practice, we consider a combinatorial subspace defined by a set of positions P where
mutations are allowed (typically the positions screened in a DMS experiment).

The fitness function can be decomposed as:

```
f(x) = f_wt + sum_i h_i(x_i) + sum_{i<j} J_ij(x_i, x_j) + higher-order terms
```

Where:

- f_wt is wild-type fitness (reference)
- h_i(x_i) is the **marginal effect** of amino acid x_i at position i
  (measured from DMS data as the fitness difference from wild type)
- J_ij(x_i, x_j) is the **pairwise epistatic coupling** between positions i and j
  for amino acids x_i and x_j

If J = 0 everywhere, the landscape is **additive** (no epistasis) - the fitness of
any multi-mutant is just the sum of its single-mutation effects. Real landscapes are
not additive. Types of epistasis:

- **Magnitude epistasis**: the combined effect differs from the sum, but the sign of
  each mutation's contribution doesn't change
- **Sign epistasis**: a mutation is beneficial in one genetic background but deleterious
  in another
- **Reciprocal sign epistasis**: two mutations are each deleterious alone but beneficial
  together (or vice versa) - this is what creates multiple local optima

### 3.2 Potts Models (for the uninitiated)

A Potts model (also called a Markov Random Field over discrete variables) is a
probabilistic model for sequences. Originally from statistical mechanics, it assigns a
probability to each sequence x = (x_1, x_2, ..., x_L) via:

```
P(x) = (1/Z) * exp(E(x))
```

where the "energy" (really a statistical potential, or "Hamiltonian" in physics
parlance) decomposes as:

```
E(x) = sum_i h_i(x_i) + sum_{i<j} J_ij(x_i, x_j)
```

Here h_i(x_i) are **fields** (site-wise biases - how much amino acid x_i is
preferred at position i) and J_ij(x_i, x_j) are **couplings** (pairwise
preferences - how much the amino acid pair (x_i, x_j) at positions (i, j)
is favoured or disfavoured). Z is the partition function (a normalisation constant
that sums over all possible sequences - intractable to compute exactly for real
protein lengths).

**Why Potts models matter for proteins**: when you fit a Potts model to an MSA of
homologous protein sequences (using methods like plmDCA, GREMLIN, or EVcouplings),
the learned couplings J_ij capture **coevolutionary signal**. Positions that are in
3D contact tend to coevolve because mutations at one position create selective
pressure for compensatory mutations at the contacting position. This is the basis
of contact prediction from sequence data alone, and it's how SLIP derives its
epistatic couplings.

**The connection to fitness**: if natural sequences are drawn from something like
P(x), then E(x) is a proxy for fitness. SLIP exploits this by using E(x) directly
as a fitness function. However, E(x) reflects evolutionary fitness (reproductive
success averaged over millions of years), which may differ from the laboratory
fitness measured in a DMS experiment (e.g. enzyme activity, binding affinity, or
thermostability). This is why mutascape uses real DMS-measured h_i values rather
than Potts-inferred fields.

**Potts models in mutascape**: we include Potts-derived couplings as one optional
epistasis model (for comparison), but our default approach uses structural contact
maps to determine the topology and draws coupling strengths from parameterised
distributions. This sidesteps the need for MSAs and grounds the epistasis in
physical proximity rather than evolutionary covariation.

### 3.3 Structural Contact Maps

Two residues are "in contact" if their C-beta atoms (C-alpha for glycine) are
within some distance threshold in the 3D structure, typically 8 Angstroms. The
biophysical rationale: residues in spatial proximity can interact through van der
Waals forces, hydrogen bonds, salt bridges, and hydrophobic packing. Mutations at
contacting positions are more likely to have non-additive fitness effects because
they physically influence each other.

We obtain contact maps from predicted structures:

**ESMFold** (Lin et al., 2023): a protein language model (ESM-2) with a structure
prediction head. Takes a single sequence as input, no MSA required. Runs in a
single forward pass on GPU. For monomeric proteins of typical DMS-dataset length
(100-300 residues), predictions are accurate enough for contact map extraction
(the binary contact/no-contact classification is robust even when fine atomic
detail may be off).

**AlphaFold2** (Jumper et al., 2021): higher accuracy, especially for harder
targets, but requires MSA construction and is much slower. For most DMS-dataset
proteins, structures are already in the AlphaFold Protein Structure Database, so
we can download rather than predict.

For mutascape, ESMFold is the default (fast, self-contained, no MSA dependency),
with AlphaFold2 as a validation check.


## 4. The mutascape Fitness Function

### 4.1 Formal Definition

Given:
- A wild-type sequence w of length L
- A set of mutable positions P = {p_1, p_2, ..., p_M} (from DMS data)
- Single-site fitness effects h_i(a) for each position p_i and amino acid a
  (from DMS measurements, relative to wild type)
- A structural contact map C where C_ij = 1 if positions p_i, p_j are in contact
- A pairwise coupling tensor J_ij(a, b) for contacting pairs

The fitness of a variant x that differs from wild type at positions S is:

```
f(x) = sum_{i in S} h_i(x_i) + alpha * sum_{i<j, i in S or j in S} C_ij * J_ij(x_i, x_j) + epsilon
```

Where:
- alpha >= 0 controls global epistatic strength (alpha = 0 is the additive limit)
- epsilon ~ N(0, sigma_noise) is optional measurement noise
- The sum over couplings includes pairs where at least one position is mutated

### 4.2 Coupling Models

**Model 1 - RandomEpistasis**: For each contacting pair (i, j) and amino acid pair
(a, b), draw J_ij(a, b) ~ N(0, sigma_J). The coupling tensor is dense over amino
acid identities but sparse over position pairs (only contacts). Parameters: sigma_J
(coupling spread), seed (reproducibility).

**Model 2 - SignConsistentEpistasis**: Couplings respect the sign structure of
single-site effects. If h_i(a) > 0 and h_j(b) > 0 (both beneficial), then
J_ij(a, b) > 0 (cooperative). If both deleterious, J > 0 (compounding damage).
Mixed signs get J < 0 (antagonistic). Magnitudes drawn from |N(0, sigma_J)|.
This creates landscapes where beneficial mutations tend to cooperate, producing
smoother fitness ridges - a biophysically plausible default.

**Model 3 - PottsEpistasis** (optional): Import J_ij(a, b) from an external Potts
model fit (plmDCA, GREMLIN output). Useful for direct comparison with SLIP. Requires
pre-computed coupling matrices.

### 4.3 Landscape Properties

With these ingredients, mutascape landscapes have several controllable properties:

- **Ruggedness**: increases with alpha and with the density of structural contacts
  in the mutated region
- **Number of local optima**: controlled by the fraction of sign/reciprocal sign
  epistasis, which is influenced by the coupling model choice
- **Correlation with additive prediction**: decreases as alpha increases, providing
  a natural difficulty axis for benchmarking
- **Dimensionality**: number of mutable positions and allowed amino acids per
  position, taken directly from the DMS dataset


## 5. Implementation Architecture

```
src/mutascape/
    __init__.py          # Public API exports
    contacts.py          # ContactMap class - PDB parsing, distance matrix, thresholding
    epistasis.py         # Coupling model classes (Random, SignConsistent, Potts)
    landscape.py         # Landscape class - ties everything together
```

### 5.1 contacts.py

Responsibilities:
- Parse PDB/mmCIF files (via Biopython)
- Extract C-beta coordinates (C-alpha for Gly)
- Compute pairwise distance matrix
- Apply threshold to get binary contact map
- Optionally filter to a subset of positions (the DMS-screened positions)

### 5.2 epistasis.py

Responsibilities:
- Define base class for coupling models
- Implement RandomEpistasis, SignConsistentEpistasis
- Each model takes a contact map and single-site effects and generates J_ij(a, b)
- Coupling tensors stored as sparse structures (only contacting pairs)

### 5.3 landscape.py

Responsibilities:
- Load DMS data (CSV with position, amino acid, fitness columns)
- Combine single-site effects + contact map + coupling model into a fitness function
- Evaluate arbitrary multi-mutant variants
- Sample random variants at specified mutation depths
- Compute landscape statistics (additivity correlation, fraction of sign epistasis)


## 6. pLM Log-Likelihood Scoring as Fitness Proxy

Protein language models (pLMs) trained on evolutionary sequence data learn
residue-level preferences conditioned on structural and functional context.
The masked marginal scoring approach exploits this: mask a single position,
compute the model's predicted probability distribution over amino acids,
and take the log-likelihood ratio relative to wild type:

```
score(pos, aa) = log P(aa | context) - log P(wt_aa | context)
```

This score reflects how compatible the substitution is with the protein's
learned structural and functional constraints, without any training on
experimental fitness data.

Meier et al. (2021) showed that masked marginal scores from ESM-1v
correlate strongly with experimental DMS measurements across 41 deep
mutational scans (Spearman rho 0.47 on average, competitive with
supervised methods). The key insight: evolutionary plausibility (what the
model learned from millions of protein sequences) is a strong prior for
mutational tolerance.

Subsequent work extended this:

- **Brandes et al. (2023)** applied ESM-1b log-likelihood ratios to
  clinical variant effect prediction, showing pLM scores distinguish
  pathogenic from benign variants across the human proteome.
- **Notin et al. (2022)** systematically benchmarked pLM scoring against
  DMS data in ProteinGym, confirming that masked marginal scores from
  ESM-2 and MSA Transformer remain competitive with more complex
  supervised and generative approaches.
- **Frazer et al. (2021)** introduced EVE, a generative model trained on
  MSAs, showing that evolutionary density models can predict variant
  effects. This is conceptually related but uses a VAE rather than masked
  language modelling.

**Relevance to mutascape**: pLM scores provide sequence-based
single-site fitness estimates for positions not covered by experimental
DMS data. This enables full-protein combinatorial landscapes where every
position is mutable, with the additive baseline coming from either
real DMS measurements (where available) or pLM predictions (everywhere
else). The structural contact map then layers epistatic interactions on
top, producing landscapes that are both complete and biophysically
plausible.


## 7. References

- Brandes, N. et al. (2023). Genome-wide prediction of disease variant effects
  with a deep protein language model. Nature Genetics, 55, 1512-1522.
- Frazer, J. et al. (2021). Disease variant prediction with deep generative
  models of evolutionary data. Nature, 599, 91-95.
- Jumper, J. et al. (2021). Highly accurate protein structure prediction with
  AlphaFold. Nature, 596, 583-589.
- Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein
  structure with a language model. Science, 379, 1123-1130.
- Meier, J. et al. (2021). Language models enable zero-shot prediction of the
  effects of mutations on protein function. NeurIPS, 34, 29287-29303.
- Notin, P. et al. (2022). Tranception: protein fitness prediction with
  autoregressive transformers and inference-time retrieval. ICML.
- Sandhu, M. et al. (2025). Investigating the determinants of performance in machine
  learning for protein fitness. Protein Science.
- Thomas, S. et al. (2022). Tuned Fitness Landscapes for Benchmarking Model-Guided
  Protein Design. (SLIP)
- Weinreich, D. et al. (2006). Darwinian evolution can follow only very few
  mutational paths to fitter proteins. Science, 312, 111-114.
