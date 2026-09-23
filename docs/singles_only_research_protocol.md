# Singles-only generator research

This study returns to the input requirement in the original technical note.
The generator receives WT sequence, WT fitness and single-mutant measurements from one assay.
Optional contacts restrict the random pair effects.
Target combinations supply evaluation labels only.
The current pilot model and manuscript remain available on their existing branches.

## Model families

All families preserve supplied singles and expose stable genotype queries.
They need no complete-library enumeration.
The linear family adds single effects and Gaussian pair effects.
The sigmoid family first converts singles to a latent scale.
Its limits extend beyond the observed single-mutant range by a fixed fraction.
The pool family combines positive and negative effects through their separate Euclidean norms.
This last family tests buffering as a simple modelling assumption.

Pair strengths are 0, 0.1, 0.3 and 1 in single-effect standard-deviation units.
Sigmoid margins are 0.05 and 0.5 of the observed range.
Three fixed seeds describe generator variation.
No parameter uses a target combination measurement at construction or query time.

## Selection and evaluation

Development proteins are DHFR, eqFP611, ParB-Noc, ParD2 and CreiLOV.
Settings within each family use only these five proteins.
Confirmation proteins are GB1, TrpB4, T7 and TEV.
They appeared in earlier project work but do not select settings in this study.
All source collections and assay transformations retain their existing provenance.

The setting score averages ranks of two diagnostics within each development protein.
These are scaled Wasserstein distance and local three-site Walsh distance.
The second metric averages spectra over complete measured binary subcubes.
It retains allele identities and requires no filling of missing measurements.
It differs from the previous full-library spectrum and the CreiLOV site-pattern projection.
The same sampled subcubes apply to measured and generated values.
The protocol reports both diagnostics separately.

The final comparisons retain additive and fixed random-pair controls.
They also include contact-restricted versions where residue mapping permits them.
Search tests use all 12 existing methods, ten search seeds and budgets of 48 and 96 queries.
All models receive the same candidate genotypes.
Target combinations remain outside model construction.
Measured combinations define the search reference.

C2ST uses 1,000 measured multi-mutants, or every eligible genotype when fewer exist.
Measured and generated fitness for one genotype remain in the same fold.
Five-fold boosted classifiers use the existing assay features, with and without sequence indicators.
C2ST remains a diagnostic alongside search transfer and interaction structure.
The study reports failures as well as improvements.

## Interpretation

Singles do not identify arbitrary interaction terms.
The generator therefore supplies a family of synthetic environments under explicit assumptions.
Structural contacts restrict possible interactions but do not determine their signs or assay-scale effects.
Matching one distribution or spectrum alone does not validate method selection.
Any revised manuscript claim must follow the held-out search results.
