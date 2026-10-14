# Comparison with SLIP and NK

This exploratory study tests whether synthetic campaigns identify useful search methods on measured protein fitness landscapes.
The primary outcome is measured regret after method selection at 96 additional evaluations.
The 48-evaluation result is secondary.
No measured multi-mutant fitness enters generator fitting or difficulty selection.

Use GB1, CreiLOV and eqFP611 because local CombinGym alignments cover searchable regions of these proteins.
Choose the GB1 alignment with coverage at all four experimental sites, even though it has fewer sequences.
Use the CreiLOV b0.7 and eqFP611 b0.5 alignments.
Retain only experimental variants whose substitutions lie in alignment match columns.
Keep all other sites at wild type.
Map GB1 experimental positions 1, 2, 3 and 4 to full-sequence positions 39, 40, 41 and 54.
Validate every wild-type residue in this mapping.

Use three connected search spaces of up to 4,096 genotypes per protein where the available data permit distinct selections.
Use the full eligible search space once when it contains only 2,048 genotypes.
Select connected search spaces through the existing seeded traversal using genotype coverage alone.
This samples regions near wild type, so the results describe those regions.
Give each protein equal weight and average its search spaces before averaging proteins.

Fit the official Mogwai Gremlin model to each full available alignment, using its default loss and optimiser.
Use 1,000 optimisation steps, batches of at most 4,096 sequences, and a fixed training seed.
Save the 50-step and 200-step states and the final state to check training stability.
Preserve interactions with fixed sequence positions when evaluating the experimental mutable region.
Verify reduced scoring against the official SLIP implementation on full aligned sequences.

Tune SLIP with its published controls and target epistatic horizon four.
Set the beneficial single-site fraction to 10% and the fraction of negative interactions among beneficial pairs to 80% for every protein.
This changes the initial zero-offset plan before search outcomes were available.
The GB1 fit has only one beneficial substitution, so its zero-offset horizon is undefined.
The eqFP611 zero-offset formula also reverses coupling signs.
The published shift controls address these tuning failures without using experimental multi-mutant fitness.
Estimate the formula's single-site and pairwise moments over all amino-acid substitutions at aligned positions.
Record negative coupling scales or undefined tuning, rather than hiding them.
Also evaluate the untuned model and horizon eight as sensitivity analyses.
SLIP does not receive experimental single-site fitness.

Compare Mutascape, Mutascape additive and Mutascape zero-pair sigmoid.
Retain the earlier Mutascape linear with pairs and Mutascape pooling settings as exploratory configurations.
Use the earlier equal-weight average of sigmoid, linear and pooling method performance without changing its weights.
It averages search outcomes rather than genotype fitness.
Keep the public generator unchanged.

Use categorical NK with independent uniform contribution tables and one interacting neighbour per site.
Include K zero and K three as sensitivity analyses, with K capped by the number of other sites.
Use Rough Mount Fuji with a random reference genotype, fitness minus Hamming distance, and independent Gaussian noise of standard deviation one.
Include House-of-Cards as a negative control.
These external generators receive the same genotype domain and their own fitness values at the known genotype identities.
Do not overwrite their single-site fitness with experimental values.

Run the same twelve search methods and corrected new-observations-v2 protocol.
Use five landscape samples with six campaign repeats per sample for stochastic generators.
Use thirty campaign repeats for fixed landscapes, including SLIP and both deterministic Mutascape controls.
Match search seeds across methods and synthetic generators.
Use separate measured search seeds to avoid coupling model selection to the evaluation randomness.
Count every new evaluation and record fallback queries, warnings and the queried genotype identities.

Report measured regret after method selection, method-rank correlations, mean squared performance discrepancy and the remaining fraction of available improvement.
Use both a random method choice and a fixed recommendation learned from the other proteins as controls.
Report results for all twelve methods and the eight model-guided methods separately.
Repeat selection within a hierarchical bootstrap over proteins, search spaces, landscape samples and campaign repeats.
Do not count optimisation settings as independent landscape samples.
Intervals describe computational variability rather than experimental measurement uncertainty.

Compare neighbouring fitness correlation, local optima, sign epistasis and local three-site Walsh spectra on identical genotype sets.
Restrict spectral calculations to complete local subcubes selected without fitness.
Report assay-scale distribution metrics only for generators that produce assay-scale fitness.
Native SLIP and NK have arbitrary score units, so their raw C2ST is not a fair comparison.
Record all settings, source hashes, input hashes and software versions before production campaigns.
Retain all results, including settings that compare poorly.
