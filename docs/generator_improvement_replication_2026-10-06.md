# Replication and interaction-scale control

The first candidate study favoured effect-dependent pair variance at 96 evaluations.
This follow-up checks Monte Carlo reproducibility on the same four proteins.
It does not supply four new biological replicates.
The HIV-1 candidate results have not been inspected when fixing this protocol.

Retain the candidate without changing its strength of one.
Compare it with current Mutascape, Mutascape additive, Mutascape linear, Mutascape pooling, native SLIP with horizon four, NK with K one and Rough Mount Fuji.
Add one uniform-variance control with the same mean squared pair coefficient scale as the effect-dependent model.
Average that scale across every allowed pair of substitutions at distinct sites.
Calculate it using measured single-site effects only.
This separates the allocation of interaction variance from its overall strength.

Use ten new landscape seeds, 2026103000 through 2026103009.
Run six campaign repeats per sample and all twelve search methods at both budgets.
Use synthetic campaign seeds starting at 2026105000 and independent measured campaign seeds starting at 2026104000.
Increment campaign seeds by 137 for each repeat.
Reuse the genotype search spaces and the MSA fits.
Run new measured campaigns and every control with the same new campaign seeds.
Store query identities and check every evaluation budget.

Report the three-protein panel, HIV-1 and the four-protein average separately.
The primary outcome remains measured regret after method selection at 96 evaluations.
Report 48 evaluations and Kendall correlation as secondary outcomes.
Use paired bootstrap intervals that repeat method selection, with ten landscape blocks and six campaign repeats per block.
Keep all outcomes even if the earlier advantage disappears.
