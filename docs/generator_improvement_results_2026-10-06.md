# Generator improvement results

Eight new candidate configurations and a later variance control completed 171,360 simulated campaigns.
An apparent advantage over SLIP and NK in the first experiment did not persist with fresh random seeds.
These results do not establish a superior generator for search-method selection.
The public generator and manuscript remain unchanged.
All changes remain uncommitted on `codex/generator-improvement-experiments`.

## What changed

The [candidate protocol](generator_improvement_protocol_2026-10-06.md) fixed eight configurations before their search outcomes were inspected.
The study changed the sigmoid limits, interaction scaling, beneficial-pair interactions and pooling exponent.
Three configurations also combined measured DMS effects with MSA-derived interactions.
Every candidate preserves WT and measured single-site fitness exactly.
None uses measured combination fitness for fitting.

The DMS-only effect-dependent model replaces a common pair standard deviation with the geometric mean of the two absolute single-site effects.
With single-site effects delta_i and delta_j, its pair term is kappa_ij sqrt(abs(delta_i delta_j)).
The seeded coefficient kappa_ij has unit variance.
This assumes that near-neutral mutations have weaker interactions.
Single-site data do not establish that assumption.

The MSA configurations remove the additive part of an evolutionary Potts model and replace it with measured single-site effects.
They retain its interaction residual after converting the scale using single-site observations.
They require an alignment and are not DMS-only models.

The three-protein panel contains GB1, CreiLOV and eqFP611.
It uses the same seven search spaces, twelve methods and measured reference campaigns as the external generator comparison.
The panel had already influenced model design, so this is an exploratory study.

## First experiment

Each stochastic configuration used five landscape samples and six campaign repeats per sample.
Fixed models received thirty campaign repeats.
The table reports protein-balanced measured regret after choosing a search method from synthetic performance.
Lower regret means that the selected method finds better measured variants.

| Configuration | Regret at 96 evaluations | Kendall correlation |
|---|---:|---:|
| Current Mutascape | 0.01141 | 0.375 |
| Wider sigmoid limits | 0.00686 | 0.584 |
| Effect-dependent pair variance | 0.00414 | 0.595 |
| Negative beneficial pairs, horizon 4 | 0.00587 | 0.550 |
| Negative beneficial pairs, horizon 8 | 0.00695 | 0.589 |
| Power pooling, exponent 1.5 | 0.00612 | 0.617 |
| DMS plus SLIP interactions, horizon 4 | 0.00735 | 0.617 |
| DMS plus SLIP interactions, horizon 8 | 0.00686 | 0.695 |
| DMS plus raw Potts regression | 0.00737 | 0.647 |
| Native SLIP, horizon 4 | 0.00732 | 0.713 |
| NK, K one | 0.00563 | 0.626 |

Effect-dependent pair variance had the best point estimate among the new candidates.
At 96 evaluations, selecting a DMS-only configuration using the other two proteins chose this configuration for all three excluded proteins.
At 48 evaluations, that transfer procedure chose different configurations and did not improve the overall result.
Paired uncertainty intervals for the apparent advantages over SLIP and NK include zero.

![Candidate results](../results/generator_improvements_20261006/figures/candidate_results.png)

## Replication changes the interpretation

The [replication protocol](generator_improvement_replication_2026-10-06.md) retained the leading candidate without changing its settings.
It used ten new landscape samples, six campaign repeats per sample and new measured reference campaigns.
An extra control uses uniform pair variance with the same mean squared coefficient scale as the effect-dependent model.
This control separates total interaction strength from the allocation of that strength between substitutions.

| Generator | First experiment | Fresh-seed replication |
|---|---:|---:|
| Current Mutascape | 0.01141 | 0.02108 |
| Mutascape additive | 0.00737 | 0.00946 |
| Mutascape linear with pairs | 0.00566 | 0.00633 |
| Mutascape pooling | 0.00637 | 0.00665 |
| Effect-dependent pair variance | 0.00414 | 0.01042 |
| Matched uniform pair variance | Not run | 0.00571 |
| Native SLIP, horizon 4 | 0.00732 | 0.00866 |
| NK, K one | 0.00563 | 0.00519 |
| Rough Mount Fuji | 0.02885 | 0.02384 |
| Fixed recommendation from other proteins | 0.00888 | 0.00442 |

Both columns use 96 evaluations and equal protein weights.
These are independent computational repeats on the same proteins, not independent biological replications.
The selected methods and their measured reference estimates can both change between columns.

The apparent effect-dependent advantage disappears.
Its replicated difference from SLIP is 0.00176, with a paired 95% interval of [-0.00201, 0.00409].
Its difference from NK is 0.00523, with an interval of [-0.00098, 0.01343].
Positive differences mean worse measured regret.

The matched uniform control has a better point estimate than SLIP but a worse estimate than NK.
Neither difference is conclusive.
Against SLIP its difference is -0.00295, with an interval of [-0.01042, 0.00260].
Against NK its difference is 0.00052, with an interval of [-0.00178, 0.00542].
Its advantage over the current default also remains uncertain.

![Replication](../results/generator_improvements_20261006/figures/selection_replication.png)

Several methods reach the synthetic maximum and become tied.
Those methods can have different measured performance.
For example, the effect-dependent model initially ties ridge-greedy search and AdaLead on eqFP611.
The replication also ties greedy search and ridge UCB at the top.
The recommendation changes when those extra methods enter the tied set.
The analysis gives tied methods equal selection weights in both experiments.

At 48 evaluations, replicated effect-dependent regret is 0.03014, compared with 0.03285 for SLIP and 0.03297 for NK.
Its uncertainty intervals still include zero against both comparators.
The model is not uniformly worse, but neither budget establishes superiority.

Restricting the replication to the eight model-guided methods gives a Kendall correlation of 0.324 for effect-dependent pairs.
SLIP reaches 0.471 and NK reaches 0.354.
The apparent agreement across all twelve methods partly reflects differences between broad method families.

## Additional protein

The [HIV-1 check](generator_improvement_check_2026-10-06.md) used an available MSA and all 1,175 eligible measured genotypes.
The fixed candidate set was recorded before inspecting its search outcomes.
Every method had measured regret zero because the starting WT and single-site observations already contain the measured maximum.
There is no remaining improvement to search for, and Kendall correlation is undefined.

This case is uninformative about method selection.
It does not support a positive claim and is excluded from the reported three-protein averages.
Its campaigns and results remain saved.
The replication protocol initially planned a four-protein average.
The analysis excludes HIV-1 after detecting this eligibility failure and records that change in its audit.

## Distribution and spectral checks

The initial effect-dependent model reduces mean C2ST AUC from 0.958 for current Mutascape to 0.906.
Measured and generated examples remain readily distinguishable.
Its mean local Walsh distance is 0.237, compared with 0.163 for current Mutascape.
The negative beneficial-pair model at horizon four gives the lowest candidate C2ST AUC, 0.852, but a Walsh distance of 0.285.
Better distribution agreement, closer spectra and useful method recommendations are different outcomes.
No candidate improves all three here.

## Recommendation

Do not promote the effect-dependent configuration on the strength of its first result.
Keep linear and pooling configurations as DMS-only research comparators.
Their method-selection estimates changed less across the two experiments than the leading new candidate's estimate.
The matched uniform control deserves further evaluation, but it has not established an advantage over NK or additive fitness.

The next study should use more unrelated proteins with measured improvement beyond WT and single-site variants.
Use a separate set of proteins to select interaction strength, then fix that rule before evaluating the remaining proteins.
Test the stability of recommendations when several methods nearly solve the synthetic landscape.
Further tuning on these same three proteins would provide weaker evidence than a broader evaluation.

## Reproduction and checks

Run the commands from the Mutascape repository root with the existing environment.
The original external comparison must be present because the candidate study reuses its reference campaigns and MSA fits.
Manifests record the inputs, settings and source hashes.

```sh
.venv/bin/python -m scripts.run_generator_improvements prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m scripts.run_generator_improvements run --jobs 5
.venv/bin/python -m scripts.analyse_generator_improvements
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.diagnose_generator_improvements
.venv/bin/python -m scripts.check_generator_improvements fit
.venv/bin/python -m scripts.check_generator_improvements prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m scripts.check_generator_improvements run --jobs 5
.venv/bin/python -m scripts.replicate_generator_improvements prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m scripts.replicate_generator_improvements run --jobs 5
.venv/bin/python -m scripts.analyse_generator_replication
.venv/bin/python -m scripts.audit_generator_improvements
.venv/bin/python -m scripts.plot_generator_improvements
```

The query audit recalculated all 171,360 new campaigns from 12,337,920 saved query identities.
It found no duplicate queries, charged starting observations, score mismatches, fallback queries or campaign warnings.
All 626 Python tests passed.
Repository lint and the browser checks passed.
The test suite emitted 89 warnings from the existing tests.

The [local Walsh diagram](../results/generator_improvements_20261006/figures/local_walsh_explanation.svg) explains the eight-genotype sets used for the spectral analysis.
