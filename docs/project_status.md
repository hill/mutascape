# Project status

Checkpoint date is 2026-08-24.

## Publication claim

Mutascape generates synthetic combinatorial protein fitness oracles for search benchmarks. It does not predict one protein's unknown landscape.

The operational validity test is task transfer. A synthetic benchmark should preserve conclusions about which search method performs better on measured combinatorial data.

## Default model

The default is `dms-pairwise-v1`.

1. Wild type and measured single mutants define the target anchor.
2. A fitted saturating readout limits stacked effects.
3. Baseline-corrected random pair tables add shared local variation.
4. A seed fixes one lazy oracle draw.
5. The coupling remains fixed at `0.1`.

The model has no structural contact filter, donor transplant, depth residual, floor texture, or explicit higher-order term.

## Current evidence

The development panel uses 16 complete SSMuLA landscapes. It compares 12 search methods at 96 queries, with 20 search trials and five oracle draws.

- Median Kendall method-rank transfer is `0.625` in the confirmation run.
- Reliable pairwise winner accuracy is `0.845`.
- The measured-reliability pass rate is `0.813`.
- Median measured split-half Kendall agreement is `0.592`.

The external panel uses seven complete GraphFLA protein landscapes. It compares the same 12 methods at 96 queries.

- Median rank transfer is `0.606`.
- Reliable pairwise winner accuracy is `0.827`.
- The measured-reliability pass rate is `0.857`.

Potts remains the strongest external rank baseline at `0.683`. RMF has the highest external winner accuracy at `0.867`. Mutascape does not establish superiority on that panel.

## Known failure

The task-validated default remains too smooth by global topology measures.

- Measured landscapes have a median 67 strict local optima.
- Mutascape and additive both have a median one.
- Measured median non-additive Walsh share is `0.824`.
- Mutascape reaches `0.150`.
- The earlier empirical prior model reaches `0.669`.

The current claim therefore covers method-ranking studies. It excludes claims about peak counts, adaptive paths, or molecular epistasis.

## Keep

- The SSMuLA, CombinGym, and fixed GraphFLA loaders.
- The 12-method budget-matched search runner.
- Trial-level regret and paired winner outputs.
- Measured split-half reliability.
- The saturating readout and exact single-mutant anchors.
- The stable lazy oracle.
- The browser studio and task-transfer chart.
- Additive, Potts, RMF, NK, ridge, and prior-model controls.
- Walsh, local-optima, and navigability diagnostics.

## Keep as research arms

- The empirical higher-order prior.
- Donor interaction transplant.
- Site-pair scales.
- Spectral calibration.
- Structural contacts and MSA couplings.
- Floor texture and censoring policies.
- Point prediction, C2ST, MMD, and interval calibration.

These components remain useful for ablations. They are outside the default generator.

## Remove from the main claim

- Accurate genotype prediction from sparse DMS.
- Calibrated biological uncertainty from oracle draws.
- Whole-protein accuracy at high mutation depth.
- Structural realism without an independent task gain.
- One scalar distribution metric as proof of realism.
- Local-optima matching under the current coupling.

## Next experiments

1. Repeat task transfer at budgets 48, 192, and 384.
2. Measure the coupling trade-off between rank transfer and global topology.
3. Add a new source-held-out protein panel before changing the default.
4. Test whether structure-aware pair scales improve the external panel.
5. Register the next panel and metric before further tuning.

## Reproducibility hooks

- `scripts/run_search_transfer.py` runs the primary task test.
- `scripts/run_topology_benchmark.py` runs secondary shape checks.
- `scripts/plot_benchmark_validity.py` builds the manuscript figures.
- `src/mutascape/data/validation/benchmark-validity-v1.json` drives the studio chart.
- `tests/test_benchmark_validity.py` checks the task metrics.
- `tests/test_benchmark_generators.py` checks exact anchors and lazy equivalence.

## Status flags

- `[ready]` small DMS-pairwise oracle
- `[ready]` exact lazy and enumerated agreement
- `[ready]` 16-landscape development task test
- `[ready]` seven-landscape external protein test
- `[ready]` browser visualisations
- `[ready]` v1 manuscript draft
- `[known gap]` global topology fidelity
- `[pending]` budget sensitivity
- `[pending]` registered second external source
