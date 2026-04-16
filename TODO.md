# Publication Roadmap

## Critical

- [ ] **Validation against real multi-mutant data** -- compare mutascape landscape statistics against GFP combinatorial data (Sarkisyan et al., 2016). Find alpha that matches real epistasis patterns. See `experiments/validation/`.
- [ ] **Benchmarking study** -- run MLDE methods (from mutantopt) across alpha sweep, show how method performance degrades with increasing epistasis.
- [ ] **Structural vs random contacts ablation** -- generate landscapes with random contacts (same density), show structural contacts produce more realistic distance-dependent epistasis.

## Important

- [ ] **PottsEpistasis implementation** -- import couplings from inverse Potts model fits for direct comparison with SLIP (Thomas et al., 2022).
- [ ] **Extended landscape characterisation** -- local optima count, fitness-distance correlation, neutral network connectivity as functions of alpha.

## Moderate

- [ ] **GB1 validation** -- second protein (Wu et al., 2016; 4-position combinatorial, ~150k variants) for generalisability.

## Final

- [ ] **Write-up** -- paper figures, methods, results. Target: Protein Science or NeurIPS Datasets & Benchmarks.
