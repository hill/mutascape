# Publication Roadmap

## Frozen pipeline and validation

- [x] Freeze named reference configs for GB1 and TrpB in code
- [x] Expose paper-critical nonlinearity parameters in the builder path (`sigma`, `noise_sigma`, `floor_hard`, `scale`, `combinatorial_upside`)
- [x] Add regression tests for the reference configs
- [x] Replace notebook-only figure generation with a scripted rebuild path
- [x] Rerun GFP under the current headline pipeline

## Tool-paper write-up

- [x] Rewrite the paper around the thermo + censoring + noise story
- [x] Remove placeholder figures / TODO blocks from the draft
- [x] Align README and project docs with the frozen reference story
- [x] Add manuscript explanation of Walsh-Hadamard spectra
- [x] Label alpha-era docs/notebooks/web demo as historical or legacy
- [ ] Tighten the discussion section after one more proofreading pass

## Benchmark paper next

- [x] Run an initial smoke pilot benchmark on the frozen pipeline with 2-3 methods
- [ ] Run a publication-grade benchmark with enough replicates for uncertainty estimates
- [ ] Sweep `noise_sigma` before introducing additional ruggedness mechanisms
- [ ] Add a hybrid `noise_sigma + sigma_nk` experiment only if it improves order-3 mismatch without ruining `fitness_W1`

## Longer-term research

- [ ] Structural-vs-random contacts ablation under the frozen evaluation metrics
- [ ] Potts / plmDCA couplings as a principled alternative to random pairwise `J_ij`
- [ ] Stronger full-protein validation beyond the current GFP supplementary rerun
