# Archived scripts

## plm_direct_gb1.py

One-off experiment (2026-04-18) that scored every GB1 combinatorial variant with ESM2 `esm2_t6_8M_UR50D` unmasked pseudo-log-likelihood and calibrated linearly to DMS singles. Result was a clear negative: `tensor_r = 0.006` (no correlation with real fitness), `walsh_L1 = 1.32` (worse than Identity baseline), and the calibration slope came out negative on singles — unmasked scoring measures "model confidence in its own input" rather than variant effect.

See `docs/experiments_log.md` for the full writeup. Kept for reproducibility of the negative result. If you want to revisit this direction, use masked-marginal scoring on `esm2_t33_650M_UR50D` with a GPU.
