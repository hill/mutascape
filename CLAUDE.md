# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
just sync            # uv sync --extra dev --extra web
just test            # full suite
just test tests/test_fourier.py::test_walsh_fractions_sum_to_one   # single test
just lint            # Ruff for Python and Biome for TypeScript
just fmt             # apply Ruff and Biome fixes
```

Web stack (Bun frontend plus FastAPI oracle):

```bash
just dev             # builds frontend, serves both on :8000
just web-check       # lint, type-check, test, and build the studio
just docs-assets     # rebuild documentation figures and generated facts
```

The suite takes about 70 seconds. Most of that is `tests/test_reference_configs.py`, which builds complete 20^4 landscapes and computes exact Walsh decompositions.

## Documentation website

The documentation pages live in `web/src/docs/`. Keep each page focused on an end-user task. Use the public package API in every code sample.

Update the site in the same change when you alter a public API, default value, benchmark protocol, claim boundary, or studio workflow. Run `just docs-assets` when an example output, plot, signature, or docstring changes. `just web-check` rejects stale generated assets.

`scripts/build_docs_assets.py` runs the examples against tracked data. It also reads public signatures and summaries from docstrings. Keep generated facts in `web/data/docs.generated.json`. Keep generated figures in `web/assets/docs/`.

## What this produces

Synthetic combinatorial protein fitness landscapes for benchmarking MLDE and ALDE methods. Generated combinations do not predict true biological fitness. The public API preserves supplied wild-type fitness.

## Current public generator

`mutascape.from_dms` in `api.py` returns a `DMSGenerator`. Its `draw(seed)` method returns a fixed `DMSLandscape` oracle with generator id `dms-sigmoid-v1`. `model.py` defines both.

The model takes WT and singles only. It sums inverse-sigmoid single effects and a seeded pair field, then maps the total through a sigmoid bounded 5% beyond the singles range. The default pair strength is 0.3. WT and singles return their measurements exactly.

`from_dms(..., pilot=...)` returns `QuantileCalibratedLandscape` from `calibrated.py`, which also fits measured combinations.

`mutascape.legacy` keeps the earlier empirical-prior `PosteriorLandscape` and its `dms-pairwise-v1` manifests. The studio still uses it through `completion.py` and `priors.py`. `load_oracle` routes old manifests there.

Unknown singles raise an error unless the user requests `complete="esm2"`. ESM completion fills missing substitutions, preserves measurements, and records query provenance. Saved manifests freeze imputed effects.

`results/creilov_c2st/` contains the raw adversarial audit. It exposes a calibration failure. Pilot experiments remain separate and must not change the public default without validation.

The structural model below is a legacy API. Its defaults and regression metrics do not describe the current public generator.

## Legacy structural model

Everything centres on `Landscape.evaluate` in `src/mutascape/landscape.py`. Reading that one method explains most of the package:

```
f(x) = g( Σ h̃ᵢ(xᵢ) + alpha·Σ Cᵢⱼ Jᵢⱼ + alpha_ho·H(x) + noise ) + readout texture
```

| Term | Lives in |
|---|---|
| `h̃ᵢ` additive singles, cached as preimages `g⁻¹(hᵢ)` | `landscape.py::_cache_preimages` |
| `Jᵢⱼ` pairwise couplings, `H` higher-order Contact-NK | `epistasis.py` |
| `Cᵢⱼ` structural contact mask | `contacts.py` |
| `g` global-epistasis function | `nonlinearity.py` |
| readout texture | `landscape.py::_readout_texture_value` |

Invariants worth knowing before you touch any of it:

- **Singles are summed in preimage space**, so `g` wraps the whole function and a single mutant's predicted fitness matches its DMS observation exactly.
- **The readout texture is applied after `g`, not inside it**, and only to variants sitting on the censored floor. It breaks exact low-fitness plateaus into stable local topology, which is what brings synthetic `n_optima` near the real value.
- **Noise is pinned to genotype by default** via a blake2b hash of seed plus mutation label. Repeated queries for the same variant return the same number, so the landscape works as a stable oracle. `deterministic_noise=False` restores fresh draws.
- **Pairwise couplings carry a WT baseline correction**, the double difference `J(mᵢ,mⱼ) - J(wᵢ,mⱼ) - J(mᵢ,wⱼ) + J(wᵢ,wⱼ)`. Single mutants therefore get no epistatic contribution, and iterating over mutated pairs only is equivalent to iterating over all pairs.
- **`evaluate` strips entries matching wild type first**, so a sparse mutation dict and a full WT-context dict give the same answer.
- **Unmeasured `(pos, aa)` effects are imputed by position mean**, never defaulted to 0. A 0 default would inject fake neutral mutations into every multi-mutant and bias the landscape toward additivity.

## Legacy reference configs

`src/mutascape/validation/reference_configs.py` holds the older `GB1_HEADLINE` and `TRPB_HEADLINE` pipelines. `tests/test_reference_configs.py` checks their regression metrics within tolerance bands.

Those tests are the tripwire for any change to `landscape.py`, `epistasis.py`, `nonlinearity.py` or `validation/fourier.py`. A failure is either a real regression or a deliberate recalibration. If deliberate, update the numbers in both the test and the README.

Random pairwise couplings are **off by default** (`sigma = 0`). Setting `sigma > 0` measurably worsened headline metrics on both GB1 and TrpB, so a change that turns them on needs validation evidence, not just passing tests.

`validation/calibration.py::build_landscape` is the central builder. Scripts and reference configs both go through it rather than constructing `Landscape` directly.

## Layout

- `src/mutascape/` core package. `landscape.py` (the oracle, with `from_dms` and `from_plm` constructors), `epistasis.py`, `nonlinearity.py`, `contacts.py`, `scoring.py` (ESM2), and `mutant.py` (a `dict` subclass supporting `+` and `-`).
- `web/` React studio and FastAPI application. Run the server through `web.server:app`.
- `src/mutascape/validation/` metrics and the frozen pipeline. `fourier.py` computes the headline `walsh_L1`, `metrics.py` covers local optima, accessible paths and landscape attributes, `real_data.py` loads ProteinGym-style CSVs.
- `scripts/` versioned analysis runs. `rebuild_reference_outputs.py` regenerates the paper figures and sweep CSVs.

## Data

`data/<protein>/` holds prepared inputs. The CSVs and a whitelist of small `.npz` distance and coupling matrices are tracked. PDB files, raw upstream downloads and large regenerable tensors are not.

`.gitignore` uses a broad ignore plus explicit negations for the six `.npz` files the frozen pipeline needs at clone time. Adding a protein means adding its `.npz` to that whitelist, otherwise the pipeline breaks on a fresh clone. The CreiLOV scripts already have this problem, since they need `creilov_msa_couplings.npz` and `creilov_combinatorial_raw.xlsx` and neither is tracked.

`prepare_*.py` scripts convert raw upstream data into the tracked CSVs. They do not download anything, they raise with the source URL.

Generated outputs under `notebooks/validation/output_*` and `notebooks/benchmark_pilot/` are ignored. Rebuild them rather than committing them.

## Notebooks

Not the source of truth. Per `notebooks/validation/README.md`, do not cite a notebook cell in the paper unless a versioned script reproduces the same output.

## Prose

A style hook lints markdown on edit and blocks on hard violations, so editing a doc can surface pre-existing problems in it. `docs/references.md` is non-compliant and deliberately left that way. Do not get pulled into fixing it while doing unrelated work.
