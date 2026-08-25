# Agent hand-off for benchmark validity v1

This document records the state of the Mutascape benchmark-validity work on 25 August 2026.

## Recommended claim

Mutascape generates task-tested synthetic search instances from wild type and measured single mutants. It does not predict an unmeasured combinatorial protein landscape.

The strongest result occurs at 48 queries. Mutascape preserves 88.5 per cent of reliable development winners and 86.4 per cent externally.

The result is budget-specific. No generator leads every metric at 96, 192, and 384 queries.

The next agent should freeze the current generator while reviewing it. Any later model choice needs a new source-held-out panel.

## Branches and locations

| Item | Branch or revision | Location | State |
|---|---|---|---|
| Code worktree | `research/benchmark-validity-v1` | `/Users/tomhill/Uni/research-project/mutascape/.mutascape/worktrees/posterior-landscapes-v1` | Committed through `9163384` |
| Main code repository | Other user work may exist | `/Users/tomhill/Uni/research-project/mutascape` | Do not replace with the worktree |
| Main code audit commit | `9163384` | Code worktree | `audit benchmark task transfer` |
| Earlier generator commit | `1d7d868` | Code worktree | `validate the DMS-pairwise benchmark oracle` |
| Manuscript repository | `research/benchmark-validity-v1` | `/Users/tomhill/Uni/research-project/mutascape_manu_v1` | Committed through `5ed8108` |
| Manuscript commit | `5ed8108` | Manuscript repository | `write the task-relative benchmark manuscript` |
| Manuscript PDF | `5ed8108` | `/Users/tomhill/Uni/research-project/mutascape_manu_v1/main.pdf` | Eight pages |
| First manuscript | Historical source | `/Users/tomhill/Uni/research-project/mutascape_manu_v0/Tom_Hill_1000_Manuscript.pdf` | Used for the first abstract and framing |
| Frozen web result | `9163384` | `src/mutascape/data/validation/benchmark-validity-v1.json` | Reference budget is 96 |
| Development outputs | Local generated data | `notebooks/iclr/output_validity_b{48,96,192,384}_primary` | Ignored by Git |
| External outputs | Local generated data | `notebooks/iclr/output_validity_external_b{48,96,192,384}_primary` | Ignored by Git |
| Benchmark data | Pinned source data | `.mutascape/benchmark-data` | SSMuLA, CombinGym, and GraphFLA roots |
| Browser interface | `9163384` | `web/src/App.tsx`, `web/src/charts.ts`, and `web/src/types.ts` | Four-budget chart added |

The manuscript repository still has user-owned local changes. `bibliography.yml` is modified, and `conferences.md` is untracked. They were not committed.

## Work completed

### Generator scope

The default model is `dms-pairwise-v1`.

It keeps supplied wild-type and single-mutant fitness values exact. It adds stable random pair interactions and a fitted saturating readout.

Pair interactions vanish when either residue has its wild-type state. A seed fixes one lazy and repeatable oracle draw.

The fixed dimensionless coupling is `0.1`. The query cost is quadratic in the number of mutations, rather than the protein length.

The supported domain contains combinations of measured substitutions. A query fails when any constituent single mutant is absent.

### Benchmark framing

The central question is task transfer. A synthetic oracle should preserve conclusions about which search method performs better on measured data.

Three linked measures are primary.

1. Kendall rank transfer compares measured and synthetic method rankings within each landscape.
2. Winner accuracy scores method pairs with a reliable measured winner.
3. Reliability pass compares synthetic transfer with measured split-half repeatability.

Distribution and topology measures remain diagnostics. They cannot establish search-benchmark validity by themselves.

### Fair control design

Mutascape, Potts, and Rough Mount Fuji now receive three matched draws. Earlier work averaged five Mutascape draws but only one control draw.

The Potts control now keeps wild type and every single mutant exact. It differs from Mutascape through its readout and coupling.

The additive and empirical-prior arms remain deterministic comparison points. NK and fitted ridge arms remain exploratory.

### Dataset audit

The development panel contains 16 complete SSMuLA landscapes. Ten are related TrpB libraries, so the panel does not contain 16 independent proteins.

The external panel contains seven GraphFLA protein landscapes from distinct families. Each source now has a declared reference sequence.

The loader no longer treats the first file row as wild type. It also sorts `posN` columns by numeric position.

Tests verify that row order and mutation-column order do not change a loaded landscape.

### Search audit

The study compares 12 search methods at budgets of 48, 96, 192, and 384 queries.

Each panel uses ten search trials and three matched generator draws. The acquisition pool contains at most 20,000 candidates.

Search paths now remain unchanged after any positive affine transformation of fitness. Regression tests cover all 12 methods.

Neural-network targets are standardised. Forest methods train on stable rank targets. Simulated annealing scales acceptance by observed variation.

The neural-network method now uses deterministic early stopping. A 384-query smoke study completed 5,600 fits without a convergence warning.

### Result validation

The runner now rejects duplicate trial keys, non-finite values, out-of-range regret, and incomplete trial groups.

Restricted method runs keep a valid empty schema. Constant rankings return an undefined coefficient without a SciPy warning.

High-budget runs no longer drop landscapes with fewer than four times the query budget. This keeps the panel fixed across budgets.

### Interface and manuscript

The browser studio reads the frozen JSON result. It retains 96 queries as the preselected reference budget.

A new chart shows reliable winner accuracy across four budgets. Solid lines show development results, and dashed lines show external results.

The chart compares Mutascape, Potts, and RMF. The live interface was tested with GB1 sampling and an exact `V39A+D40A` query.

The manuscript now has eight pages and 24 cited sources. It defines task-relative realism and reports the whole budget sweep.

The related-work section covers SLIP, SSMuLA, GraphFLA, FLIP2, CombinGym, NK, RMF, global epistasis, and recent landscape methods.

## Work tried and moved out of the main claim

| Component | Observation | Current status |
|---|---|---|
| Additive generator | Often preserves single effects too strongly | Main control |
| Low-coupling Potts | Strong matched pairwise control | Main control |
| Rough Mount Fuji | Strong at some budgets and winner measures | Main control |
| Earlier empirical prior | Added depth, texture, and higher-order variation without consistent transfer gains | Ablation only |
| Structural contact filters | Plausible mechanism without an independent task gain | Research arm |
| MSA couplings | Related to SLIP, but not needed for the current positive result | Research arm |
| Donor interaction transplant | Added complexity without a stable main-task gain | Research arm |
| Site-pair scales | Added parameters weakly identified by singles | Research arm |
| Explicit higher-order priors | Improved some shape measures but reduced interpretability | Research arm |
| Assay-floor texture | Changed topology but did not support the main transfer result | Research arm |
| Walsh spectra | Useful interaction-order diagnostic | Secondary only |
| Strict local optima | Useful graph-shape diagnostic | Secondary only |
| Wasserstein distance | Misses genotype-graph arrangement | Optional only |
| Classifier two-sample test | Depends on features and power | Optional only |
| MMD | Distribution check with the same graph limitation | Optional only |

The old components remain available for ablations. They should not return to the default without new task-transfer evidence.

## Experimental protocol

Each target exposes wild type and all supported single mutants. Every higher-order measured variant stays hidden from the generator.

Measured and synthetic arms use the same candidates, feature encoding, trial seeds, and query budget.

Normalised simple regret is the global fitness gap divided by the full measured fitness range.

The study averages regret across trials for each method. It then compares measured and synthetic method rankings within each landscape.

A paired bootstrap uses 1,000 resamples for each method pair. A pair enters winner accuracy only when its measured interval excludes zero.

Measured reliability uses 500 random split-half partitions. The fifth percentile defines a landscape-specific reliability floor.

## Development results

Each cell reports `Kendall tau / winner accuracy / reliability pass rate`.

| Budget | Mutascape | Potts | RMF | Empirical prior | Additive |
|---:|---:|---:|---:|---:|---:|
| 48 | 0.529 / 0.885 / 0.875 | 0.446 / 0.812 / 0.750 | 0.515 / 0.865 / 0.813 | 0.318 / 0.728 / 0.563 | 0.087 / 0.575 / 0.438 |
| 96 | 0.454 / 0.787 / 0.688 | 0.401 / 0.758 / 0.625 | 0.458 / 0.823 / 0.688 | 0.248 / 0.714 / 0.563 | 0.293 / 0.699 / 0.500 |
| 192 | 0.505 / 0.737 / 0.938 | 0.550 / 0.706 / 0.875 | 0.342 / 0.728 / 0.625 | 0.350 / 0.784 / 0.625 | 0.467 / 0.726 / 0.750 |
| 384 | 0.462 / 0.645 / 0.813 | 0.498 / 0.645 / 0.750 | 0.326 / 0.680 / 0.438 | 0.418 / 0.798 / 0.625 | 0.418 / 0.657 / 0.688 |

The 48-query Mutascape arm passes 14 of 16 reliability floors. Potts passes 12, and RMF passes 13.

At 96 queries, Mutascape exceeds the empirical prior and additive on 14 development landscapes. It exceeds Potts on nine and ties on two.

The Mutascape comparison with RMF splits eight to eight at 96 queries.

## External results

Each cell reports `Kendall tau / winner accuracy / reliability pass rate`.

| Budget | Mutascape | Potts | RMF | Empirical prior | Additive |
|---:|---:|---:|---:|---:|---:|
| 48 | 0.439 / 0.864 / 1.000 | 0.337 / 0.764 / 0.714 | 0.230 / 0.749 / 0.714 | 0.337 / 0.712 / 0.571 | 0.485 / 0.759 / 0.857 |
| 96 | 0.644 / 0.742 / 0.571 | 0.642 / 0.781 / 0.571 | 0.388 / 0.787 / 0.429 | 0.328 / 0.697 / 0.143 | 0.644 / 0.703 / 0.857 |
| 192 | 0.657 / 0.848 / 0.714 | 0.603 / 0.838 / 0.857 | 0.432 / 0.960 / 0.571 | 0.464 / 0.929 / 0.429 | 0.674 / 0.828 / 0.857 |
| 384 | 1.000 / 0.640 / 1.000 | 1.000 / 0.663 / 1.000 | 0.606 / 0.910 / 0.333 | 0.724 / 0.573 / 0.500 | 1.000 / 0.427 / 1.000 |

The 48-query Mutascape arm passes all seven external reliability floors. Additive has higher rank transfer but lower winner accuracy at this budget.

Only six external landscapes have an evaluable reliability score at 384 queries. Lunzer has a degenerate measured ranking at this coverage.

The rank coefficient reaches one for Mutascape, Potts, and additive at 384 queries. Many search methods find the optimum, which compresses their ranking.

This saturation must not be described as full landscape recovery.

## Main observations

1. Mutascape has a clear low-budget decision-transfer result.
2. Potts and RMF remain strong controls and sometimes lead later budgets.
3. Rank transfer and winner accuracy can favour different generators.
4. High query coverage can make rank transfer look better while winner accuracy falls.
5. No single distribution or topology measure captures benchmark validity.
6. An ensemble of draws measures benchmark sensitivity, not biological uncertainty.
7. The current evidence supports method-ranking studies within the tested protocol.

The paper should not claim broad generator superiority. It should not claim accurate genotype values, molecular epistasis, or calibrated uncertainty.

## Important fixes and their effects

| Issue | Effect before the fix | Fix and check |
|---|---|---|
| Unequal generator draw counts | Reduced Mutascape variance compared with controls | Three matched draws for Mutascape, Potts, and RMF |
| Potts changed wild type and singles | Confounded the readout comparison | Pair terms vanish at wild-type states, with exact-anchor tests |
| GraphFLA used the first row as reference | Changed the genotype origin after row reordering | Declared reference sequences and ordering regression tests |
| GraphFLA positions used file order | Could reorder positions lexicographically | Numeric `posN` sorting and column-order tests |
| High budgets removed small landscapes | Changed the panel between budgets | Keep any landscape with at least four candidates |
| Annealing used raw fitness magnitude | Changed queries after an offset or scale change | Acceptance uses observed variation and fitness gaps |
| MLP trained on raw assay units | Changed acquisition across assay scales | Standardised targets and deterministic early stopping |
| Forest used raw fitness values | Changed fitted paths across assay scales | Stable rank targets and rounded acquisition scores |
| Constant ranks reached SciPy | Produced warnings and misleading output | Record an undefined coefficient directly |
| Restricted runs lost output columns | Broke downstream analysis | Return a fixed transfer schema |
| Score tables lacked a final audit | Allowed incomplete or invalid results | Validate trial keys, values, regret, and group sizes |

## Known issues and limitations

- The development panel contains ten related TrpB landscapes.
- The external panel has only seven protein families.
- Ten search trials give limited resolution for close method pairs.
- The current panels have guided model selection and must now count as development data.
- Global topology remains smoother than many measured landscapes.
- Pairwise latent effects cannot represent every higher-order mechanism.
- Whole-protein lazy queries can extrapolate far beyond the measured mutation depth.
- Oracle-draw intervals do not have calibrated biological coverage.
- External reliability has one undefined landscape at 384 queries.
- Generated result directories are ignored by Git and exist only on this machine.
- The manuscript repository contains user changes outside the committed manuscript work.

## Verification performed

- `449` Python tests passed.
- Ruff passed for `src`, `tests`, and `scripts`.
- Biome passed for the web source.
- `10` Bun tests passed.
- The production browser bundle built successfully.
- The frozen JSON regenerated byte-for-byte from all eight result directories.
- The local browser loaded the budget chart and returned an exact variant query.
- Typst compiled the manuscript to eight letter-sized pages.
- Every manuscript page was rendered to PNG and inspected.
- The prose linter reported no hard wording fault in the edited Markdown files.

Expected test warnings come from legacy single-site imputation and one pending Seaborn deprecation. The audited search runs produced no MLP convergence warning.

## Reproduction commands

Run these commands from the code worktree.

```bash
just benchmark-data

for budget in 48 96 192 384; do
  uv run python scripts/run_search_transfer.py \
    --ssmula-root .mutascape/benchmark-data/ssmula \
    --combingym-root .mutascape/benchmark-data/combingym \
    --output "notebooks/iclr/output_validity_b${budget}_primary" \
    --budget "$budget" \
    --seeds 10 \
    --draws 3 \
    --generators primary \
    --dms-pairwise-coupling 0.1
done
```

Run the external panel with these commands.

```bash
for budget in 48 96 192 384; do
  uv run python scripts/run_search_transfer.py \
    --ssmula-root .mutascape/benchmark-data/ssmula \
    --combingym-root .mutascape/benchmark-data/combingym \
    --graphfla-root .mutascape/benchmark-data/graphfla \
    --output "notebooks/iclr/output_validity_external_b${budget}_primary" \
    --budget "$budget" \
    --seeds 10 \
    --draws 3 \
    --generators primary \
    --dms-pairwise-coupling 0.1
done
```

Freeze the result after every run completes.

```bash
uv run python scripts/freeze_benchmark_validity.py \
  --development 48=notebooks/iclr/output_validity_b48_primary \
  --development 96=notebooks/iclr/output_validity_b96_primary \
  --development 192=notebooks/iclr/output_validity_b192_primary \
  --development 384=notebooks/iclr/output_validity_b384_primary \
  --external 48=notebooks/iclr/output_validity_external_b48_primary \
  --external 96=notebooks/iclr/output_validity_external_b96_primary \
  --external 192=notebooks/iclr/output_validity_external_b192_primary \
  --external 384=notebooks/iclr/output_validity_external_b384_primary \
  --primary-budget 96
```

Run the checks with these commands.

```bash
just test
just lint
cd web
bun test
bun run build
```

Build the manuscript from its repository.

```bash
cd /Users/tomhill/Uni/research-project/mutascape_manu_v1
typst compile main.typ
```

## Key entry points

| File | Role |
|---|---|
| `src/mutascape/completion.py` | Conditional generator and lazy sampled oracle |
| `src/mutascape/benchmarks/generators.py` | Mutascape, Potts, RMF, NK, and other benchmark arms |
| `src/mutascape/benchmarks/datasets.py` | SSMuLA, CombinGym, and GraphFLA manifests and loaders |
| `src/mutascape/benchmarks/methods.py` | The 12 search methods and affine-invariant acquisition logic |
| `src/mutascape/benchmarks/validity.py` | Reliability and paired winner measures |
| `scripts/run_search_transfer.py` | Trial runner, validation, and result aggregation |
| `scripts/freeze_benchmark_validity.py` | Frozen JSON builder and consistency checks |
| `scripts/plot_benchmark_validity.py` | Manuscript figure builder |
| `docs/benchmark_protocol.md` | Short protocol specification |
| `docs/project_status.md` | Current claim and project choices |
| `web/src/App.tsx` | Studio cards and result layout |
| `web/src/charts.ts` | Reference and budget chart configuration |
| `/Users/tomhill/Uni/research-project/mutascape_manu_v1/main.typ` | Manuscript source |

## Recommended next steps

1. Review commits `1d7d868`, `9163384`, and `5ed8108` before changing code.
2. Treat the current development and GraphFLA panels as used data.
3. Register one new source-held-out protein panel before changing the generator.
4. Increase search trials for the 48-query and 96-query confirmatory runs.
5. Add uncertainty intervals for panel medians and pooled winner accuracy.
6. Keep the method set, candidate pools, metrics, and coupling fixed during confirmation.
7. Prepare a tagged package release and an archived result bundle after review.

The smallest useful next experiment is a higher-trial rerun at 48 queries with no model change. A new external source is more valuable than another metric.
