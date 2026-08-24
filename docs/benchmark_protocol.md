# Task-transfer benchmark protocol

## Research question

Can a synthetic oracle preserve method comparisons from measured combinatorial protein landscapes?

The protocol does not test genotype prediction. It tests whether a synthetic benchmark supports similar search conclusions.

## Generator input

Each target supplies these fields.

- A wild-type sequence.
- Wild-type fitness.
- One measured fitness for each supported substitution.
- Optional measured combinations.
- A declared position alphabet.

An unseen query is valid only when every constituent single was measured. A supplied combination keeps its measured value.

## Generator

The primary arm is `dms-pairwise-v1` with coupling `0.1`.

The generator composes measured singles through a fitted saturating readout. Stable baseline-corrected random pair tables add local variation. A seed fixes one lazy oracle draw.

An ensemble measures sensitivity across benchmark instances. It does not represent calibrated biological uncertainty.

## Development split

The development panel contains 16 SSMuLA landscapes. Related landscapes share a protein group.

Each target exposes wild type and all supported singles. Every higher-order value remains hidden until search scoring. Readout calibration excludes the target protein group.

The coupling was selected through a small four-method development sweep. All 12-method confirmation runs keep it fixed.

## External split

The external manifest contains seven GraphFLA protein landscapes. Each belongs to a different protein family and contains at least 512 complete variants.

The panel contains HisA-HisF, LamB, influenza haemagglutinin, Hsp90, ParB, beta-lactamase, and SARS-CoV-2 RBD landscapes.

Promoter, DNA binding-site, and RNA landscapes are excluded from the protein claim. No external target enters coupling selection or prior fitting.

## Search methods

The primary comparison contains 12 methods.

- Random search.
- Greedy local search.
- Simulated annealing.
- Genetic search.
- Ridge greedy acquisition.
- Ridge upper confidence bound.
- Gaussian process expected improvement.
- Gaussian process upper confidence bound.
- Random forest greedy acquisition.
- Random forest Thompson sampling.
- Multilayer perceptron upper confidence bound.
- AdaLead.

Every method receives the same candidates, query budget, feature encoding, and trial seeds.

## Search protocol

The primary budget is 96 queries. Development uses 20 search trials and five generator draws. External validation uses 10 trials and three draws.

The acquisition pool contains at most 20,000 candidates. Normalised simple regret is

`(global best - best found) / (global best - global worst)`.

Trial-level output records target, protein, arm, method, trial, best fitness, regret, space size, coverage, and runtime.

## Primary metrics

### Method-rank transfer

Average regret across trials for each method. Calculate Kendall agreement between measured and synthetic method rankings within each landscape. Report the median across landscapes.

### Reliable pairwise winners

Pair measured trials by seed. Bootstrap each method-pair regret difference 1,000 times. Score a pair only when its 95 per cent interval excludes zero.

Winner accuracy is the share of reliable pairs whose synthetic and measured winner agrees. A synthetic tie counts as a miss.

### Measured reliability floor

Split measured trials into two halves 500 times. Calculate Kendall agreement between the two method rankings.

The fifth percentile is the landscape reliability floor. A synthetic arm passes when its transfer reaches that floor.

## Secondary diagnostics

Secondary checks include these measures.

- Walsh or multiallelic interaction-order variance.
- Strict local optima and peak fraction.
- Beneficial fraction relative to wild type.
- Greedy peak percentile and regret.
- Fitness Wasserstein distance.
- C2ST and MMD when provenance controls are valid.

These metrics diagnose failures. No one measure establishes benchmark validity.

## Baselines

The required controls are additive, Potts, Rough Mount Fuji, NK, and pairwise ridge. The previous empirical prior model remains an ablation.

Potts is the closest baseline because it combines measured fields with random pair tables. Mutascape adds the fitted readout and strict baseline correction.

## Acceptance rule

The default supports a method-ranking claim when it meets both conditions.

1. Pairwise winner accuracy exceeds 90 per cent on development and external panels.
2. The reliability pass rate exceeds 80 per cent on both panels.

Median rank transfer provides a graded companion result. Potts performance must appear beside Mutascape.

The current default meets this method-ranking rule. It does not meet a global topology rule.

## Reporting checklist

Every benchmark report should state these items.

- Target positions, alphabets, wild type, and maximum depth.
- Measured and imputed substitutions as separate counts.
- Generator version, coupling, and draw seeds.
- Search methods, budget, trials, and candidate pool.
- Trial-level regret or an accessible result table.
- Method-rank transfer and pairwise winner intervals.
- Known topology gaps.

## Reproduction

Run the development panel with this command.

```bash
uv run python scripts/run_search_transfer.py \
  --ssmula-root .mutascape/benchmark-data/ssmula \
  --combingym-root .mutascape/benchmark-data/combingym \
  --output notebooks/iclr/output_search_transfer \
  --budget 96 --seeds 20 --draws 5 --generators \
  --dms-pairwise-coupling 0.1
```

Add `--graphfla-root` for the fixed external panel. Source manifests pin the required revisions.
