# Literature Basis for Mutascape Version One

## Positioning

Mutascape should focus on DMS-conditioned generation and posterior evaluation. Existing projects already cover landscape analysis, fitness prediction, and search benchmarking.

| Work | Main method | Evaluation used | Relevance to Mutascape |
| --- | --- | --- | --- |
| [GraphFLA](https://papers.neurips.cc/paper_files/paper/2025/hash/323fcf83a0f26c7c04eae0d91125d518-Abstract-Datasets_and_Benchmarks_Track.html) | Graph analysis of biological landscapes | Twenty topographic features across 5,300 landscapes | Reuse its metric definitions and avoid claiming general landscape analysis as new. |
| [CombinGym](https://github.com/sitonglab/CombinGym) | Prediction from lower mutation orders | 0-vs-rest through 3-vs-rest, Spearman, and NDCG | Adopt its order splits and design metric. |
| [FLIP2](https://www.biorxiv.org/content/10.64898/2026.02.23.707496v2) | Protein fitness prediction under distribution shift | Number, position, mutation, fitness, and scaffold shifts | Keep simple baselines and report grouped transfer. |
| [SSMuLA](https://doi.org/10.1016/j.cels.2025.101387) | MLDE and ALDE across 16 small landscapes | Fifty repeats, 96 and 384 samples, local optima, and epistasis | Match its query budgets and use its empirical landscapes. |
| [Martí-Gómez and McCandlish](https://pubmed.ncbi.nlm.nih.gov/42395394/) | Empirical Bayes inference with site-specific epistasis | Held-out fitness and structured interaction recovery | Compare against local epistasis regression when code becomes available. |
| [Spence et al.](https://www.biorxiv.org/content/10.1101/2025.04.12.648556v2) | Diffusion-based graph ruggedness | Sparse and complete empirical landscapes | Add local ruggedness and motivate composed biological constraints. |
| [FLIGHTED](https://pmc.ncbi.nlm.nih.gov/articles/PMC10996587/) | Bayesian assay observation models | Predictive error and uncertainty calibration | Keep assay noise separate from latent landscape variation. |
| [Zhu et al.](https://proceedings.mlr.press/v258/zhu25c.html) | Spectral graph theory and PCNNs | Prediction and optimisation from sparse samples | Compare spectral summaries and graph-based baselines. |
| [FLEXS](https://flexs.readthedocs.io/en/latest/) | Protein sequence optimisation harness | Fitness reached under query budgets | Offer an adapter after native search tests pass. |
| [Proto](https://arcinstitute.org/news/proto) | Composed generative biology programs | Constraint satisfaction and design workflows | Defer its adapter until the oracle has validated search behaviour. |

## Generative evaluation

[Classifier two-sample tests](https://arxiv.org/abs/1610.06545) ask whether held-out classifiers distinguish real and generated samples. Mutascape applies C2ST to fitness-labelled genotype graph patches.

[Maximum mean discrepancy](https://www.jmlr.org/papers/v13/gretton12a.html) provides a second distribution test. [Distributional precision and recall](https://papers.nips.cc/paper_files/paper/2018/hash/f7696a9b362ac5a51c3dc8f098b73923-Abstract.html) can later separate poor fidelity from poor coverage.

Chance C2ST performance does not prove biological realism. Weak features and low power can also yield chance results. Mutascape therefore pairs detectability with hidden-fitness, topology, calibration, and search-transfer tests.

## Distinct contribution to test

No reviewed project combines all of these parts in one workflow.

- Sparse DMS measurements supply single-mutant effects for a whole-protein oracle.
- Cross-protein empirical priors complete missing interactions.
- Posterior samples expose uncertainty in unseen sequence space.
- Held-out empirical landscapes test fitness and topology.
- Search rankings test whether the synthetic oracle supports valid method comparisons.

The paper should state this as a tested contribution. It should avoid an unqualified first-of-kind claim until the final search is complete.
