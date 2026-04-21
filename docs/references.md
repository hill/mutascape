# References

Canonical papers behind the mutascape design, organised by topic. Each entry is one-line on what the paper shows and how it enters the project.

## Thermodynamic / global epistasis

The theoretical foundation for wrapping the additive + pairwise fitness function in a saturating non-linearity. Central claim across this literature: most observed "higher-order epistasis" in protein DMS is not a genuine higher-order genetic interaction, but a by-product of an underlying approximately-additive stability model passed through a sigmoid-like mapping to fitness.

- **Tareen, A., Kooshkbaghi, M., Posfai, A., Ireland, W. T., McCandlish, D. M., & Kinney, J. B. (2022).** *MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect.* **Genome Biology** 23:98. https://doi.org/10.1186/s13059-022-02661-7
  - Useful rhetoric for the Mutascape model: DMS/MAVE readouts are indirect, nonlinear, and noisy. The paper's latent-phenotype framing supports separating the synthetic genotype-phenotype score from the thermodynamic/nonlinear response and explicit noise model.

- **Otwinowski, J. (2018).** *Biophysical Inference of Epistasis and the Effects of Mutations on Protein Stability and Function.* **Molecular Biology and Evolution** 35(10): 2345-2354. https://doi.org/10.1093/molbev/msy141
  - Fits fitness = g(Sum DDG_i) jointly with per-position DDG_i on GB1 singles + doubles. Recovers a sigmoidal g and shows that it explains most pairwise epistasis. Our `ThermodynamicSigmoid` is directly modelled on this formulation.

- **Otwinowski, J., McCandlish, D. M., & Plotkin, J. B. (2018).** *Inferring the shape of global epistasis.* **PNAS** 115(32): E7550-E7558. https://doi.org/10.1073/pnas.1804015115
  - Formalises "global epistasis": fitness = g(additive) where g is monotonic but otherwise unknown. Non-parametric (isotonic) fit from DMS data. Motivates the idea that a single scalar non-linearity can absorb a large fraction of observed high-order epistasis. We use a parametric sigmoid for simplicity; their isotonic approach is a natural future direction.

- **Sailer, Z. R., & Harms, M. J. (2017).** *Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps.* **Genetics** 205(3): 1079-1088. https://doi.org/10.1534/genetics.116.195214
  - Shows analytically that a non-linear transform of an additive landscape generates apparent higher-order Walsh variance. Quantifies how much of the order-3/4 Walsh variance in real DMS can be explained by a global non-linearity. This is the theoretical justification for why adding g to mutascape should raise its Walsh-3/4 fractions from zero without explicit higher-order couplings.

- **Husain, K., & Murugan, A. (2020).** *Physical Constraints on Epistasis.* **Molecular Biology and Evolution** 37(10): 2865-2874. https://doi.org/10.1093/molbev/msaa124
  - Modern statistical-physics perspective: generic physical constraints (stability thresholds, binding cooperativity) produce generic epistatic patterns. Argues that the "shape" of global epistasis is predictable from biophysics, not ad hoc fitting.

## Protein-specific DMS validations

- **Olson, C. A., Wu, N. C., & Sun, R. (2014).** *A Comprehensive Biophysical Description of Pairwise Epistasis throughout an Entire Protein Domain.* **Current Biology** 24(22): 2643-2651. https://doi.org/10.1016/j.cub.2014.09.072
  - All pairwise doubles across 55 GB1 positions (~535k variants). Demonstrates that most apparent pairwise epistasis is stability-mediated and can be captured by a non-linear transform of an additive stability model. The companion dataset to Wu 2016 for benchmarking any future higher-order coupling work.

- **Wu, N. C., Dai, L., Olson, C. A., Lloyd-Smith, J. O., & Sun, R. (2016).** *Adaptation in protein fitness landscapes is facilitated by indirect paths.* **eLife** 5:e16965. https://doi.org/10.7554/eLife.16965
  - The 4-position complete factorial GB1 DMS used in our validation notebook (`data/gb1/`).

- **Nisthal, A., Wang, C. Y., Ary, M. L., & Mayo, S. L. (2019).** *Protein stability engineering insights revealed by domain-wide comprehensive mutagenesis.* **PNAS** 116(33): 16367-16377. https://doi.org/10.1073/pnas.1903888116
  - Singles DDG measurements for GB1. Empirically grounds the claim that fitness in GB1 is stability-dominated and therefore well described by the thermodynamic non-linearity hypothesis. If GB1-specific validation of `ThermodynamicSigmoid` doesn't improve metrics, this is the reference to revisit for what physical variable IS driving fitness.

- **Sarkisyan, K. S., et al. (2016).** *Local fitness landscape of the green fluorescent protein.* **Nature** 533: 397-401. https://doi.org/10.1038/nature17995
  - The sparse GFP full-protein multi-mutant dataset used as the supplementary generalization check. It is not a complete factorial landscape.

## Epistasis theory and landscape statistics

- **Reidys, C. M., & Stadler, P. F. (2002).** *Combinatorial Landscapes.* **SIAM Review** 44(1): 3-54. https://www.jstor.org/stable/4148412
  - Formal landscape theory reference. The useful framing is that a landscape is a configuration space plus a fitness function plus an adjacency/accessibility structure. This supports describing Mutascape as a fully queryable genotype graph, not merely a table of scores.

- **Poelwijk, F. J., Socolich, M., & Ranganathan, R. (2019).** *Learning the pattern of epistasis linking genotype and phenotype in a protein.* **Nature Communications** 10: 4213. https://doi.org/10.1038/s41467-019-12130-8
  - Walsh-Hadamard variance decomposition on a complete factorial protein landscape. Useful rhetoric: complete maps reveal high-order epistasis and accessible-path topology, but exhaustive epistasis mapping becomes infeasible quickly; sparsity is a potential organizing principle. This is the reference for the `fourier.py` module in mutascape.

- **Poelwijk, F. J., Tanase-Nicola, S., Kiviet, D. J., & Tans, S. J. (2011).** *Reciprocal sign epistasis is a necessary condition for multi-peaked fitness landscapes.* **Journal of Theoretical Biology** 272(1): 141-144. https://doi.org/10.1016/j.jtbi.2010.12.015
  - Formalises reciprocal sign epistasis as the fundamental driver of multi-peakedness. The validation metric `reciprocal_sign_frac` in `accessible_paths` is grounded here.

- **Weinreich, D. M., Delaney, N. F., Depristo, M. A., & Hartl, D. L. (2006).** *Darwinian evolution can follow only very few mutational paths to fitter proteins.* **Science** 312(5770): 111-114. https://doi.org/10.1126/science.1123539
  - Accessible-path enumeration on a small complete factorial landscape. The reference for `accessible_paths` in `metrics.py`.

- **Szendro, I. G., Schenk, M. F., Franke, J., Krug, J., & de Visser, J. A. G. M. (2013).** *Quantitative analyses of empirical fitness landscapes.* **Journal of Statistical Mechanics: Theory and Experiment** P01005. https://doi.org/10.1088/1742-5468/2013/01/P01005
  - Roughness-to-slope ratio and other scalar landscape metrics. Reference for `roughness_slope_ratio` in `metrics.py`.

- **Kauffman, S., & Weinberger, E. (1989).** *The NK model of rugged fitness landscapes and its application to maturation of the immune response.* **Journal of Theoretical Biology** 141(2): 211-245. https://doi.org/10.1016/S0022-5193(89)80019-0
  - The NK landscape abstraction. Relevant as a baseline against which mutascape's structurally-informed couplings are compared.

- **Jones, T., & Forrest, S. (1995).** *Fitness Distance Correlation as a Measure of Problem Difficulty for Genetic Algorithms.* Proceedings of the Sixth International Conference on Genetic Algorithms, 184-192.
  - Fitness-distance correlation. Reference for `fitness_distance_correlation` in `metrics.py`.

## Protein stability as a fitness driver

- **Tokuriki, N., & Tawfik, D. S. (2009).** *Stability effects of mutations and protein evolvability.* **Current Opinion in Structural Biology** 19(5): 596-604. https://doi.org/10.1016/j.sbi.2009.08.003
  - Review: most mutations are destabilising; fitness is bounded by stability. Motivates the choice of a sigmoidal stability-fitness map.

- **Bershtein, S., Segal, M., Bekerman, R., Tokuriki, N., & Tawfik, D. S. (2006).** *Robustness-epistasis link shapes the fitness landscape of a randomly drifting protein.* **Nature** 444: 929-932. https://doi.org/10.1038/nature05385
  - Direct experimental demonstration that stability thresholds generate epistasis. The empirical counterpart to Sailer & Harms 2017's theoretical result.

- **Starr, T. N., & Thornton, J. W. (2016).** *Epistasis in protein evolution.* **Protein Science** 25(7): 1204-1218. https://doi.org/10.1002/pro.2897
  - Review of epistasis mechanisms in protein evolution. Useful for framing the distinction between "specific" (pairwise contact) and "nonspecific" (global/thermodynamic) epistasis.

## Synthetic landscape generators (prior art / positioning)

- **Sandhu, M., Chen, J. Z., Matthews, D. S., Spence, M. A., Pulsford, S. B., Gall, B., Kaczmarski, J. A., Nichols, J., Tokuriki, N., & Jackson, C. J. (2025).** *Computational and Experimental Exploration of Protein Fitness Landscapes: Navigating Smooth and Rugged Terrains.* **Biochemistry** 64: 1673-1684. https://doi.org/10.1021/acs.biochem.4c00673
  - Useful rhetoric for the introduction: smooth landscapes admit many accessible routes to high fitness, while rugged landscapes make evolution/search path-dependent through local optima and valleys. Also supports using ruggedness metrics such as roughness-to-slope, local maxima, sign epistasis, and graph/Fourier views.

- **Thomas, N., Agarwala, A., Belanger, D., et al. (2022).** *Tuned Fitness Landscapes for Benchmarking Model-Guided Protein Design.* bioRxiv 2022.10.28.514293. https://doi.org/10.1101/2022.10.28.514293
  - SLIP: landscapes derived entirely from MSA-fitted Potts models. Comparison point in the mutascape design doc. Our `PottsEpistasis` TODO entry would enable direct head-to-head comparison.

- **Sandhu, M., Mater, A. C., Matthews, D. S., Spence, M. A., Lenskiy, A. A., & Jackson, C. (2025).** *Investigating the determinants of performance in machine learning for protein fitness prediction.* **Protein Science** 34(8): e70235. https://doi.org/10.1002/pro.70235
  - Uses NK and empirical fitness landscapes to evaluate ML performance under interpolation, extrapolation, positional extrapolation, sparsity, ruggedness, and sequence-length changes. This is the current citation for the NK-style benchmark comparison in the paper.

## Protein language models for single-site effects

- **Lin, Z., et al. (2023).** *Evolutionary-scale prediction of atomic-level protein structure with a language model.* **Science** 379(6637): 1123-1130. https://doi.org/10.1126/science.ade2574
  - ESMFold. Used indirectly via `ContactMap.from_sequence` for structure prediction.

- **Rives, A., et al. (2021).** *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* **PNAS** 118(15): e2016239118. https://doi.org/10.1073/pnas.2016239118
  - ESM. Underlies `ESM2Scorer` for masked-marginal single-site effect prediction on proteins without DMS data.

- **Meier, J., et al. (2021).** *Language models enable zero-shot prediction of the effects of mutations on protein function.* **NeurIPS 2021**. https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html
  - Masked-marginal scoring methodology used by `ESM2Scorer`.
