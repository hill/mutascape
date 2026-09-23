# How prior work validates synthetic fitness landscapes

Full-text review of nine works. The question is narrow. When a paper builds or
borrows a synthetic protein fitness landscape, what evidence does it give that
the landscape is fit for its stated purpose?

This document covers validation evidence only. `literature_review_v1.md` covers
positioning against benchmark suites.

## Summary

| Work | Synthetic landscape | Validation evidence |
| --- | --- | --- |
| SLIP (Thomas 2022) | Tuned Potts, five PDB targets | Optimisation difficulty alone. Spearman gap between a CNN and Ridge |
| Ehrlich (Stanton 2024) | Closed-form spaced motifs | Rejects realism as a requirement. Argues for shared geometry |
| FLEXS and AdaLead (Sinai 2020) | ViennaRNA and PyRosetta oracles | Oracle consistency far from the seed, not resemblance to measurement |
| Sandhu 2025 | NK, N=6, K from 0 to 5 | Qualitative agreement of model rankings with one measured landscape |
| Rough Mount Fuji (Neidhart 2014) | Additive plus uncorrelated noise | Fits the ruggedness parameter so model statistics match measurement |
| GraphFLA (2025) | NK, only as a test harness | Replicates the published claims of 61 source papers |
| ORBIT (2026) | ORBIT-Synth, order-controlled | Recovers a planted interaction order before touching GB1 |
| SSMuLA (2025) | None | 16 measured landscapes, so no generator to validate |
| CombinGym (2026) | None | 14 measured landscapes, plus wet-lab RhlA variants |

## Five ways to assess a synthetic landscape

**Difficulty calibration.** Tune the landscape until it separates methods. SLIP
and Ehrlich both do this. Neither compares any landscape statistic to a
measurement.

**Ground-truth recovery.** Plant a known structure, then check the paper's own
measuring tool recovers it. GraphFLA and ORBIT both do this. The synthetic
landscape is a unit test, not a benchmark.

**Statistic matching.** Fit one model parameter so a scalar landscape statistic
matches measurement. The Rough Mount Fuji and NK theory lineage does this on
landscapes of six to eight binary loci.

**Trend concordance.** Check that conclusions drawn on the synthetic landscape
hold on a measured one. Sandhu 2025 does this qualitatively against GB1.

**Refusal.** Use only measured landscapes. SSMuLA and CombinGym both take this
route, and SSMuLA says why.

## SLIP

Thomas and colleagues fit Potts models to MSAs, then rescale the single and pair
terms so the epistatic horizon lands on a target value. Assessment reduces to one
number.

The protocol tunes the horizon to powers of two from 2 to 1024. It samples 5,000
training sequences at one to three mutations. It trains Ridge and a three-layer
one-dimensional CNN. It tests on 200 variants at distance six, built from the top
20 single mutants. It reports the Spearman gap over 20 replicates.

Untuned landscapes are degenerate. On 3er7 the horizon reaches 166 against a
sequence length of 118. The gap is negative on four of five targets before
tuning. Tuning lifts it by 0.2 to 0.4 on three targets. On 3bfo the CNN never
wins.

Realism enters once. The authors screened 748 Potts models by contact-prediction
accuracy and picked five by hand. No landscape statistic is ever compared to a
measured landscape. The discussion admits the horizon is optimisation-motivated
and that a variance-based crossover may suit regression better.

## Ehrlich functions

The clearest statement of the opposing position. Stanton and colleagues list four
benchmark criteria, low cost, well-characterised solutions, non-trivial
difficulty, and a stylised similarity to real work. They then argue that a test
function need not correlate with the application as long as the geometry is
shared.

Assessment is simple regret and the feasible fraction of the population, over 32
trials with a genetic algorithm. A difficulty sweep varies motif count, motif
length, sequence length and an objective quantisation parameter. Quantisation
dominates.

They attack empirical surrogates head on. A model trained on measurements is
reliable only near its data, and blind optimisation of it finds spurious optima.
They implemented Rough Mount Fuji as a control and dropped it, because a genetic
algorithm solves it at once.

This paper is the strongest argument against mutascape's premise. Answer it
directly in the introduction.

## FLEXS and AdaLead

The harness uses measured transcription-factor binding over all 65,536 DNA
8-mers, ViennaRNA duplex energies, and PyRosetta on the 66-residue antifreeze
protein 3MSI.

The stated reason for these choices is not resemblance. The authors want a ground
truth whose quality holds as the search drifts away from the seed. A model
trained on measurements fails that test, and they say so.

Model quality becomes a knob. The noisy oracle interpolates between the true
landscape and exponential noise, weighted by distance to the nearest measured
neighbour. Three criteria follow, optimisation quality, behaviour under a
useless model, and improvement as the model improves.

On small landscapes they enumerate every local peak by brute force and count how
many each algorithm finds. AdaLead finds 12.8 of 353 peaks above 0.75 on a
14-mer RNA landscape, against 9.2 for DyNA-PPO and 1.0 for CbAS.

## Sandhu 2025

The nearest published precedent to mutascape's transfer claim, and it is thin.

The NK landscapes use N=6, K from 0 to 5, and a six-letter alphabet. Per-site
contributions come from a uniform distribution on the unit interval. The paper
says the alphabet is arbitrary and that NK has no biophysical grounding. The
replicate count is inconsistent between the results and the methods, four in one
place and eight in the other.

Validation is a qualitative concordance check against one measured landscape, the
four-site GB1 scan of Wu 2016. Three trends hold in both. Extrapolation accuracy
falls with distance. Performance falls with sparser sampling. Decision trees
tolerate sequence padding where neural networks do not.

No summary statistic of the NK landscapes is compared to GB1. Ruggedness is set
through K and never measured. The paper surveys local optima counts, graph
Dirichlet energy, Fourier decomposition and the roughness-to-slope ratio, then
computes none of them.

The authors turn the gap into a tool. Because NK assigns fitness at random per
character, no positional pattern exists to extrapolate. They use NK as a negative
control that separates genuine positional extrapolation from luck.

## Rough Mount Fuji

The one lineage that matches synthetic statistics to measured ones. Fitness is an
additive gradient away from a reference sequence plus uncorrelated noise.
Ruggedness is the ratio of the additive coefficient to the noise standard
deviation.

Neidhart, Szendro and Krug derive closed forms for the local maximum count, the
global peak position and the fitness correlation function. They then fit the
ratio to data.

Franke 2011 and Szendro 2013 both recover a ratio near 0.21 to 0.25 for the
eight-locus *Aspergillus niger* landscape, by two independent routes. Neidhart
matches the number of second-step beneficial mutations to the Miller 2011 phage
experiment, giving a ratio between 0.71 and 1.69. Szendro 2013 uses the model to
interpolate ruggedness measures across ten measured landscapes.

The tail of the noise distribution matters as much as the ruggedness ratio. Both
parameters are needed to describe a measured landscape.

## GraphFLA and ORBIT

Both use synthetic landscapes as ground truth for their own instrument, never as
a benchmark that must look real.

GraphFLA validates by reproducing the qualitative and quantitative claims of all
61 papers behind its 155 complete landscapes. It then stress-tests its 20
features on a synthetic NK reference at n=15 and k=7, under incompleteness,
Gaussian noise to 0.2 standard deviations, and biased sampling.

ORBIT builds pure-order Walsh controls with exact fourth-order parity and a
cooperativity sweep. It checks that its diagnostics recover the planted order,
then applies them to GB1 and FLIP.

## SSMuLA

No synthetic landscape appears anywhere in the paper. Sixteen measured landscapes
cover six protein systems, and twelve carry at least one per cent active
variants.

Six attributes split into two families. Fitness statistics ignore sequence, so
percentage active, Cauchy peak location, kurtosis and KDE peak count. Ruggedness
needs the sequence-to-fitness map, so local optima count and percentage of
non-magnitude pairwise epistasis.

A local optimum beats every active single-substitution neighbour. Sign and
reciprocal sign epistasis merge into non-magnitude, the classes a greedy uphill
walk cannot navigate.

Budgets run from 120 to 2,016 samples, with 480 as the headline. Every machine
learning condition averages over 50 replicates. Method gain over single-step
directed evolution is then Spearman-correlated against each of the six
attributes. No regression model appears. The best result is a 3.5-fold gain for
focused-training MLDE on TrpB3E, one of the least navigable landscapes.

The paper motivates the design by pointing at frequent misalignment between
theoretical landscape modelling and experimental work. Read that as a direct shot
at synthetic generators.

## CombinGym

Also purely empirical. Fourteen landscapes, nine proteins, over 400,000 measured
variants. Hierarchical splits run from zero-vs-rest through three-vs-rest.
Metrics are Spearman rank correlation and NDCG at the top one per cent, which
agree only moderately at Pearson 0.623.

Epistasis motivates the work but no ruggedness statistic is reported. Wet-lab
RhlA variants replace any synthetic validation.

## Where the mutascape metrics come from

**Walsh spectrum.** Poelwijk, Socolich and Ranganathan 2019 supplies the
Walsh-Hadamard variance decomposition. `docs/references.md` cites it as the
reference for `fourier.py`. Sailer and Harms 2017 supplies the theory that a
global-epistasis function on an additive landscape manufactures apparent
third-order and fourth-order variance. The scalar `walsh_L1` in
`src/mutascape/validation/fourier.py` is mutascape's own construction. It carries
no citation and matches no published statistic found in this review.

**Method-ranking similarity.** SSMuLA supplied the 16 development landscapes and
the budget scale. The protocol in `docs/benchmark_protocol.md` is original and
currently uncited. It has three parts, the median Kendall coefficient across
landscapes, the reliable pairwise winner rate over 1,000 bootstraps, and the
split-half reliability floor at the fifth percentile over 500 splits.

## What is unclaimed

No reviewed work asks whether a generated landscape preserves the measured
ranking of search methods. No reviewed work defines a reliability floor that says
when the measured ranking is stable enough to transfer at all.

Sandhu 2025 is the nearest precedent and differs on three axes. It is qualitative
rather than a rank coefficient. It uses one measured landscape rather than 23. It
compares supervised architectures rather than search methods under a fixed query
budget.

## Actions

1. State the transfer claim as a specific negative, so no prior generator is
   validated by rank transfer against measured landscapes.
2. Cite Sandhu 2025 as the qualitative antecedent, not only as the NK baseline.
   `docs/references.md` currently undersells it.
3. Cite SSMuLA in `docs/benchmark_protocol.md` for the development panel and the
   budget scale. The document currently reads as though the whole protocol is
   original.
4. Answer Ehrlich in the introduction. That paper argues a benchmark needs shared
   geometry and nothing more, which is the strongest case against conditioning on
   measured data.
5. Report the Walsh and topology diagnostics as statistic matching in the Rough
   Mount Fuji sense, and keep them separate from the transfer claim.

## Sources

- Thomas, N., Agarwala, A., Belanger, D., Song, Y. S., & Colwell, L. (2022). *Tuned Fitness Landscapes for Benchmarking Model-Guided Protein Design.* bioRxiv 2022.10.28.514293. <https://doi.org/10.1101/2022.10.28.514293>
- Stanton, S., Alberstein, R., Frey, N., Watkins, A., & Cho, K. (2024). *Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms.* arXiv 2407.00236. <https://arxiv.org/abs/2407.00236>
- Sinai, S., Wang, R., Whatley, A., Slocum, S., Locane, E., & Kelsic, E. D. (2020). *AdaLead, an adaptive greedy search algorithm for sequence design.* arXiv 2010.02141. <https://arxiv.org/abs/2010.02141>
- Sandhu, M., Mater, A. C., Matthews, D. S., Spence, M. A., Lenskiy, A. A., & Jackson, C. (2025). *Investigating the determinants of performance in machine learning for protein fitness prediction.* Protein Science 34(8), e70235. <https://doi.org/10.1002/pro.70235>
- Neidhart, J., Szendro, I. G., & Krug, J. (2014). *Adaptation in tunably rugged fitness landscapes, the Rough Mount Fuji Model.* Genetics 198(2), 699-721. <https://arxiv.org/abs/1402.3065>
- Wittmann, B. J., et al. (2025). *SSMuLA.* Cell Systems. <https://doi.org/10.1016/j.cels.2025.101387>. Preprint at <https://doi.org/10.1101/2024.10.24.619774>
- CombinGym (2026). bioRxiv 2026.03.24.714074. <https://www.biorxiv.org/content/10.64898/2026.03.24.714074v1.full>
- GraphFLA (2025). NeurIPS Datasets and Benchmarks. arXiv 2510.24826.
- ORBIT (2026). arXiv 2608.24953.
