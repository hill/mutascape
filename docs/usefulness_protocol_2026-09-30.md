# Mutascape usefulness experiments

The larger search study did not establish reliable method selection across proteins.
This exploratory experiment tests two changes before any manuscript revision.
It uses CreiLOV, eqFP611, TrpB4, Pab1 and Ube4b.
These datasets have appeared in earlier analyses, so this is not independent confirmation.

## Questions

Can varied interaction assumptions expose failures that one singles-only landscape misses?
Does conditioning random pair effects on measured combinations improve distribution and search agreement?

The existing combination model estimates aggregate mutation effects and a fitness distribution.
Its individual pair effects remain random.
The proposed model also conditions that shared pair field on the observed combination residuals.
Its covariance accounts for the WT correction shared by pair coefficients.
It preserves WT and every supplied single exactly.
It selects observation variance from four fixed ratios through the calibration likelihood.
No evaluation fitness enters this calculation.

## Fixed design

- Use 1,000 evaluation combinations per protein and all supplied WT and singles.
- Reserve the existing complete three-site subcubes where available.
- Keep evaluation combinations and every subcube corner outside each calibration sample.
- Collect three independent calibration samples with nested sizes of 48 and 192 combinations.
- Use three landscape seeds for fidelity metrics.
- Compare default Mutascape, linear pair effects, stronger sigmoid pair effects and the additive control.
- Compare the current combination model with the conditioned pair model at both calibration sizes.
- Include the deterministic conditioned mean at 192 measurements as a regression control.
- Fit quantile maps over the same declared candidate library for both combination models.
- Use at most 8,192 uniformly sampled, unlabelled library combinations to estimate these maps.
- Report grouped sequence-aware C2ST, scaled Wasserstein distance, Spearman correlation and local Walsh distance.

Pab1 and Ube4b contain sparse libraries with too few complete three-site subcubes for the saved spectral design.
They receive no spectral score.
Their large Cartesian libraries also exceed the public combination model's enumeration limit.
The research adapter therefore calibrates to the declared candidate library, rather than its full Cartesian product.
This change applies equally to both combination models.

## Search test

All twelve existing methods search the same 1,000 withheld combinations, plus the known WT and singles.
The primary budget is 96 new evaluations.
Measured fitness has 30 search repeats.
Each synthetic condition has three independent samples and ten repeats per sample.
Measured and synthetic searches use separate seed streams.
Each combination model uses one landscape seed from each of the three calibration samples.
This measures their combined variation without separating the two sources.

The search comparison includes default, additive, linear and stronger singles-only scenarios.
It also includes both combination models with 192 measurements and the deterministic conditioned mean.
All calibration measurements remain outside the search library.
They are an extra data cost and are not part of the 96 search evaluations.
This tests benchmark resemblance, rather than the efficiency of an end-to-end laboratory campaign.

The primary search metric is measured regret after choosing the method with lowest synthetic mean regret.
Also report Kendall correlation and regret relative to the best measured method.
An equal-weight singles-only scenario ensemble supplies one further selector.
The fixed selector averages method ranks from the other four measured proteins.
It excludes all measured outcomes from the target protein.
Bootstrap intervals repeat selection and resample proteins, landscape samples and paired search repeats.

No model becomes the default because it wins one metric or one protein.
Report every condition, including failures.
Any positive result needs a frozen follow-up on new proteins before a general transfer claim.

## Execution amendment

The first search check stopped because annealing spent only two of its 96 evaluations.
The restricted library contains isolated genotypes, so local search can exhaust its available neighbours.
Before completing the comparison, the runner added uniform random sampling for any unspent evaluations.
Every method and condition receives this same fallback.
The result files record the number of fallback evaluations in each campaign.
The initial manifest remains available alongside the amended hashes.
No fitting code, candidate, metric or dataset changed after this failure.
