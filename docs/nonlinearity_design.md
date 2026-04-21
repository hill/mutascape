# Design: thermodynamic non-linearity layer

> **Status note (2026-04-21).** The thermodynamic layer is now part of the frozen publication pipeline. The original validation plan at the end of this document is retained as historical context; the current GB1/TrpB reference results are in `docs/project_status.md`.

Companion to the implementation in `src/mutascape/nonlinearity.py` and its integration into `Landscape`. This document states the mathematical model, the parameter identification procedure, and the API.

## Motivation

See `docs/gb1_validation_analysis.md` for the historical empirical argument. In one line: under the original pairwise model a single scalar $\alpha$ controlled all epistatic behaviour, but real landscapes such as GB1 have multiple near-independent epistatic properties that cannot be jointly satisfied by a single dial over random Gaussian pairwise couplings. Adding a global non-linearity introduces a second mechanism that contributes apparent higher-order epistasis and path inaccessibility without requiring any additional data beyond what single-site DMS already provides.

## Model

The extended fitness function is

$$f(x) = g\!\Big(\underbrace{\sum_i \tilde{h}_i(x_i)}_{\text{preimage-additive}} + \alpha \sum_{i<j} C_{ij}\,J_{ij}(x_i, x_j) + \varepsilon\Big)$$

where $\tilde{h}_i(x_i) = g^{-1}(h_i(x_i))$ is the preimage of the observed single-site effect under the non-linearity $g$. Taking the preimage ensures that a single mutant's predicted fitness matches the DMS observation exactly:

$$f(\text{single mutation at } i \text{ to } a) = g(g^{-1}(h_i(a))) = h_i(a).$$

Multi-mutant predictions stack preimages in stability-equivalent units and then re-apply $g$, so that two mild singles can combine to cross the saturating cliff and produce a dramatic non-additive fitness change, or two beneficial singles can saturate against the ceiling. This is the Otwinowski (2018) / Olson (2014) formulation adapted to mutascape's structure-informed pairwise coupling setup.

## Sigmoid parameterisation

We use a 3-parameter logistic with floor $F$, ceiling $C$, and scale $s$:

$$g(z) = F + (C - F)\,\sigma\!\Big(\tfrac{z}{s} + \tau\Big), \qquad \sigma(u) = \tfrac{1}{1 + e^{-u}}$$

The shift $\tau$ is **not free**: we pin $g(0) = 0$ (wild-type fitness is zero by mutascape convention), which gives

$$\tau = \mathrm{logit}\!\Big(\tfrac{-F}{C - F}\Big).$$

The inverse is

$$g^{-1}(y) = s\,\Big(\mathrm{logit}\!\big(\tfrac{y - F}{C - F}\big) - \tau\Big).$$

Three parameters total ($F, C, s$), one of which is pinned and two of which are fit or set from data.

## Parameter identification from singles only

Given a set of observed single-site fitness effects $\{h_i(a)\}$ (either from DMS or a pLM):

1. **Floor and ceiling**: set from the empirical range with a small margin to keep observed values strictly interior (otherwise $g^{-1}$ diverges):
   $F = \min(h) - m\,(\max(h) - \min(h))$, $C = \max(h) + m\,(\max(h) - \min(h))$.
   Default $m = 0.05$.

2. **Scale $s$**: maximum-likelihood fit under the prior $g^{-1}(h) \sim \mathcal{N}(0, 1)$. The negative log-likelihood is

   $$-\sum_i \log \phi\big(g^{-1}(h_i)\big) - \sum_i \log\big|{\partial g^{-1}}/{\partial h}\big|_{h_i}$$

   where $\phi$ is the standard normal density and the Jacobian term comes from the change of variables. Minimised with `scipy.optimize.minimize_scalar` over a bounded interval.

The Gaussian prior on $g^{-1}(h)$ is convention, not physics: we are declaring that implicit stability effects look standard-normal. Different priors (Laplace for sparse effects, heavy-tailed for lethal-dominated proteins) can be swapped in without changing the API.

## Integration with `Landscape`

- New optional field `nonlinearity: NonLinearity = Identity()`.
- `__post_init__` precomputes `self._preimage[pos][aa] = g.inverse(h_i(a))` once, so per-evaluation cost is unchanged.
- `evaluate` / `evaluate_batch` sum preimages, add pairwise, add noise, then return `g(raw)`.
- `from_dms` and `from_plm` accept `nonlinearity` as `None` (use Identity), a constructed `NonLinearity` instance, or the literal string `"fit"` which triggers `ThermodynamicSigmoid.fit_from_singles` on the DMS data automatically.

Identity non-linearity is the default and reproduces existing behaviour bit-for-bit.

## Backwards compatibility and noise

- Existing calibration, metrics, and figures do not depend on the fitness function's internal structure; they consume only scalar fitness values. Nothing downstream changes.
- Measurement noise is added pre-non-linearity so that noise has its natural (pre-sigmoid, stability-scale) statistics. This matches the biophysical picture in which noise is experimental and adds before the folding readout saturates. If this proves wrong empirically, we can flip noise to post-sigmoid with a single-line change.

## Historical validation plan

The following was the original validation plan. The unit tests and GB1/TrpB reruns are now complete; the realized headline model uses fitted censored nonlinearity plus `noise_sigma = 0.3`, with random pairwise couplings disabled.

1. Unit-test: `Identity` landscape equals non-g landscape. `ThermodynamicSigmoid` forward/inverse round-trip to machine precision. Fit recovers a known $s$ on synthetic data.
2. GB1 rerun: re-execute `notebooks/validation/gb1_validation.ipynb` with the sigmoid fit from the 76 GB1 singles. Expected changes (from `docs/gb1_validation_analysis.md`):
   - Additive correlation at depth 2 drops from 0.89 toward 0.70 or below.
   - Walsh order-3 fraction rises from ~0 to at least 0.10.
   - Walsh order-4 fraction rises to at least 0.02.
   - Local-optima count rises from 5 to at least 30.
   - Accessible-path fraction drops from 0.77 to 0.55 or below.
   - Depth-2 deviation-distribution spike at zero disappears.
3. Recalibrate $\alpha$ jointly with the fitted sigmoid and compare against the Identity baseline. A falsifying outcome is no movement on any of the held-out metrics, which would redirect the project to empirical pairwise $J$ or higher-order couplings (section 5 of the analysis doc).

## Future extensions

- **Isotonic $g$** (Otwinowski 2018 non-parametric). Drop-in replacement for `ThermodynamicSigmoid` using a monotonic spline fitted from singles. Better when the true $g$ is not sigmoidal.
- **Joint $g + J_{ij}$ fit from a small multi-mutant panel.** When the user has a modest number of doubles (Option B in `gb1_validation_analysis.md`), jointly optimise $g$ parameters and $J_{ij}$ values instead of sampling $J$ from a prior.
- **pLM-informed prior.** If singles are from a pLM rather than DMS, the implicit ΔΔG prior may differ from standard-normal. Allow the prior distribution to be specified at fit time.
