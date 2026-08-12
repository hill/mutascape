"""Global-epistasis function layer for the fitness function.

Applied as fitness = g(sum_i g^{-1}(h_i) + alpha * sum_{i<j} C_ij J_ij + noise),
so that single-mutant predictions match observed DMS exactly (preimage
cancels the outer g) while multi-mutant predictions stack in stability-
equivalent units and pass through the global-epistasis function. This
converts an additive + pairwise model into one whose apparent higher-
order epistasis, local-optima density, and path inaccessibility match
empirical protein landscapes far more closely.

This is the "global epistasis" formulation of Otwinowski, McCandlish &
Plotkin (2018, PNAS) and the parametric biophysical fit of Otwinowski
(2018, MBE). Sailer & Harms (2017, Genetics) shows analytically why a
global-epistasis function over an additive base produces apparent higher-
order epistasis. The sigmoidal form is motivated by stability-limited
fitness (two-state folding; Tokuriki & Tawfik 2009).

See docs/nonlinearity_design.md for the model derivation and
docs/references.md for the broader literature.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import norm


@runtime_checkable
class NonLinearity(Protocol):
    """A monotonic invertible map applied to raw fitness values."""

    def forward(self, z: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Apply the global-epistasis function. Maps stability-space to fitness-space."""
        ...

    def inverse(self, y: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Apply the inverse. Maps observed fitness back to stability-space."""
        ...


@dataclass(frozen=True)
class Identity:
    """Passthrough. Recovers the original pre-readout Landscape behaviour."""

    def forward(self, z):
        return z

    def inverse(self, y):
        return y


@dataclass(frozen=True)
class GlobalEpistasisSigmoid:
    """Global-epistasis sigmoid: 3-parameter logistic with WT pinned at fitness 0.

    Parametric form of the global-epistasis function from
    Otwinowski, McCandlish & Plotkin (2018, PNAS) and Otwinowski (2018,
    MBE). See module docstring for citations and motivation.

    g(z) = floor + (ceiling - floor) * sigmoid(z/scale + tau)

    where tau is chosen so g(0) = 0, i.e. tau = logit(-floor / (ceiling - floor)).
    Only (floor, ceiling, scale) are free; tau is derived.

    An optional `floor_hard` clips the output at a hard lower bound,
    producing a censored sigmoid. This models the dead-variant plateau
    seen in real DMS data (e.g. GB1 has 77% of combinatorial variants at
    exactly the assay detection limit). When set, `g(z) = max(smooth(z),
    floor_hard)`; observed singles at or above `floor_hard` still satisfy
    `g(g^{-1}(h)) = h` so the preimage cache works unchanged.
    """

    floor: float
    ceiling: float
    scale: float
    floor_hard: float | None = None

    def __post_init__(self) -> None:
        if self.ceiling <= self.floor:
            raise ValueError(f"ceiling ({self.ceiling}) must exceed floor ({self.floor})")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not (self.floor < 0 < self.ceiling):
            raise ValueError(
                f"WT fitness 0 must lie strictly between floor ({self.floor}) "
                f"and ceiling ({self.ceiling}); otherwise tau is undefined"
            )
        if self.floor_hard is not None and self.floor_hard >= 0:
            raise ValueError(
                f"floor_hard ({self.floor_hard}) must be < 0 so WT (fitness 0) is not clipped"
            )

    @property
    def tau(self) -> float:
        """Logit shift that pins g(0) = 0."""
        q = -self.floor / (self.ceiling - self.floor)
        return float(np.log(q / (1.0 - q)))

    def forward(self, z):
        z = np.asarray(z, dtype=np.float64)
        u = z / self.scale + self.tau
        out = self.floor + (self.ceiling - self.floor) * expit(u)
        if self.floor_hard is not None:
            out = np.maximum(out, self.floor_hard)
        return out if out.ndim else float(out)

    def inverse(self, y):
        y = np.asarray(y, dtype=np.float64)
        # Clip into the open interval so logit is defined.
        span = self.ceiling - self.floor
        eps = 1e-9 * span
        norm_y = np.clip((y - self.floor) / span, eps / span, 1.0 - eps / span)
        logit = np.log(norm_y / (1.0 - norm_y))
        out = self.scale * (logit - self.tau)
        return out if out.ndim else float(out)

    @classmethod
    def fit_from_singles(
        cls,
        single_site_effects: dict[int, dict[str, float]],
        margin: float = 0.05,
        prior_std: float = 1.0,
        floor: float | None = None,
        ceiling: float | None = None,
        combinatorial_upside: float = 1.0,
    ) -> "GlobalEpistasisSigmoid":
        """Fit floor, ceiling, and scale from observed single-site fitness effects.

        Floor and ceiling are set to the observed range padded by `margin` of
        the range on each side, so all observed values sit strictly inside the
        sigmoid's support. Scale is chosen by maximum likelihood under the
        prior g^{-1}(h) ~ Normal(0, prior_std^2), with the standard change-of-
        variables Jacobian.

        Parameters
        ----------
        floor, ceiling : float, optional
            Override the auto-derived floor or ceiling with a specific value.
            Useful when combinatorial DMS extends the fitness range beyond
            what singles alone reveal. For GB1 (singles range [-1, 2.9],
            combinatorials reach 7.8), passing ceiling=8 matches the real
            landscape ceiling and dramatically improves held-out metrics.
        combinatorial_upside : float, default 1.0
            Multiplier applied to the singles-derived ceiling upside (hi - 0)
            when `ceiling` is not explicitly given. 1.0 matches singles-only
            range; 3.0 is a reasonable prior for stability-limited proteins
            where stabilising combinations can extend beyond any single's
            effect. Typical range 1-5.

        Raises ValueError if the observed range does not straddle zero (WT),
        in which case the global-epistasis parameterisation is ill-defined.
        """
        values = np.array(
            [v for effects in single_site_effects.values() for v in effects.values()],
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError("single_site_effects contains no values")

        lo, hi = float(values.min()), float(values.max())
        if not (lo < 0 < hi):
            raise ValueError(
                f"Observed fitness range [{lo}, {hi}] does not straddle WT (0). "
                "GlobalEpistasisSigmoid requires both positive and negative effects; "
                "check that fitness is defined relative to WT."
            )
        span = hi - lo
        if floor is None:
            floor = lo - margin * span
        if ceiling is None:
            # Extend the positive side by combinatorial_upside to model the
            # expected fitness headroom from stacking stabilising mutations.
            ceiling = hi * combinatorial_upside + margin * span

        q = -floor / (ceiling - floor)
        tau = np.log(q / (1.0 - q))
        full_span = ceiling - floor
        norm_y = (values - floor) / full_span
        logit_y = np.log(norm_y / (1.0 - norm_y))

        def neg_log_lik(scale: float) -> float:
            if scale <= 0:
                return np.inf
            deltaG = scale * (logit_y - tau)
            # Jacobian: d(inverse)/dy = scale * full_span / ((y - floor)(ceiling - y))
            log_jac = (
                np.log(scale)
                + np.log(full_span)
                - np.log(values - floor)
                - np.log(ceiling - values)
            )
            log_prior = norm.logpdf(deltaG, loc=0.0, scale=prior_std)
            return float(-(log_prior + log_jac).sum())

        result = minimize_scalar(neg_log_lik, bounds=(1e-3, 100.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"sigmoid scale fit failed: {result.message}")

        return cls(floor=float(floor), ceiling=float(ceiling), scale=float(result.x))


def resolve(nonlinearity: NonLinearity | str | None, single_site_effects) -> NonLinearity:
    """Normalise a user-supplied nonlinearity argument into a concrete instance.

    Accepts None (-> Identity), a NonLinearity instance (returned as-is), or
    the literal string "fit" which triggers GlobalEpistasisSigmoid.fit_from_singles.
    """
    if nonlinearity is None:
        return Identity()
    if isinstance(nonlinearity, str):
        if nonlinearity == "fit":
            return GlobalEpistasisSigmoid.fit_from_singles(single_site_effects)
        raise ValueError(
            f"Unknown nonlinearity string {nonlinearity!r}; use 'fit' or pass an instance"
        )
    return nonlinearity
