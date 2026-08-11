"""Make-or-break probe: can a heat-kernel spectral field (no texture, no iid-eta)
reproduce the real GB1/TrpB Walsh spectrum AND local-optima count?

Latent landscape z(x) = additive DMS preimage (order 1, fixed)
                        + theta * structured spectral field (orders >= 2),
where the field's per-order variance follows lambda_k = exp(-t*k). Fitness is
g(z) with the existing fitted censored global-epistasis sigmoid. No readout
texture, no iid Gaussian noise carrying the spectrum. Sweep (t, theta) and report
the realised Walsh spectrum, Walsh-L1, fitness-W1, and emergent optima count.

Throwaway diagnostic, not part of the package.
"""

from itertools import combinations

import numpy as np

from mutascape.validation.comparisons import fitness_wasserstein
from mutascape.validation.fourier import (
    fitness_tensor_from_real,
    observed_alphabets,
    walsh_decompose,
)
from mutascape.validation.reference_configs import GB1_HEADLINE, TRPB_HEADLINE


def additive_preimage_tensor(landscape, positions, alphabets):
    """Order-1 latent field from the DMS preimage effects g^-1(h_i)."""
    shape = tuple(len(alphabets[p]) for p in positions)
    z = np.zeros(shape, dtype=np.float64)
    pre = landscape._preimage_effects
    for ai, p in enumerate(positions):
        eff = pre.get(p, {})
        vec = np.array([eff.get(aa, 0.0) for aa in alphabets[p]], dtype=np.float64)
        z = z + vec.reshape([-1 if d == ai else 1 for d in range(len(positions))])
    return z


def spectral_field(shape, t, rng):
    """Structured Gaussian field, orders 2..L, per-order variance ~ exp(-t*k).

    Each order-k component is a sum of pure-interaction (axis-centred) random
    tensors over every k-subset, normalised to unit variance, then weighted by
    exp(-t*k). Returns a unit-variance field (caller scales by theta*additive_rms).
    """
    L = len(shape)
    field = np.zeros(shape, dtype=np.float64)
    for k in range(2, L + 1):
        part = np.zeros(shape, dtype=np.float64)
        for S in combinations(range(L), k):
            sub_shape = tuple(shape[a] for a in S)
            r = rng.normal(size=sub_shape)
            for axi in range(k):  # centre each S-axis -> pure order-k interaction
                r = r - r.mean(axis=axi, keepdims=True)
            embed = r.reshape([shape[a] if a in S else 1 for a in range(L)])
            part = part + embed  # broadcast-accumulate to full shape
        sd = part.std()
        if sd > 0:
            part = part / sd
        field = field + np.exp(-t * k) * part
    sd = field.std()
    return field / sd if sd > 0 else field


def fast_n_optima(tensor):
    """Count strict local maxima over the Hamming-1 neighbourhood (vectorised).

    Handles ties on the censored floor plateau correctly: a cell is a peak iff
    it is strictly greater than the max of all OTHER cells along every axis, so
    tied maxima (equal floor values) are not counted. Matches
    metrics.local_optima_from_tensor on a fully-evaluated tensor.
    """
    is_peak = np.ones(tensor.shape, dtype=bool)
    for ax in range(tensor.ndim):
        m1 = tensor.max(axis=ax, keepdims=True)
        count_max = (tensor == m1).sum(axis=ax, keepdims=True)
        is_unique_max = (tensor == m1) & (count_max == 1)
        masked = np.where(tensor == m1, -np.inf, tensor)
        m2 = masked.max(axis=ax, keepdims=True)
        others_max = np.where(is_unique_max, m2, m1)  # exclude only self
        is_peak &= tensor > others_max
    return int(is_peak.sum())


def run(config, name):
    real_df = config.load_real_dataframe()
    positions = list(config.positions)
    alphabets = observed_alphabets(real_df, positions, config.wt_aa_at_position)
    real_tensor, _ = fitness_tensor_from_real(
        real_df, positions, config.wt_aa_at_position, alphabets
    )
    real_filled = np.where(np.isnan(real_tensor), np.nanmean(real_tensor), real_tensor)
    real_walsh = walsh_decompose(real_tensor)
    real_opt = fast_n_optima(real_filled)

    landscape = config.build_landscape()
    g = landscape.nonlinearity
    additive = additive_preimage_tensor(landscape, positions, alphabets)
    add_rms = additive.std()
    shape = additive.shape

    rwf = tuple(round(real_walsh.get(o, 0.0), 3) for o in range(1, len(shape) + 1))
    print(f"\n=== {name} ===   real Walsh={rwf}  real n_opt={real_opt}")
    print(f"  {'t':>4} {'theta':>6} | {'Walsh(synth)':>26} | {'L1':>6} {'W1':>6} {'n_opt':>6}")

    rng = np.random.default_rng(0)
    best = None
    for t in (0.3, 0.5, 0.8, 1.2, 1.8):
        field = spectral_field(shape, t, rng)
        for theta in (1.0, 1.5, 2.0, 3.0, 4.0):
            z = additive + theta * add_rms * field
            fit = np.asarray(g.forward(z), dtype=np.float64)
            sw = walsh_decompose(fit)
            l1 = sum(abs(real_walsh.get(o, 0.0) - sw.get(o, 0.0)) for o in real_walsh)
            w1 = fitness_wasserstein(real_tensor, fit)
            nopt = fast_n_optima(fit)
            swf = tuple(round(sw.get(o, 0.0), 3) for o in range(1, len(shape) + 1))
            flag = ""
            if best is None or l1 < best[0]:
                best = (l1, t, theta, swf, w1, nopt)
                flag = "  <- best L1"
            print(f"  {t:>4} {theta:>6} | {str(swf):>26} | {l1:>6.3f} {w1:>6.3f} {nopt:>6}{flag}")
    print(f"  BEST spectrum match: t={best[1]} theta={best[2]} -> Walsh={best[3]} "
          f"L1={best[0]:.3f} W1={best[4]:.3f} n_opt={best[5]} (real {real_opt})")
    return best, real_opt


gb1_best, gb1_real = run(GB1_HEADLINE, "GB1")
trpb_best, trpb_real = run(TRPB_HEADLINE, "TrpB")

print("\n=== make-or-break: does ONE (t, theta) work for both proteins? ===")
print(f"  GB1  best t={gb1_best[1]} theta={gb1_best[2]}  (n_opt {gb1_best[5]} vs real {gb1_real})")
print(f"  TrpB best t={trpb_best[1]} theta={trpb_best[2]}  (n_opt {trpb_best[5]} vs real {trpb_real})")
