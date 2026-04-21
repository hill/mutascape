"""Synthetic combinatorial protein fitness landscapes for MLDE/ALDE benchmarks.

The publication reference pipeline is exposed through frozen validation configs:

>>> from mutascape.validation.reference_configs import GB1_HEADLINE, evaluate_reference_config
>>> result = evaluate_reference_config(GB1_HEADLINE)
>>> result.walsh_l1

Lower-level APIs still support custom landscapes from DMS, structure-derived
contact maps, optional pairwise couplings, ESM2 single-site scoring, and fitted
thermodynamic nonlinearities. Those features are useful for larger benchmark
candidates, but the validated headline GB1/TrpB configs keep pairwise couplings
off by setting ``sigma = 0``.
"""

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    AMINO_ACIDS,
    CouplingTensor,
    EpistasisModel,
    MSACouplingsEpistasis,
    RandomEpistasis,
    SignConsistentEpistasis,
)
from mutascape.landscape import Landscape, Variant
from mutascape.mutant import Mutant
from mutascape.nonlinearity import Identity, NonLinearity, ThermodynamicSigmoid
from mutascape.plotting import (
    plot_alpha_sweep,
    plot_contact_map,
    plot_epistasis_scatter,
    plot_fitness_distribution,
)
from mutascape.scoring import ESM2Scorer, PLMScorer

try:
    from mutascape.interactive import morphing_landscape
except ModuleNotFoundError as exc:
    if exc.name != "plotly":
        raise
    _plotly_exc = exc

    def morphing_landscape(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "morphing_landscape requires the optional Plotly dependency."
        ) from _plotly_exc

__all__ = [
    "AMINO_ACIDS",
    "ContactMap",
    "CouplingTensor",
    "ESM2Scorer",
    "EpistasisModel",
    "Identity",
    "Landscape",
    "MSACouplingsEpistasis",
    "Mutant",
    "NonLinearity",
    "PLMScorer",
    "RandomEpistasis",
    "SignConsistentEpistasis",
    "ThermodynamicSigmoid",
    "Variant",
    "morphing_landscape",
    "plot_alpha_sweep",
    "plot_contact_map",
    "plot_epistasis_scatter",
    "plot_fitness_distribution",
]

__version__ = "0.1.0"
