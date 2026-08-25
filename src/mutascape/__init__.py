"""Task-validated synthetic protein fitness oracles for search benchmarks."""

from mutascape.api import LandscapeGenerator, Oracle, from_dms, load_oracle
from mutascape.campaign import (
    DEFAULT_BATCH_SIZE,
    Campaign,
    CampaignResult,
    SearchMethod,
    run_campaign,
)
from mutascape.completion import (
    CompletedLandscape,
    DMSPairwiseLandscape,
    LandscapePrior,
    PosteriorLandscape,
)
from mutascape.contacts import ContactMap
from mutascape.domain import MutationDomain
from mutascape.epistasis import (
    AMINO_ACIDS,
    CouplingTensor,
    EpistasisModel,
    MSACouplingsEpistasis,
    RandomEpistasis,
    SignConsistentEpistasis,
)
from mutascape.landscape import Landscape, Variant
from mutascape.mutant import Mutant, VariantLike, parse_variant
from mutascape.nonlinearity import GlobalEpistasisSigmoid, Identity, NonLinearity
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
    "DEFAULT_BATCH_SIZE",
    "Campaign",
    "CampaignResult",
    "ContactMap",
    "CompletedLandscape",
    "DMSPairwiseLandscape",
    "CouplingTensor",
    "ESM2Scorer",
    "EpistasisModel",
    "GlobalEpistasisSigmoid",
    "Identity",
    "Landscape",
    "LandscapeGenerator",
    "LandscapePrior",
    "MutationDomain",
    "MSACouplingsEpistasis",
    "Mutant",
    "NonLinearity",
    "Oracle",
    "PLMScorer",
    "PosteriorLandscape",
    "RandomEpistasis",
    "SearchMethod",
    "SignConsistentEpistasis",
    "Variant",
    "VariantLike",
    "from_dms",
    "load_oracle",
    "morphing_landscape",
    "run_campaign",
    "plot_alpha_sweep",
    "plot_contact_map",
    "plot_epistasis_scatter",
    "plot_fitness_distribution",
    "parse_variant",
]

__version__ = "0.1.0"
