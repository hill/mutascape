"""
mutascape - Synthetic combinatorial protein fitness landscapes with tunable,
structurally informed epistasis.

Built on real single-site DMS data with epistatic topology derived from
predicted 3D structure (ESMFold/AlphaFold2).

Quick start
-----------
>>> from mutascape import Landscape, ContactMap, RandomEpistasis
>>> contacts = ContactMap.from_pdb("structure.pdb", threshold=8.0)
>>> landscape = Landscape.from_dms(
...     dms_path="dms_data.csv",
...     contact_map=contacts,
...     epistasis=RandomEpistasis(sigma=0.3, seed=42),
...     alpha=1.0,
... )
>>> fitness = landscape.evaluate({57: "G", 72: "V"})
"""

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    AMINO_ACIDS,
    CouplingTensor,
    EpistasisModel,
    RandomEpistasis,
    SignConsistentEpistasis,
)
from mutascape.landscape import Landscape, Variant
from mutascape.mutant import Mutant
from mutascape.scoring import PLMScorer

try:
    from mutascape.scoring import ESM2Scorer
except ImportError:
    pass

from mutascape.plotting import (
    plot_alpha_sweep,
    plot_contact_map,
    plot_epistasis_scatter,
    plot_fitness_distribution,
)

__all__ = [
    "AMINO_ACIDS",
    "ContactMap",
    "CouplingTensor",
    "ESM2Scorer",
    "EpistasisModel",
    "Landscape",
    "Mutant",
    "PLMScorer",
    "RandomEpistasis",
    "SignConsistentEpistasis",
    "Variant",
    "plot_alpha_sweep",
    "plot_contact_map",
    "plot_epistasis_scatter",
    "plot_fitness_distribution",
]

__version__ = "0.1.0"
