"""mutascape - Synthetic combinatorial protein fitness landscapes with tuneable,
structurally informed epistasis.

Built on real single-site DMS data with epistatic topology derived from
3D protein structure.

>>> from mutascape import Landscape, ContactMap, Mutant, RandomEpistasis
>>> contacts = ContactMap.from_pdb("structure.pdb", threshold=8.0)
>>> landscape = Landscape.from_dms(
...     dms_path="dms_data.csv",
...     wt_sequence="MAEQLA...",
...     contact_map=contacts,
...     epistasis=RandomEpistasis(sigma=0.3, seed=42),
...     alpha=1.0,
... )
>>> landscape["A57G"]
>>> landscape[Mutant("A57G") + Mutant("V72L")]
"""

from mutascape.contacts import ContactMap
from mutascape.epistasis import (
    AMINO_ACIDS,
    CouplingTensor,
    EpistasisModel,
    RandomEpistasis,
    SignConsistentEpistasis,
)
from mutascape.interactive import morphing_landscape
from mutascape.landscape import Landscape, Variant
from mutascape.mutant import Mutant
from mutascape.plotting import (
    plot_alpha_sweep,
    plot_contact_map,
    plot_epistasis_scatter,
    plot_fitness_distribution,
)
from mutascape.scoring import ESM2Scorer, PLMScorer

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
    "morphing_landscape",
    "plot_alpha_sweep",
    "plot_contact_map",
    "plot_epistasis_scatter",
    "plot_fitness_distribution",
]

__version__ = "0.1.0"
