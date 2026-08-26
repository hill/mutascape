"""Versioned empirical priors bundled with Mutascape."""

import json
from pathlib import Path
from typing import Any

from mutascape.completion import LandscapePrior

PRIOR_ROOT = Path(__file__).resolve().parent / "data" / "priors"
DEFAULT_PRIOR_ID = "combinatorial-v2"


def list_prior_metadata() -> list[dict[str, Any]]:
    """Return public metadata for each bundled prior."""
    priors = []
    for path in sorted(PRIOR_ROOT.glob("*.json")):
        bundle = json.loads(path.read_text())
        parameters = LandscapePrior.from_dict(bundle["parameters"])
        priors.append(
            {key: value for key, value in bundle.items() if key != "parameters"}
            | {
                "nLandscapes": parameters.n_landscapes,
                "nProteins": parameters.n_proteins,
                "globalEpistasis": True,
                "depthShrinkage": {
                    str(depth): round(weight, 4)
                    for depth, weight in sorted(parameters.depth_shrinkage.items())
                },
            }
        )
    return priors


def load_prior_bundle(
    prior_id: str = DEFAULT_PRIOR_ID,
) -> tuple[dict[str, Any], LandscapePrior]:
    """Load one versioned prior and its public metadata."""
    path = PRIOR_ROOT / f"{prior_id}.json"
    if not path.exists():
        raise ValueError(f"unknown posterior prior {prior_id}")
    bundle = json.loads(path.read_text())
    prior = LandscapePrior.from_dict(bundle["parameters"])
    metadata = {key: value for key, value in bundle.items() if key != "parameters"}
    metadata["nLandscapes"] = prior.n_landscapes
    metadata["nProteins"] = prior.n_proteins
    metadata["depthShrinkage"] = {
        str(depth): round(weight, 4) for depth, weight in sorted(prior.depth_shrinkage.items())
    }
    return metadata, prior
