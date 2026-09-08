"""Calibrate masked ESM scores on measured singles and fill missing substitutions."""

from dataclasses import replace

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mutascape.completion import PosteriorLandscape
from mutascape.epistasis import AMINO_ACIDS
from mutascape.scoring import ESM2Scorer

DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"


def fit_single_calibration(scores: np.ndarray, fitness: np.ndarray):
    """Fit ridge calibration with one score and an intercept on the assay scale."""
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    fitness = np.asarray(fitness, dtype=float)
    if len(scores) != len(fitness) or len(fitness) < 3:
        raise ValueError("ESM calibration needs at least three measured singles")
    if not np.isfinite(scores).all() or not np.isfinite(fitness).all():
        raise ValueError("ESM calibration needs finite scores and fitness")
    if np.std(scores) < 1e-12 or np.std(fitness) < 1e-12:
        raise ValueError("ESM calibration needs variation in scores and measured fitness")
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(scores, fitness)


def restore_completion(generator: PosteriorLandscape, completion: dict) -> PosteriorLandscape:
    """Restore frozen imputed effects without loading model weights."""
    if completion.get("method") != "esm2-ridge-v1":
        raise ValueError("unknown single completion method")
    effects = {}
    for position, amino_acid, effect in completion["effects"]:
        key = (int(position), str(amino_acid))
        if (
            not 1 <= key[0] <= len(generator.wild_type_sequence)
            or key[1] not in AMINO_ACIDS
            or key[1] == generator.wild_type_sequence[key[0] - 1]
            or key in generator.single_effects
            or key in effects
            or not np.isfinite(effect)
        ):
            raise ValueError("invalid or overlapping imputed substitution")
        effects[key] = float(effect)
    preimages = dict(generator.preimage_effects)
    if generator.nonlinearity is not None:
        preimages.update(
            {key: float(generator.nonlinearity.inverse(value)) for key, value in effects.items()}
        )
    return replace(
        generator,
        single_effects={**generator.single_effects, **effects},
        preimage_effects=preimages,
        imputed_effects=effects,
        completion_metadata=completion,
    )


def complete_singles(
    generator: PosteriorLandscape, *, model_name: str = DEFAULT_ESM_MODEL
) -> PosteriorLandscape:
    """Fill the whole protein while retaining the readout fitted on measured singles."""
    missing = [
        (position, aa)
        for position, wt in enumerate(generator.wild_type_sequence, 1)
        for aa in AMINO_ACIDS
        if aa != wt and (position, aa) not in generator.single_effects
    ]
    if not missing:
        return generator
    scores = ESM2Scorer(model_name=model_name, batch_size=8).score(generator.wild_type_sequence)
    keys = list(generator.single_effects)
    y = np.asarray([generator.single_effects[key] for key in keys])
    calibration = fit_single_calibration(np.asarray([scores[p][aa] for p, aa in keys]), y)
    predictions = calibration.predict(np.asarray([scores[p][aa] for p, aa in missing])[:, None])
    # Bound extrapolation by measured singles without changing the assay readout.
    predictions = np.clip(predictions, y.min(), y.max())
    completion = {
        "method": "esm2-ridge-v1",
        "model": model_name,
        "calibration_singles": len(keys),
        "clip_bounds": [float(y.min()), float(y.max())],
        "effects": [[p, aa, float(value)] for (p, aa), value in zip(missing, predictions)],
    }
    return restore_completion(generator, completion)
