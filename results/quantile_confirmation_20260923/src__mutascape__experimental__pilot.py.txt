"""A fixed landscape calibrated with measured combinatorial variants."""

from dataclasses import dataclass
from itertools import combinations
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mutascape.completion import dms_pairwise_unit_coefficient
from mutascape.domain import MutationDomain
from mutascape.mutant import VariantLike, parse_variant
from mutascape.nonlinearity import GlobalEpistasisSigmoid


def _design(keys, effects):
    depth = np.asarray([len(key) for key in keys])
    additive = np.asarray([sum(effects[k] for k in key) for key in keys])
    extra = np.maximum(depth - 1, 0)
    return additive, np.column_stack([extra, additive * extra / np.maximum(depth, 1)])


@dataclass(frozen=True)
class PilotCalibratedLandscape:
    """Fit a sigmoid burden model from same-assay singles and a combination pilot.

    This experimental model needs measured combinations. It does not replace
    the singles-only default. Pilot measurements fit aggregate terms and are
    not imposed as exact anchors.
    """

    domain: MutationDomain
    anchors: dict
    effects: dict
    coefficients: NDArray[np.float64]
    pair_scale: float
    readout: GlobalEpistasisSigmoid
    calibration_keys: tuple

    @classmethod
    def fit(cls, wild_type_sequence: str, singles: pd.DataFrame, pilot: pd.DataFrame) -> Self:
        """Fit using only the supplied wild type, singles, and calibration variants."""
        if singles.empty or pilot.empty:
            raise ValueError("singles and pilot must contain measurements")
        anchors = {}
        for mutations, fitness in singles[["mutations", "fitness"]].itertuples(
            index=False, name=None
        ):
            key = tuple(
                sorted(parse_variant(mutations, wild_type_sequence=wild_type_sequence).items())
            )
            if len(key) > 1:
                raise ValueError("single-mutant input contains a combination")
            if not np.isfinite(fitness):
                raise ValueError("anchor fitness must be finite")
            if key in anchors:
                raise ValueError("duplicate anchor genotype")
            anchors[key] = float(fitness)
        if () not in anchors:
            raise ValueError("single-mutant input must include wild type")
        substitutions = tuple(key[0] for key in anchors if key)
        domain = MutationDomain(wild_type_sequence, substitutions)
        keys = tuple(tuple(sorted(domain.parse(m).items())) for m in pilot.mutations)
        if any(len(k) < 2 for k in keys):
            raise ValueError("pilot must contain only multi-mutants")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate pilot genotype")
        y = pilot.fitness.to_numpy(dtype=float) - anchors[()]
        if not np.isfinite(y).all():
            raise ValueError("pilot fitness must be finite")
        single_effects = {k[0]: v - anchors[()] for k, v in anchors.items() if k}
        observed = np.r_[y, list(single_effects.values()), 0.0]
        span = float(np.ptp(observed))
        if span <= 1e-12:
            raise ValueError("calibration fitness has no variation")
        readout = GlobalEpistasisSigmoid(
            floor=float(observed.min() - 0.05 * span),
            ceiling=float(observed.max() + 0.05 * span),
            scale=1.0,
        )
        effects = {k: float(readout.inverse(v)) for k, v in single_effects.items()}
        additive, design = _design(keys, effects)
        if len(keys) < 3 or np.linalg.matrix_rank(design) < 2:
            raise ValueError("pilot must identify both burden terms")
        model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1, fit_intercept=False))
        model.fit(design, readout.inverse(y) - additive)
        residual = readout.inverse(y) - additive - model.predict(design)
        n_pairs = np.asarray([len(k) * (len(k) - 1) / 2 for k in keys])
        scale = float(np.sqrt(np.sum(residual**2) / np.sum(n_pairs)))
        return cls(
            domain, anchors, effects, model[-1].coef_ / model[0].scale_, scale, readout, keys
        )

    def to_dict(self) -> dict:
        """Return fitted parameters and input identities for model restoration."""
        return {
            "model": "pilot-burden-v1",
            "wild_type_sequence": self.domain.wild_type_sequence,
            "anchors": [
                {"mutations": list(key), "fitness": value} for key, value in self.anchors.items()
            ],
            "effects": [
                {"position": p, "residue": aa, "effect": value}
                for (p, aa), value in self.effects.items()
            ],
            "coefficients": self.coefficients.tolist(),
            "pair_scale": self.pair_scale,
            "readout": {
                "floor": self.readout.floor,
                "ceiling": self.readout.ceiling,
                "scale": self.readout.scale,
            },
            "calibration_keys": [list(key) for key in self.calibration_keys],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Self:
        """Restore an experimental model without refitting calibration data."""
        if payload.get("model") != "pilot-burden-v1":
            raise ValueError("unsupported pilot model version")
        anchors = {
            tuple((int(p), aa) for p, aa in row["mutations"]): float(row["fitness"])
            for row in payload["anchors"]
        }
        effects = {
            (int(row["position"]), row["residue"]): float(row["effect"])
            for row in payload["effects"]
        }
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        pair_scale = float(payload["pair_scale"])
        if (
            coefficients.shape != (2,)
            or not np.isfinite(coefficients).all()
            or not np.isfinite(pair_scale)
            or pair_scale < 0
            or () not in anchors
        ):
            raise ValueError("invalid pilot model parameters")
        domain = MutationDomain(payload["wild_type_sequence"], tuple(effects))
        keys = tuple(tuple((int(p), aa) for p, aa in key) for key in payload["calibration_keys"])
        return cls(
            domain,
            anchors,
            effects,
            coefficients,
            pair_scale,
            GlobalEpistasisSigmoid(**payload["readout"]),
            keys,
        )

    def draw(self, seed: int) -> PilotLandscapeDraw:
        """Return a reproducible oracle with one shared pair field."""
        return PilotLandscapeDraw(self, int(seed))


@dataclass(frozen=True)
class PilotLandscapeDraw:
    """One fixed draw from the experimental pilot-calibrated model."""

    generator: PilotCalibratedLandscape
    seed: int

    def evaluate_batch(self, variants: list[VariantLike]) -> NDArray[np.float64]:
        """Evaluate supported variants in input order, preserving measured singles."""
        g = self.generator
        keys = [tuple(sorted(g.domain.parse(v).items())) for v in variants]
        if not keys:
            return np.empty(0, dtype=float)
        additive, design = _design(keys, g.effects)
        noise = np.asarray(
            [
                sum(
                    dms_pairwise_unit_coefficient(a, b, g.domain.wild_type_sequence, self.seed) / 2
                    for a, b in combinations(key, 2)
                )
                for key in keys
            ]
        )
        values = g.anchors[()] + g.readout.forward(
            additive + design @ g.coefficients + g.pair_scale * noise
        )
        for index, key in enumerate(keys):
            if key in g.anchors:
                values[index] = g.anchors[key]
        return np.asarray(values, dtype=float)

    def evaluate(self, variant: VariantLike) -> float:
        """Evaluate one supported genotype."""
        return float(self.evaluate_batch([variant])[0])

    def __getitem__(self, variant: VariantLike) -> float:
        return self.evaluate(variant)
