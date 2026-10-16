"""Wire formats for everything Mutascape persists.

A saved oracle is the reproducibility claim: the manifest plus the original DMS
data must rebuild the same landscape months later. These models are the only
description of what those files contain, so a field that is read anywhere is
declared here rather than narrowed at the point of use.

Each payload carries a literal version tag. Restoring a file whose tag this
package does not know fails on the tag rather than on a missing key further in.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A JSON number that must be a real measurement. Serialised NaN or Infinity is
# a corrupt file, not a value any of these fields may legitimately take.
Real = Annotated[float, Field(allow_inf_nan=False)]

# Position and residue, as written by `[position, residue]` in JSON.
Substitution = tuple[int, str]


class Payload(BaseModel):
    """Shared settings for every persisted payload.

    Unknown keys pass through so a file written by a later version still loads
    on the fields this one understands.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)


class ReadoutSpec(Payload):
    """Bounds and steepness of a saved global-epistasis sigmoid."""

    floor: Real
    ceiling: Real
    scale: Real = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.ceiling <= self.floor:
            raise ValueError("ceiling must exceed floor")
        return self


class AnchorRow(Payload):
    """One measurement the generator reproduces exactly."""

    mutations: list[Substitution]
    fitness: Real


class EffectRow(Payload):
    """One latent single-site effect."""

    position: int
    residue: str
    effect: Real


class PilotModel(Payload):
    """Depth and pair terms fitted to a combination calibration sample."""

    model: Literal["pilot-burden-v1"] = "pilot-burden-v1"
    wild_type_sequence: str = Field(min_length=1)
    anchors: list[AnchorRow]
    effects: list[EffectRow]
    # Depth, then the depth-dependent rescaling of the summed single-site effect.
    coefficients: list[Real] = Field(min_length=2, max_length=2)
    pair_scale: Real = Field(ge=0)
    readout: ReadoutSpec
    calibration_keys: list[list[Substitution]]

    @model_validator(mode="after")
    def _has_wild_type(self) -> Self:
        if not any(not row.mutations for row in self.anchors):
            raise ValueError("anchors must include wild type")
        return self


class QuantileModel(Payload):
    """A pilot model plus the measured distribution its scores map onto.

    Version 1 predates the latent upper tail and restores with the linear one.
    """

    model: Literal["quantile-pilot-v1", "quantile-pilot-v2"] = "quantile-pilot-v2"
    reference: PilotModel
    pilot_fitness: list[Real] = Field(min_length=3)
    upper_tail: Literal["latent", "linear"] | None = None
    # Files written before the tail was configurable used this share.
    tail_fraction: Real = Field(default=0.1, gt=0, le=1)

    @model_validator(mode="after")
    def _tail_matches_version(self) -> Self:
        if self.model == "quantile-pilot-v1" and self.upper_tail not in (None, "linear"):
            raise ValueError("version 1 payloads predate the latent tail")
        return self

    @property
    def resolved_upper_tail(self) -> Literal["latent", "linear"]:
        if self.upper_tail is not None:
            return self.upper_tail
        return "linear" if self.model == "quantile-pilot-v1" else "latent"


class PriorPayload(Payload):
    """Interaction and assay-range statistics pooled over training landscapes.

    The per-order and per-depth maps are keyed by integer written as a JSON
    object key, so they arrive as strings and are coerced back.
    """

    order_location: dict[int, Real] = Field(default_factory=dict)
    order_scale: dict[int, Real] = Field(default_factory=dict)
    depth_location: dict[int, Real] = Field(default_factory=dict)
    depth_scale: dict[int, Real] = Field(default_factory=dict)
    depth_between: dict[int, Real] = Field(default_factory=dict)
    depth_shrinkage: dict[int, Real] = Field(default_factory=dict)
    order_shape: dict[int, list[Real]] = Field(default_factory=dict)
    lower_bound: Real
    upper_bound: Real
    readout_scale_ratio: Real = 1.0
    combinatorial_upside: Real = 1.0
    combinatorial_downside: Real = 1.0
    magnitude_edges: list[Real] = Field(default_factory=list)
    magnitude_scale: list[Real] = Field(default_factory=list)
    texture_scale: Real = 0.0
    texture_order: int = 3
    texture_iid_fraction: Real = 0.3
    degrees_of_freedom: Real = 5.0
    n_landscapes: int = 0
    n_proteins: int = 0


class OracleManifest(Payload):
    """Settings that rebuild one fixed single-site-only oracle.

    `source_digest` binds the manifest to the DMS data it was built from, so a
    reload against different measurements fails rather than returning a
    different landscape under the same name.
    """

    generator: Literal["dms-sigmoid-v1"]
    mutascape_version: str = "unknown"
    seed: int
    strength: Real = Field(ge=0)
    mode: Literal["linear", "sigmoid", "pool"]
    margin: Real = Field(gt=0)
    lower: Real | None = None
    upper: Real | None = None
    contacts: list[tuple[int, int]] | None = None
    source_digest: str
    observations: int = Field(ge=0)
    wild_type_fitness: Real
    positions: list[int] = Field(default_factory=list)
    substitutions: int = Field(ge=0)
    maximum_depth: int | None = Field(default=None, ge=1)
    completion: dict | None = None


class LegacyOracleManifest(Payload):
    """Settings that identify one oracle from the empirical-prior generator."""

    generator: Literal["dms-pairwise-v1"]
    mutascape_version: str = "unknown"
    prior: str
    seed: int
    coupling: Real
    source_digest: str
    observations: int = Field(ge=0)
    wild_type_fitness: Real
    positions: list[int] = Field(default_factory=list)
    substitutions: int = Field(ge=0)
    maximum_depth: int | None = Field(default=None, ge=1)
    completion: dict | None = None


def generator_id(payload: object) -> str | None:
    """Return the generator tag of a decoded manifest, before it is parsed."""
    if isinstance(payload, dict):
        tag = payload.get("generator")
        return tag if isinstance(tag, str) else None
    return None
