"""Posterior landscape services shared by FastAPI and the browser studio."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mutascape.completion import LandscapePrior, PosteriorLandscape
from mutascape.epistasis import AMINO_ACIDS
from mutascape.landscape import _allocate_weighted_counts
from mutascape.mutant import Mutant

PRIOR_ROOT = Path(__file__).resolve().parent / "data" / "priors"
VALIDATION_ROOT = Path(__file__).resolve().parent / "data" / "validation"
DEFAULT_PRIOR_ID = "combinatorial-v2"
DEFAULT_ORACLE_COUPLING = 0.1
DEFAULT_ORACLE_KIND = "dms-pairwise-v1"
MAX_POSTERIOR_COST = 2_000_000
CANONICAL_AAS = set(AMINO_ACIDS)


def _oracle(posterior: PosteriorLandscape, seed: int):
    return posterior.sample_dms_pairwise(seed, coupling=DEFAULT_ORACLE_COUPLING)


def benchmark_validity_summary() -> dict[str, Any]:
    """Return the frozen task-transfer result shown in the studio."""
    return json.loads((VALIDATION_ROOT / "benchmark-validity-v1.json").read_text())


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word.capitalize() for word in rest)


def list_prior_metadata() -> list[dict[str, Any]]:
    """Return metadata for each bundled prior without exposing its parameters."""
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


def load_prior_bundle(prior_id: str = DEFAULT_PRIOR_ID) -> tuple[dict[str, Any], LandscapePrior]:
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


def posterior_from_source(
    source: dict[str, Any],
    prior_id: str = DEFAULT_PRIOR_ID,
) -> tuple[PosteriorLandscape, dict[str, Any]]:
    """Condition a bundled prior on normalised web source observations."""
    metadata, prior = load_prior_bundle(prior_id)
    rows = []
    observation_rows = source.get("observationRows") or _legacy_observation_rows(source)
    for row in observation_rows:
        rows.append(
            {
                "mutations": {int(key): value for key, value in row["mutations"].items()},
                "fitness": float(row["fitness"]),
                "depth": int(row["depth"]),
            }
        )
    observations = pd.DataFrame(rows)
    posterior = PosteriorLandscape.condition(
        source["wtSequence"], observations, prior, prior_id=prior_id
    )
    return posterior, metadata


def _legacy_observation_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"mutations": {}, "fitness": 0.0, "depth": 0}]
    rows.extend(
        {
            "mutations": {str(row["position"]): row["mutant"]},
            "fitness": float(row["fitness"]),
            "depth": 1,
        }
        for row in source["dmsRows"]
    )
    return rows


def query_posterior(
    posterior: PosteriorLandscape,
    entries: list[dict[str, str | None]],
    *,
    positions: list[int],
    wild_type: dict[int, str],
    seed: int,
    ensemble_size: int,
) -> list[dict[str, Any]]:
    """Return fixed oracle values and posterior summaries for exact variants."""
    if ensemble_size < 2 or ensemble_size > 500:
        raise ValueError("ensemble size must be in [2, 500]")
    variants, labels = _parse_entries(
        posterior,
        entries,
        positions=positions,
        wild_type=wild_type,
    )
    _check_cost(variants, ensemble_size)
    oracles = [_oracle(posterior, seed + index) for index in range(ensemble_size)]
    ensemble = np.vstack([oracle.evaluate_batch(variants) for oracle in oracles])
    additive = np.asarray([_additive_fitness(posterior, variant) for variant in variants])
    intervals = {
        level: (
            np.quantile(ensemble, (1.0 - level) / 2.0, axis=0),
            np.quantile(ensemble, 1.0 - (1.0 - level) / 2.0, axis=0),
        )
        for level in (0.5, 0.8, 0.95)
    }
    rows = []
    for index, (label, variant) in enumerate(zip(labels, variants)):
        rows.append(
            {
                "index": index,
                "mutationLabel": label,
                "depth": len(variant),
                "observed": oracles[0].is_observed(variant),
                "additiveFitness": float(additive[index]),
                "fixedOracleFitness": float(ensemble[0, index]),
                "posteriorMean": float(np.mean(ensemble[:, index])),
                "posteriorStd": float(np.std(ensemble[:, index], ddof=1)),
                "intervals": {
                    str(int(level * 100)): {
                        "lower": float(bounds[0][index]),
                        "upper": float(bounds[1][index]),
                    }
                    for level, bounds in intervals.items()
                },
                "samples": [float(value) for value in ensemble[:, index]],
            }
        )
    return rows


def fixed_posterior_entries(
    posterior: PosteriorLandscape,
    entries: list[dict[str, str | None]],
    *,
    positions: list[int],
    wild_type: dict[int, str],
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate entries on one stable posterior draw using the scalar API shape."""
    variants, labels = _parse_entries(
        posterior,
        entries,
        positions=positions,
        wild_type=wild_type,
    )
    values = _oracle(posterior, seed).evaluate_batch(variants)
    return [
        {"index": index, "mutationLabel": label, "fitness": float(value)}
        for index, (label, value) in enumerate(zip(labels, values))
    ]


def preview_posterior(
    posterior: PosteriorLandscape,
    *,
    source_summary: dict[str, Any],
    prior_metadata: dict[str, Any],
    sample_size: int,
    depth_weights: dict[int, float],
    seed: int,
    ensemble_size: int,
) -> dict[str, Any]:
    """Build an uncertainty-aware sample for the browser studio."""
    if sample_size < 1 or sample_size > 5_000:
        raise ValueError("posterior preview size must be in [1, 5000]")
    if ensemble_size < 2 or ensemble_size > 200:
        raise ValueError("preview ensemble size must be in [2, 200]")
    variants = _sample_supported_variants(
        posterior,
        sample_size=sample_size,
        depth_weights=depth_weights,
        seed=seed,
    )
    _check_cost(variants, ensemble_size)
    entries = [
        {
            "mutationLabel": Mutant.from_dict(variant, _wild_type_subset(posterior, variant)).label,
            "compactSequence": None,
        }
        for variant in variants
    ]
    positions = sorted({position for position, _ in posterior.single_effects})
    wild_type = {position: posterior.wild_type_sequence[position - 1] for position in positions}
    query = query_posterior(
        posterior,
        entries,
        positions=positions,
        wild_type=wild_type,
        seed=seed,
        ensemble_size=ensemble_size,
    )
    profile = _posterior_profile(query)
    rows = []
    for variant, result in zip(variants[:100], query[:100]):
        rows.append(
            {
                "sequence": _mutant_sequence(posterior.wild_type_sequence, variant),
                "variant": result["mutationLabel"],
                "depth": result["depth"],
                "additiveFitness": result["additiveFitness"],
                "fixedOracleFitness": result["fixedOracleFitness"],
                "posteriorMean": result["posteriorMean"],
                "posteriorStd": result["posteriorStd"],
                "lower80": result["intervals"]["80"]["lower"],
                "upper80": result["intervals"]["80"]["upper"],
            }
        )
    graph = _posterior_graph(
        posterior,
        variants[:60],
        seed=seed,
        ensemble_size=ensemble_size,
    )
    return {
        "source": source_summary,
        "prior": prior_metadata,
        "settings": {
            "seed": seed,
            "ensembleSize": ensemble_size,
            "generatorKind": DEFAULT_ORACLE_KIND,
            "coupling": DEFAULT_ORACLE_COUPLING,
        },
        "profile": profile,
        "benchmarkValidity": benchmark_validity_summary(),
        "shape": posterior_shape_summary(posterior, seed=seed),
        "sample": {
            "requested": sample_size,
            "returnedRows": len(rows),
            "depthWeights": {str(key): value for key, value in depth_weights.items()},
            "rows": rows,
            "graph": graph,
        },
        "notices": [
            "Bands show variation across fixed synthetic oracle draws.",
            "Each oracle keeps the measured wild type and single mutants exact.",
        ],
    }


def _parse_entries(
    posterior: PosteriorLandscape,
    entries: list[dict[str, str | None]],
    *,
    positions: list[int],
    wild_type: dict[int, str],
) -> tuple[list[dict[int, str]], list[str]]:
    variants = []
    labels = []
    for entry in entries:
        label = entry.get("mutationLabel")
        sequence = entry.get("compactSequence")
        if bool(label) == bool(sequence):
            raise ValueError("each entry needs one mutationLabel or compactSequence")
        if label:
            mutant = posterior.domain.parse(str(label))
        else:
            compact = str(sequence).upper()
            if len(compact) != len(positions):
                raise ValueError(f"compactSequence must contain {len(positions)} amino acids")
            invalid = sorted(set(compact) - CANONICAL_AAS)
            if invalid:
                raise ValueError(
                    f"compactSequence contains unsupported residues: {','.join(invalid)}"
                )
            mutant = posterior.domain.parse(
                {
                    position: amino_acid
                    for position, amino_acid in zip(positions, compact)
                    if amino_acid != wild_type[position]
                }
            )
        variant = dict(mutant)
        variants.append(variant)
        labels.append(Mutant.from_dict(variant, wild_type).label)
    return variants, labels


def _check_cost(variants: list[dict[int, str]], ensemble_size: int) -> None:
    cost = ensemble_size * sum(max(1, len(variant) ** 2) for variant in variants)
    if cost > MAX_POSTERIOR_COST:
        raise ValueError(f"posterior request cost {cost} exceeds the {MAX_POSTERIOR_COST} limit")


def _additive_fitness(posterior: PosteriorLandscape, variant: dict[int, str]) -> float:
    return float(
        posterior.observations[()]
        + sum(
            posterior.single_effects[(position, amino_acid)]
            for position, amino_acid in variant.items()
        )
    )


def _sample_supported_variants(
    posterior: PosteriorLandscape,
    *,
    sample_size: int,
    depth_weights: dict[int, float],
    seed: int,
) -> list[dict[int, str]]:
    alternatives: dict[int, list[str]] = {}
    for position, amino_acid in posterior.single_effects:
        alternatives.setdefault(position, []).append(amino_acid)
    alternatives = {position: sorted(set(values)) for position, values in alternatives.items()}
    weights = {
        int(depth): float(weight) for depth, weight in depth_weights.items() if float(weight) > 0
    }
    if not weights:
        raise ValueError("depth weights need at least one positive value")
    if any(depth < 0 or depth > len(alternatives) for depth in weights):
        raise ValueError(f"mutation depth must be in [0, {len(alternatives)}]")
    maximum = max(weights)
    capacities = _depth_capacities([len(values) for values in alternatives.values()], maximum)
    allocations = _allocate_weighted_counts(
        sample_size,
        weights,
        {depth: capacities[depth] for depth in weights},
    )
    rng = np.random.default_rng(seed ^ 0x5EED5EED)
    positions = np.asarray(sorted(alternatives), dtype=int)
    variants = []
    seen: set[tuple[tuple[int, str], ...]] = set()
    for depth, target in sorted(allocations.items()):
        accepted = 0
        while accepted < target:
            chosen = rng.choice(positions, size=depth, replace=False)
            variant = {
                int(position): str(rng.choice(alternatives[int(position)])) for position in chosen
            }
            key = tuple(sorted(variant.items()))
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant)
            accepted += 1
    return variants


def _depth_capacities(alphabet_sizes: list[int], maximum: int) -> dict[int, int]:
    values = [0] * (maximum + 1)
    values[0] = 1
    for size in alphabet_sizes:
        for depth in range(maximum, 0, -1):
            values[depth] += values[depth - 1] * size
    return {depth: value for depth, value in enumerate(values)}


def _posterior_profile(query: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for depth in sorted({int(row["depth"]) for row in query}):
        group = [row for row in query if row["depth"] == depth]
        samples = np.asarray([row["samples"] for row in group], dtype=float).ravel()
        q05, q25, median, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            {
                "depth": depth,
                "count": len(group),
                "q05": float(q05),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "q95": float(q95),
                "fixedMedian": float(np.median([row["fixedOracleFitness"] for row in group])),
                "additiveMedian": float(np.median([row["additiveFitness"] for row in group])),
                "meanStd": float(np.mean([row["posteriorStd"] for row in group])),
            }
        )
    return rows


def _posterior_graph(
    posterior: PosteriorLandscape,
    sampled: list[dict[int, str]],
    *,
    seed: int,
    ensemble_size: int,
) -> dict[str, Any]:
    wild_type = {
        position: posterior.wild_type_sequence[position - 1]
        for position, _ in posterior.single_effects
    }
    node_mutations: dict[str, dict[int, str]] = {"wt": {}}
    sampled_labels = {Mutant.from_dict(variant, wild_type).label for variant in sampled}
    edges: set[tuple[str, str]] = set()
    for variant in sampled:
        current: dict[int, str] = {}
        previous = "wt"
        for position, amino_acid in sorted(variant.items()):
            current[position] = amino_acid
            label = Mutant.from_dict(current, wild_type).label
            node_mutations.setdefault(label, dict(current))
            edges.add((previous, label))
            previous = label
    entries = [{"mutationLabel": label, "compactSequence": None} for label in node_mutations]
    query = query_posterior(
        posterior,
        entries,
        positions=sorted(wild_type),
        wild_type=wild_type,
        seed=seed,
        ensemble_size=ensemble_size,
    )
    nodes = [
        {
            "id": result["mutationLabel"],
            "variant": result["mutationLabel"],
            "depth": result["depth"],
            "fitness": result["fixedOracleFitness"],
            "posteriorMean": result["posteriorMean"],
            "posteriorStd": result["posteriorStd"],
            "sampled": result["mutationLabel"] in sampled_labels,
        }
        for result in query
    ]
    return {
        "nodes": nodes,
        "edges": [{"source": source, "target": target} for source, target in sorted(edges)],
    }


def _wild_type_subset(
    posterior: PosteriorLandscape,
    variant: dict[int, str],
) -> dict[int, str]:
    return {position: posterior.wild_type_sequence[position - 1] for position in variant}


def _mutant_sequence(wild_type_sequence: str, variant: dict[int, str]) -> str:
    sequence = list(wild_type_sequence)
    for position, amino_acid in variant.items():
        sequence[position - 1] = amino_acid
    return "".join(sequence)


# Enumerating a cube is only worth offering while it stays interactive.
MAX_SHAPE_CUBE = 50_000


def posterior_shape_summary(
    posterior: PosteriorLandscape,
    *,
    seed: int,
    draws: int = 5,
) -> dict[str, Any] | None:
    """Compare the shape of a posterior draw against an additive surrogate.

    The additive surrogate is what a model fitted on singles alone produces. It
    has one peak and carries no variance above interaction order one, so a
    campaign simulated on it cannot rank search methods the way a rugged
    landscape does. Returning both lets the studio show that difference.

    Returns None when the supported domain is too large to enumerate.
    """
    from mutascape.benchmarks.topology import summarise

    positions = sorted({position for position, _ in posterior.single_effects})
    if not positions:
        return None
    alphabets = {
        position: sorted(
            {posterior.wild_type_sequence[position - 1]}
            | {aa for site, aa in posterior.single_effects if site == position}
        )
        for position in positions
    }
    size = 1
    for position in positions:
        size *= len(alphabets[position])
        if size > MAX_SHAPE_CUBE:
            return None

    wild_type = {position: posterior.wild_type_sequence[position - 1] for position in positions}
    variants: list[dict[int, str]] = []
    for combination in product(*(alphabets[position] for position in positions)):
        variants.append(
            {
                position: amino_acid
                for position, amino_acid in zip(positions, combination)
                if amino_acid != wild_type[position]
            }
        )
    shape = tuple(len(alphabets[position]) for position in positions)
    wt = posterior.observations[()]

    additive = np.asarray(
        [_additive_fitness(posterior, variant) - wt for variant in variants], dtype=np.float64
    )
    posterior_summaries = [
        summarise(
            np.asarray(
                _oracle(posterior, seed + index).evaluate_batch(variants), dtype=np.float64
            ).reshape(shape)
        )
        for index in range(draws)
    ]
    keys = ("n_optima", "peak_fraction", "walsh_nonadditive")
    return {
        "cubeSize": size,
        "draws": draws,
        "additive": {
            _camel(key): value
            for key, value in summarise(additive.reshape(shape)).items()
            if key in keys
        },
        "posterior": {
            _camel(key): float(np.median([entry[key] for entry in posterior_summaries]))
            for key in keys
        },
    }
