"""Vary missing-pattern filling while keeping the full CreiLOV domain fixed."""

from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks import roots
from mutascape.benchmarks.datasets import load_combingym_landscapes, load_ssmula_landscapes
from mutascape.benchmarks.generators import dms_pairwise_readout_tensor
from mutascape.benchmarks.splits import mutation_order_split
from mutascape.benchmarks.topology import measured_tensor, supported_cube
from mutascape.completion import LandscapePrior, PosteriorLandscape
from mutascape.validation.fourier import aggregate_binary_cube, walsh_l1_distance
from scripts.audit_preprint_results import _spectrum
from scripts.run_creilov_shape_validation import _binary_indices, _full_dms_observations


def main():
    datasets = load_ssmula_landscapes(
        str(roots.ssmula()), strict=False
    ) + load_combingym_landscapes(str(roots.combingym()), strict=False)
    target = next(d for d in datasets if d.name == "CreiLOV_fluorescence")
    train = mutation_order_split(target.variants, max_train_depth=1).train
    cube = supported_cube(target, train)
    real, _ = measured_tensor(target, cube)
    mask = np.isfinite(real)
    indices = _binary_indices(mask, cube)
    binary = aggregate_binary_cube(real[mask], indices, len(cube.positions))
    prior = LandscapePrior.fit([d for d in datasets if d.protein != target.protein], max_order=3)
    observations = _full_dms_observations(Path("data/creilov/creilov.csv"), target, train)
    posterior = PosteriorLandscape.condition(target.wild_type_sequence, observations, prior)
    rules = {"mean": np.nanmean, "median": np.nanmedian, "minimum": np.nanmin, "maximum": np.nanmax}

    def filled(values, rule):
        return np.where(np.isfinite(values), values, rule(values))

    spectra = {name: _spectrum(filled(binary, rule)) for name, rule in rules.items()}
    rows = []
    for model, coupling, draws in [("no_pairs", 0.0, 1), ("Mutascape", 0.1, 30)]:
        for draw in range(draws):
            values = dms_pairwise_readout_tensor(
                cube, posterior, coupling=coupling, seed=20260826 + draw
            )[mask]
            predicted = aggregate_binary_cube(values, indices, len(cube.positions))
            for name, rule in rules.items():
                rows.append(
                    dict(
                        model=model,
                        draw=draw,
                        filling=name,
                        walsh_l1=walsh_l1_distance(
                            spectra[name], _spectrum(filled(predicted, rule))
                        ),
                    )
                )
    results = pd.DataFrame(rows)
    results.to_csv("results/review_response_20260923/walsh_filling.csv", index=False)
    print(results.groupby(["model", "filling"]).walsh_l1.median())


if __name__ == "__main__":
    main()
