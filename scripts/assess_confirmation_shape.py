"""Run the recorded classifier and spectral checks on the new protein family."""

import json

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from mutascape.benchmarks.datasets import _graphfla_spec, load_landscape
from mutascape.benchmarks.evaluation import classifier_two_sample_test, completion_fidelity_features
from mutascape.benchmarks.generators import dms_pairwise_readout_tensor
from mutascape.benchmarks.splits import mutation_order_split
from mutascape.benchmarks.topology import measured_tensor, supported_cube
from mutascape.completion import LandscapePrior, PosteriorLandscape
from mutascape.validation.fourier import walsh_l1_distance
from scripts.audit_preprint_results import _spectrum
from scripts.run_review_confirmation import GRAPH, ROOT


def main():
    target = json.loads((ROOT / "confirmation_manifest.json").read_text())["targets"][0]
    dataset = load_landscape(
        GRAPH, _graphfla_spec(target["file"], target["protein"], target["reference"])
    )
    split = mutation_order_split(dataset.variants, max_train_depth=1)
    cube = supported_cube(dataset, split.train)
    measured, coverage = measured_tensor(dataset, cube)
    assert coverage == 1
    prior = LandscapePrior.from_dict(json.loads((ROOT / "confirmation_prior.json").read_text()))
    posterior = PosteriorLandscape.condition(dataset.wild_type_sequence, split.train, prior)
    test = dataset.variants.loc[dataset.variants.depth > 1].sample(n=2000, random_state=20260923)
    real_features = completion_fidelity_features(split.train, test).to_numpy()
    groups = np.arange(len(test))
    # Match genotype order through each site's declared alphabet.
    indices = tuple(
        np.array(
            [
                cube.alphabets[p].index(m.get(p, dataset.wild_type_sequence[p - 1]))
                for m in test.mutations
            ]
        )
        for p in cube.positions
    )
    real_spectrum = _spectrum(measured)
    rows, spectra = [], []
    saved_features = {"measured": real_features, "groups": groups}
    for draw in range(3):
        tensor = dms_pairwise_readout_tensor(cube, posterior, coupling=0.1, seed=20260923 + draw)
        # Dense tensors are WT-centred. Classifier features expect assay fitness.
        values = tensor[indices] + posterior.observations[()]
        np.testing.assert_allclose(
            values[:20],
            posterior.draw(20260923 + draw).evaluate_batch(test.mutations.tolist()[:20]),
            atol=1e-12,
        )
        features = completion_fidelity_features(split.train, test, values).to_numpy()
        saved_features[f"draw_{draw}"] = features
        spectrum = _spectrum(tensor)
        for order, value in spectrum.items():
            spectra.append(
                dict(draw=draw, order=order, measured=real_spectrum[order], synthetic=value)
            )
        for classifier in ("logistic", "histogram"):
            result = classifier_two_sample_test(
                real_features,
                features,
                real_groups=groups,
                synthetic_groups=groups,
                classifier=classifier,
                seed=20260923,
                n_permutations=0,
            )
            rows.append(
                dict(
                    draw=draw,
                    classifier=classifier,
                    auc=result.auc,
                    balanced_accuracy=result.accuracy,
                    raw_w1=wasserstein_distance(measured.ravel(), tensor.ravel()),
                    raw_w1_real_sd=wasserstein_distance(measured.ravel(), tensor.ravel())
                    / measured.std(),
                    spectrum_l1=walsh_l1_distance(real_spectrum, spectrum),
                )
            )
        print(f"Assessed confirmation draw {draw}", flush=True)
    np.savez_compressed(ROOT / "confirmation_features.npz", **saved_features)
    test.to_csv(ROOT / "confirmation_classifier_genotypes.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "confirmation_shape.csv", index=False)
    pd.DataFrame(spectra).to_csv(ROOT / "confirmation_spectra.csv", index=False)


if __name__ == "__main__":
    main()
