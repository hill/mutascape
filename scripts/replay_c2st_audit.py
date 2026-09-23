"""Recompute classifier results from frozen features without source workbooks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mutascape.benchmarks.evaluation import classifier_two_sample_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.directory / "metadata.json").read_text())
    metrics = pd.read_csv(args.directory / "metrics.csv")
    for row in metrics.itertuples(index=False):
        pilot_protocol = "split_seed" in metrics.columns
        seed = row.split_seed if pilot_protocol else row.seed
        left = "measured" if pilot_protocol else row.left
        right = row.arm if pilot_protocol else row.right
        with np.load(args.directory / f"features_{seed}.npz") as features:
            result = classifier_two_sample_test(
                features[left],
                features[right],
                real_groups=features["groups"],
                synthetic_groups=features["groups"],
                n_permutations=0,
                seed=seed,
                n_splits=metadata.get("folds", 5),
                classifier=row.classifier,
            )
        np.testing.assert_allclose(result.fold_aucs, json.loads(row.fold_aucs), atol=1e-12)
        np.testing.assert_allclose(
            result.fold_accuracies, json.loads(row.fold_accuracies), atol=1e-12
        )
    print(f"Reproduced AUC and accuracy for all {len(metrics)} comparisons.")


if __name__ == "__main__":
    main()
