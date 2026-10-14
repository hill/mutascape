"""Check that full-dataset C2ST keeps genotype pairs together."""

import numpy as np
from sklearn.model_selection import GroupKFold

from scripts.evaluate_full_c2st import paired_folds


def test_saved_folds_match_classifier_and_keep_genotypes_out_of_training():
    rows = np.arange(17, 118)
    groups, folds = paired_folds(rows, seed=42)
    paired = np.tile(groups, 2)
    labels = np.repeat([0, 1], len(rows))
    for fold, (train, test) in enumerate(GroupKFold(5).split(paired[:, None], labels, paired)):
        assert not np.intersect1d(paired[train], paired[test]).size
        np.testing.assert_array_equal(test[test < len(rows)], np.flatnonzero(folds == fold))
        assert (labels[test] == 0).sum() == (labels[test] == 1).sum()


def test_fold_assignment_is_reproducible_and_changes_with_seed():
    rows = np.arange(101)
    a, folds = paired_folds(rows, seed=42)
    b, repeated = paired_folds(rows, seed=42)
    _, alternative = paired_folds(rows, seed=43)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(folds, repeated)
    assert not np.array_equal(folds, alternative)
    assert np.ptp(np.bincount(folds)) <= 1
