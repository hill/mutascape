"""Recalculate all new campaign scores and verify their query identities."""

import numpy as np
import pandas as pd

from scripts.check_generator_improvements import OUT as CHECK
from scripts.check_generator_improvements import verify as verify_check
from scripts.compare_external_generators import digest, save_json
from scripts.replicate_generator_improvements import OUT as REPLICATION
from scripts.replicate_generator_improvements import verify as verify_replication
from scripts.run_generator_improvements import OUT, verify


def audit(root, verifier):
    manifest = verifier()
    expected = len(manifest["inventory"]) * len(manifest["models"]) * 5
    if root == CHECK:
        expected += len(manifest["inventory"]) * 5
    elif root == REPLICATION:
        expected = len(manifest["inventory"]) * (len(manifest["models"]) + 1) * 10
    files = sorted(root.glob("*/*/trials_*.csv"))
    assert len(files) == expected
    checked, queries_checked, warnings, fallback = 0, 0, 0, 0
    for path in files:
        table = pd.read_csv(path)
        design = np.load(path.parent / "design.npz")
        known = design["known"]
        values = np.load(path.parent / f"{table.model.iloc[0]}_{table['sample'].iloc[0]}.npy")
        queries = np.load(path.with_suffix(".npz"))["queries"]
        assert queries.shape == (len(table), 96)
        assert (table.spent == table.budget).all()
        for i, row in enumerate(table.itertuples(index=False)):
            chosen = queries[i, : row.budget]
            assert len(np.unique(chosen)) == row.budget
            assert not np.isin(chosen, known).any()
            assert ((chosen >= 0) & (chosen < len(values))).all()
            assert (queries[i, row.budget :] == -1).all()
            best = max(values[known].max(), values[chosen].max())
            np.testing.assert_allclose(row.best, best, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                row.regret, (values.max() - best) / np.ptp(values), rtol=1e-12, atol=1e-12
            )
            checked += 1
            queries_checked += row.budget
        warnings += int(table.warnings.sum())
        fallback += int(table.fallback_queries.sum())
    report = dict(
        campaigns_recalculated=checked,
        new_evaluations_checked=queries_checked,
        warnings=warnings,
        fallback_queries=fallback,
        duplicate_queries=0,
        known_queries_charged=0,
        score_mismatches=0,
        source_sha256=digest(__file__),
        manifest_sha256=digest(root / "manifest.json"),
    )
    save_json(root / "query_audit.json", report)
    print(root, report, flush=True)


if __name__ == "__main__":
    for root, verifier in [(OUT, verify), (CHECK, verify_check), (REPLICATION, verify_replication)]:
        audit(root, verifier)
