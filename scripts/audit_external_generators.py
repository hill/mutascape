"""Recalculate campaign outcomes from saved query identities."""

import numpy as np
import pandas as pd

from scripts.compare_external_generators import OUT, SETTINGS, digest, save_json, verify


def main():
    verify()
    checked = 0
    total_queries = 0
    for path in sorted(OUT.glob("*/*/trials_*.csv")):
        if "smoke" in path.name:
            continue
        table = pd.read_csv(path)
        design = np.load(path.parent / "design.npz")
        known = design["known"]
        values = np.load(path.parent / f"{table.model.iloc[0]}_{table['sample'].iloc[0]}.npy")
        query_data = np.load(path.with_suffix(".npz"))
        queries, curves = query_data["queries"], query_data["regret_curves"]
        assert queries.shape == (len(table), 96)
        if table.model.iloc[0] in SETTINGS:
            measured = np.load(path.parent / "measured_0.npy")
            np.testing.assert_array_equal(values[known], measured[known])
        for i, row in enumerate(table.itertuples(index=False)):
            selected = queries[i, : row.budget]
            assert len(np.unique(selected)) == row.budget
            assert not np.isin(selected, known).any()
            assert ((selected >= 0) & (selected < len(values))).all()
            assert (queries[i, row.budget :] == -1).all()
            trace = np.maximum.accumulate(np.r_[values[known].max(), values[selected]])
            regret = (values.max() - trace) / np.ptp(values)
            np.testing.assert_allclose(curves[i, : row.budget + 1], regret, rtol=1e-12, atol=1e-12)
            assert np.isnan(curves[i, row.budget + 1 :]).all()
            np.testing.assert_allclose(row.regret, regret[-1], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(row.best, trace[-1], rtol=1e-12, atol=1e-12)
            checked += 1
            total_queries += row.budget
    report = dict(
        campaigns_recalculated=checked,
        new_evaluations_checked=total_queries,
        duplicate_queries=0,
        known_queries_charged=0,
        score_mismatches=0,
        source_sha256=digest(__file__),
    )
    save_json(OUT / "query_audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
