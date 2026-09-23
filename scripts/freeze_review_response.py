"""Check review output completeness and record source and code hashes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/review_response_20260923")


def main():
    readout = pd.read_csv(ROOT / "readout_summary.csv")
    confirmation = pd.read_csv(ROOT / "confirmation_summary.csv")
    shape = pd.read_csv(ROOT / "confirmation_shape.csv")
    pilot = pd.read_csv(ROOT / "pilot/matched_metrics.csv")
    assert len(readout) == 60
    assert len(confirmation) == 24
    assert len(shape) == 6
    assert len(pilot) == 36
    baseline = pd.read_csv("results/preprint_audit_20260923/search_summary.csv")
    identifiers = {
        "dms_pairwise_mean": "Mutascape",
        "potts_low_mean": "Potts",
        "rmf_high_mean": "RMF",
        "additive": "Additive",
    }
    for row in readout.itertuples():
        if row.model not in identifiers:
            continue
        old = baseline.loc[
            (baseline.collection == row.collection)
            & (baseline.budget == row.budget)
            & (baseline.model == identifiers[row.model])
        ].iloc[0]
        np.testing.assert_allclose(
            # The frozen manuscript summary stores six decimal places.
            [row.kendall, row.accuracy],
            [old.kendallTau, old.pairwiseAccuracy],
            atol=5.1e-7,
            rtol=0,
        )
        assert row.eligible == old.eligible_pairs
    assert readout.groupby(["collection", "budget"]).eligible.nunique().eq(1).all()
    assert confirmation.groupby("budget").eligible.nunique().eq(1).all()
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "metadata.json")
    code = [
        Path("src/mutascape/experimental/pilot.py"),
        Path("src/mutascape/benchmarks/validity.py"),
        Path("scripts/run_search_transfer.py"),
    ]
    code += sorted(Path("scripts").glob("*review*.py"))
    code += [
        Path("scripts/check_walsh_filling.py"),
        Path("scripts/validate_pilot_generator.py"),
        Path("scripts/assess_confirmation_shape.py"),
        Path("scripts/run_readout_ablation.py"),
        Path("scripts/check_confirmation_selection.py"),
    ]
    payload = {
        "protocol": "review-response-2026-09-23",
        "development_assay_entries": 31,
        "development_protein_groups": 17,
        "prospective_families": 1,
        "budgets": [48, 96, 192, 384],
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        "output_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "null_summary": pd.read_csv(ROOT / "null_summary.csv").to_dict("records"),
        "confirmation_summary": confirmation.to_dict("records"),
    }
    (ROOT / "metadata.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Verified complete comparisons and recorded {len(files)} result hashes")


if __name__ == "__main__":
    main()
