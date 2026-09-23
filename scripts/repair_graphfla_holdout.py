"""Replace the overlapping GraphFLA trials and preserve verified unaffected trials."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from mutascape.benchmarks.datasets import (
    load_combingym_landscapes,
    load_graphfla_landscapes,
    load_ssmula_landscapes,
)
from scripts.run_search_transfer import _graphfla_training, _validate_scores, _write_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/preprint_audit_20260923"))
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(".mutascape/worktrees/posterior-landscapes-v1/notebooks/iclr"),
    )
    parser.add_argument(
        "--graphfla-root",
        type=Path,
        default=Path(
            ".mutascape/worktrees/posterior-landscapes-v1/.mutascape/benchmark-data/graphfla"
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    training = load_ssmula_landscapes(".mutascape/benchmark-data/ssmula", strict=False)
    training += load_combingym_landscapes(".mutascape/benchmark-data/combingym", strict=False)
    targets = load_graphfla_landscapes(args.graphfla_root, strict=True)
    overlap = [d.name for d in targets if d.protein in {t.protein for t in training}]
    if overlap != ["GraphFLA_Poelwijk2019"]:
        raise ValueError(f"overlapping targets changed: {overlap}")
    prior_training = {d.name: [t.name for t in _graphfla_training(d, training)] for d in targets}
    for budget in (48, 96, 192, 384):
        started = time.perf_counter()
        corrected = args.results / f"graphfla_eqfp611_b{budget}"
        archived = args.archive / f"output_validity_external_b{budget}_primary"
        output = args.results / f"graphfla_b{budget}"
        if args.rerun:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_search_transfer.py",
                    "--ssmula-root",
                    ".mutascape/benchmark-data/ssmula",
                    "--combingym-root",
                    ".mutascape/benchmark-data/combingym",
                    "--graphfla-root",
                    str(args.graphfla_root),
                    "--only",
                    *overlap,
                    "--generators",
                    "primary",
                    "--budget",
                    str(budget),
                    "--seeds",
                    "10",
                    "--draws",
                    "3",
                    "--output",
                    str(corrected),
                ],
                check=True,
            )
        old = pd.read_csv(archived / "scores.csv", float_precision="round_trip")
        old["source"] = "graphfla"
        old["protein"] = old.dataset.map({d.name: d.protein for d in targets})
        assert old.protein.notna().all()
        new = pd.read_csv(corrected / "scores.csv", float_precision="round_trip")
        assert set(new.dataset) == set(overlap)
        # These controls do not use the prior. Their agreement checks seed identity.
        keys = ["dataset", "arm", "method", "trial"]
        columns = keys + ["best_fitness", "regret", "space_size", "coverage"]
        controls = [
            "measured",
            "additive",
            *[f"{p}_{d}" for p in ("potts_low", "rmf_high") for d in range(3)],
        ]
        a = old.loc[old.dataset.isin(overlap) & old.arm.isin(controls), columns]
        b = new.loc[new.arm.isin(controls), columns]
        pd.testing.assert_frame_equal(
            a.sort_values(keys).reset_index(drop=True),
            b.sort_values(keys).reset_index(drop=True),
            atol=1e-12,
            rtol=1e-12,
        )
        retained = old.loc[~old.dataset.isin(overlap) & old.arm.isin(new.arm.unique())]
        scores = pd.concat([retained, new], ignore_index=True)
        _validate_scores(scores, expected_trials=10)
        output.mkdir(parents=True, exist_ok=True)
        run_args = SimpleNamespace(
            output=output,
            budget=budget,
            graphfla_root=args.graphfla_root,
            target_source="ssmula",
            minimum_coverage=0.5,
            minimum_candidates=384,
            floor_policy="censored",
            global_epistasis=True,
            dms_pairwise_coupling=0.1,
            seeds=10,
            draws=3,
            seed=20260823,
        )
        _write_results(scores, run_args, "graphfla_protein_holdout", started)
        pairs = pd.read_csv(output / "pairwise_transfer.csv")
        assert set(pairs.loc[pairs.arm == "dms_pairwise_mean", "dataset"]) == {
            d.name for d in targets
        }
        original_pairs = pd.read_csv(archived / "pairwise_transfer.csv")
        columns = ["dataset", "scored_pairs"]
        pd.testing.assert_frame_equal(
            original_pairs.loc[original_pairs.arm == "dms_pairwise_mean", columns].reset_index(
                drop=True
            ),
            pairs.loc[pairs.arm == "dms_pairwise_mean", columns].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(archived / "measured_reliability.csv"),
            pd.read_csv(output / "measured_reliability.csv"),
            atol=1e-12,
            rtol=1e-12,
        )
        (output / "prior_training.json").write_text(json.dumps(prior_training, indent=2) + "\n")
        provenance = {
            "summary_timing_scope": "Summary recalculation only. Trial times remain in scores.csv.",
            "recomputed_datasets": overlap,
            "retained_datasets": sorted(set(old.dataset) - set(overlap)),
            "controls_verified": controls,
            "seed": 20260823,
            "target_seed_index": 5,
            "sources": [
                {
                    "path": str(p),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in (archived / "scores.csv", corrected / "scores.csv")
            ],
        }
        (output / "repair_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Wrote corrected GraphFLA panel at {budget} queries", flush=True)


if __name__ == "__main__":
    main()
