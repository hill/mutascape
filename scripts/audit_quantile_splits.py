"""Check confirmation exclusions, including reconstructed historical CreiLOV pilots."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.confirm_quantile_pilot import OUT, PROTEINS, load
from scripts.test_pair_pilot_transfer import OUT as PREVIOUS


def main():
    records = []
    for name in PROTEINS:
        pilot = pd.read_csv(OUT / name / "pilot.csv").row.to_numpy()
        test = pd.read_csv(OUT / name / "test.csv").row.to_numpy()
        assert len(set(pilot)) == 192 and len(set(test)) == 1000
        assert set(pilot).isdisjoint(test)
        excluded = set()
        if name in ("DHFR", "eqFP611", "ParB-Noc"):
            for role in ("pilot", "test"):
                for split in range(3):
                    excluded.update(pd.read_csv(PREVIOUS / name / f"{role}_{split}.csv").row)
        elif name == "CreiLOV":
            frame, *_ = load(name)
            variants = set()
            old = set()
            for seed in (42, 43, 44):
                old.update(pd.read_csv(f"results/creilov_c2st/samples_{seed}.csv").variant)
            pool = frame.loc[(frame.depth > 1) & ~frame.variant.isin(old)]
            for seed in (42, 43, 44):
                variants.update(pool.sample(n=192, random_state=seed).variant)
            files = list(Path("results").glob("creilov*/samples_*.csv")) + list(
                Path("results").glob("creilov*/pilot_*.csv")
            )
            files += list(Path("results/c2st_alternatives_20260923").glob("pilot_*.csv")) + list(
                Path("results/c2st_alternatives_20260923").glob("test_*.csv")
            )
            for path in files:
                data = pd.read_csv(path)
                if "variant" in data:
                    variants.update(data.variant)
            excluded.update(frame.index[frame.variant.isin(variants)])
            repair = json.loads((OUT / "creilov_exclusion_repair.json").read_text())
            for role, actual in (("pilot", pilot), ("test", test)):
                original = pd.read_csv(
                    OUT / "repaired_inputs" / "CreiLOV_before_exclusion" / f"{role}.csv"
                ).row.to_numpy(copy=True)
                for change in repair["changes"]:
                    if change["role"] == role:
                        assert original[change["slot"]] == change["old_row"]
                        original[change["slot"]] = change["new_row"]
                np.testing.assert_array_equal(actual, original)
        assert set(pilot).isdisjoint(excluded)
        assert set(test).isdisjoint(excluded)
        records.append(dict(protein=name, pilot=192, test=1000, excluded=len(excluded), overlap=0))
    (OUT / "split_audit.json").write_text(json.dumps(records, indent=2) + "\n")
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
