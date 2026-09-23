"""Recheck the frozen confirmation inventory without reading fitness values."""

import hashlib
import json
import math

import pandas as pd

from scripts.run_review_confirmation import GRAPH, ROOT


def main():
    inventory = pd.read_csv(ROOT / "graphfla_structural_inventory.csv")
    selection = pd.read_csv(ROOT / "confirmation_selection.csv")
    manifest = json.loads((ROOT / "confirmation_manifest.json").read_text())
    assert set(inventory.file) == set(selection.file)
    for record in inventory.itertuples():
        path = GRAPH / "data/BioSequence" / record.file
        columns = pd.read_csv(path, nrows=0).columns
        positions = [c for c in columns if c.startswith("pos")]
        frame = pd.read_csv(path, usecols=positions, dtype=str)
        alphabets = [sorted(frame[c].dropna().unique()) for c in positions]
        cartesian = math.prod(map(len, alphabets))
        assert len(frame) == record.rows, record.file
        assert len(frame.drop_duplicates()) == record.unique_genotypes, record.file
        assert cartesian == record.cartesian, record.file
        assert (len(frame.drop_duplicates()) == cartesian) == record.complete, record.file
    chosen = selection.loc[selection.reason == "Selected unused family", "file"]
    assert set(chosen) == {t["file"] for t in manifest["targets"]}
    for target in manifest["targets"]:
        record = inventory.set_index("file").loc[target["file"]]
        assert record.complete and 384 <= record.cartesian <= 200000
        path = GRAPH / "data/BioSequence" / target["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == target["sha256"]
    print(f"Verified {len(inventory)} inventory rows and {len(chosen)} selected family")


if __name__ == "__main__":
    main()
