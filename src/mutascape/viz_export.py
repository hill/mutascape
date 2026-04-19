"""Export GB1 data as a compact JSON for the web visualization at /web/.

Reads the three authoritative GB1 files from data/gb1/ and writes
web/src/data/gb1.json. Keyed measurements use a 4-letter string of amino
acids at the four Olson positions (39, 40, 41, 54) to keep the file small.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mutascape.mutant import Mutant

REPO_ROOT = Path(__file__).resolve().parents[2]
GB1_DIR = REPO_ROOT / "data" / "gb1"
OUTPUT = REPO_ROOT / "web" / "data" / "gb1.json"

OLSON_POSITIONS = (39, 40, 41, 54)


def read_wt() -> str:
    lines = (GB1_DIR / "gb1_wt.fasta").read_text().splitlines()
    return "".join(line for line in lines if not line.startswith(">"))


def read_singles() -> dict[str, dict[str, float]]:
    singles: dict[str, dict[str, float]] = {str(p): {} for p in OLSON_POSITIONS}
    with (GB1_DIR / "gb1.csv").open() as f:
        for row in csv.DictReader(f):
            pos = row["position"]
            if pos not in singles:
                continue
            singles[pos][row["mutant"]] = float(row["fitness"])
    return singles


def read_measured(wt: str) -> dict[str, float]:
    """Return {4-letter AA key at Olson positions: fitness}.

    The WT combination is included explicitly at fitness = 0.0.
    """
    wt_aas = "".join(wt[p - 1] for p in OLSON_POSITIONS)
    measured: dict[str, float] = {wt_aas: 0.0}

    with (GB1_DIR / "gb1_combinatorial.csv").open() as f:
        for row in csv.DictReader(f):
            mutant = Mutant(row["mutant"].replace(":", "+"))
            aas = [wt[p - 1] for p in OLSON_POSITIONS]
            for i, pos in enumerate(OLSON_POSITIONS):
                if pos in mutant:
                    aas[i] = mutant[pos]
            measured["".join(aas)] = float(row["fitness"])

    return measured


def main() -> None:
    wt = read_wt()
    wt_aas = "".join(wt[p - 1] for p in OLSON_POSITIONS)
    data = {
        "protein": "GB1",
        "reference": "Wu et al. 2016, eLife 5:e16965",
        "wt": wt,
        "olson": {
            "positions": list(OLSON_POSITIONS),
            "wt_aas": wt_aas,
            "singles": read_singles(),
            "measured": read_measured(wt),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"wrote {OUTPUT} ({size_mb:.2f} MB, {len(data['olson']['measured'])} measured variants)")


if __name__ == "__main__":
    main()
