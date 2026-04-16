"""Load and process real multi-mutant datasets for validation.

Parses ProteinGym CSV format into a standardised DataFrame that can be
passed directly to the metrics module. Handles mutation notation parsing,
single-site effect lookup, and additive prediction computation.
"""

import re
from pathlib import Path

import pandas as pd


def parse_proteingym_mutant(mutant_str: str) -> dict[int, tuple[str, str]]:
    """Parse ProteinGym mutation string into position -> (wt, mut) dict.

    E.g. 'K3R:V55A:Q94R' -> {3: ('K', 'R'), 55: ('V', 'A'), 94: ('Q', 'R')}
    """
    mutations = {}
    for m in mutant_str.split(":"):
        match = re.match(r"([A-Z])(\d+)([A-Z])", m)
        if not match:
            raise ValueError(f"Cannot parse mutation: {m}")
        wt, pos, mut = match.group(1), int(match.group(2)), match.group(3)
        mutations[pos] = (wt, mut)
    return mutations


def load_single_site_effects(dms_csv: str | Path) -> dict[int, dict[str, float]]:
    """Load mutascape-format single-site DMS data into nested dict.

    Returns dict[position, dict[mutant_aa, fitness_effect]].
    """
    df = pd.read_csv(dms_csv)
    effects: dict[int, dict[str, float]] = {}
    for _, row in df.iterrows():
        pos = int(row["position"])
        if pos not in effects:
            effects[pos] = {}
        effects[pos][row["mutant"]] = float(row["fitness"])
    return effects


def load_proteingym_multimutant(
    proteingym_csv: str | Path,
    dms_csv: str | Path,
    wt_fitness: float = 3.72,
    min_depth: int = 1,
    max_depth: int = 15,
) -> pd.DataFrame:
    """Load ProteinGym multi-mutant data and compute additive predictions.

    Filters to variants where all constituent single-site effects are known.

    Parameters
    proteingym_csv : path to ProteinGym CSV (mutant, DMS_score columns)
    dms_csv : path to mutascape single-site CSV (position, wildtype, mutant, fitness)
    wt_fitness : absolute wild-type fitness for converting DMS_score to relative
    min_depth : minimum mutation depth to include
    max_depth : maximum mutation depth to include

    Returns
    DataFrame with columns: mutations (dict), fitness (relative to WT),
        additive_prediction, depth, epistatic_deviation
    """
    pg = pd.read_csv(proteingym_csv)
    effects = load_single_site_effects(dms_csv)

    rows = []
    n_skipped = 0

    for _, row in pg.iterrows():
        parsed = parse_proteingym_mutant(row["mutant"])
        depth = len(parsed)

        if depth < min_depth or depth > max_depth:
            continue

        # Check all single-site effects are known
        additive = 0.0
        all_known = True
        mutations: dict[int, str] = {}

        for pos, (wt, mut) in parsed.items():
            if pos not in effects or mut not in effects[pos]:
                all_known = False
                break
            additive += effects[pos][mut]
            mutations[pos] = mut

        if not all_known:
            n_skipped += 1
            continue

        # Convert absolute DMS_score to relative fitness
        relative_fitness = row["DMS_score"] - wt_fitness

        rows.append(
            {
                "mutations": mutations,
                "fitness": relative_fitness,
                "additive_prediction": additive,
                "depth": depth,
                "epistatic_deviation": relative_fitness - additive,
            }
        )

    df = pd.DataFrame(rows)

    total = len(pg)
    retained = len(df)
    print(f"Loaded {retained}/{total} variants ({n_skipped} skipped, missing single-site effects)")
    print(f"Depth distribution: {df['depth'].value_counts().sort_index().to_dict()}")

    return df


def sample_synthetic_data(
    landscape,
    depths: list[int],
    n_per_depth: int = 3000,
) -> pd.DataFrame:
    """Sample synthetic variants from a mutascape landscape.

    Returns DataFrame in the same format as load_proteingym_multimutant.
    """
    rows = []
    for depth in depths:
        variants, fitness = landscape._sample(n=n_per_depth, n_mutations=depth)
        for v, f in zip(variants, fitness):
            additive = landscape.additive_fitness(v)
            rows.append(
                {
                    "mutations": v,
                    "fitness": float(f),
                    "additive_prediction": float(additive),
                    "depth": depth,
                    "epistatic_deviation": float(f) - float(additive),
                }
            )
    return pd.DataFrame(rows)
