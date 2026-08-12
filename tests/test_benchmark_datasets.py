from pathlib import Path

import pandas as pd
import pytest

from mutascape.benchmarks.datasets import LandscapeSpec, load_landscape


def test_load_combingym_full_sequences(tmp_path: Path):
    table = pd.DataFrame(
        {
            "genotype": ["ACD", "AED", "AEF", "FCF"],
            "n_mut": [0, 1, 2, 2],
            "fitness": [1.0, 0.7, 1.2, 0.1],
        }
    )
    path = tmp_path / "landscape.xlsx"
    table.to_excel(path, index=False)
    spec = LandscapeSpec(
        name="toy",
        protein="toy-protein",
        source="combingym",
        relative_path=path.name,
        fitness_column="fitness",
        sequence_column="genotype",
        depth_column="n_mut",
    )

    dataset = load_landscape(tmp_path, spec)

    assert dataset.wild_type_sequence == "ACD"
    assert dataset.mutable_positions == (1, 2, 3)
    assert dataset.variants["depth"].tolist() == [0, 1, 2, 2]
    assert dataset.variants.loc[2, "mutations"] == {2: "E", 3: "F"}
    assert dataset.alphabets[1] == ("A", "F")


def test_load_ssmula_site_sequences(tmp_path: Path):
    table = pd.DataFrame(
        {
            "AAs": ["VDG", "ADG", "AEG", "***"],
            "fitness": [0.5, 0.3, 0.8, 0.0],
            "n_mut": [0, 1, 2, 3],
        }
    )
    path = tmp_path / "toy.csv"
    table.to_csv(path, index=False)
    spec = LandscapeSpec(
        name="toy",
        protein="toy-protein",
        source="ssmula",
        relative_path=path.name,
        fitness_column="fitness",
        sequence_column="AAs",
        depth_column="n_mut",
        positions=(39, 40, 41),
        wild_type="VDG",
    )

    dataset = load_landscape(tmp_path, spec)

    assert dataset.n_variants == 3
    assert dataset.wild_type_sequence == "VDG"
    assert dataset.mutable_positions == (1, 2, 3)
    assert dataset.variants.loc[2, "mutations"] == {1: "A", 2: "E"}
    assert dataset.spec.positions == (39, 40, 41)


def test_full_sequence_loader_rejects_missing_wild_type(tmp_path: Path):
    table = pd.DataFrame({"genotype": ["AED"], "n_mut": [1], "fitness": [0.7]})
    path = tmp_path / "landscape.csv"
    table.to_csv(path, index=False)
    spec = LandscapeSpec(
        name="toy",
        protein="toy",
        source="combingym",
        relative_path=path.name,
        fitness_column="fitness",
        sequence_column="genotype",
        depth_column="n_mut",
    )

    with pytest.raises(ValueError, match="depth-zero"):
        load_landscape(tmp_path, spec)
