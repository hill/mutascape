"""Normalise SSMuLA and CombinGym landscapes for completion benchmarks.

Each loader returns the same table contract:

``sequence``
    Full protein sequence for CombinGym or the mutable-site sequence for SSMuLA.
``fitness``
    Raw assay value selected by the manifest.
``depth``
    Hamming distance from the library wild type.
``mutations``
    A dictionary from one-based sequence position to mutant amino acid. SSMuLA
    uses its mutable-site sequence, so its positions run from one to library
    width. The source residue numbers remain in ``LandscapeSpec.positions``.

The module reads user-supplied dataset checkouts. It does not download or
redistribute source data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Source = Literal["ssmula", "combingym", "graphfla"]


@dataclass(frozen=True)
class LandscapeSpec:
    """Metadata and schema needed to load one empirical landscape."""

    name: str
    protein: str
    source: Source
    relative_path: str
    fitness_column: str
    sequence_column: str
    depth_column: str | None
    positions: tuple[int, ...] | None = None
    wild_type: str | None = None
    reference_sequence: str | None = None
    phenotype: str | None = None
    source_url: str = ""
    source_revision: str = ""


@dataclass(frozen=True)
class LandscapeDataset:
    """One normalised empirical landscape and its provenance."""

    spec: LandscapeSpec
    variants: pd.DataFrame
    wild_type_sequence: str
    mutable_positions: tuple[int, ...]
    alphabets: dict[int, tuple[str, ...]]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def protein(self) -> str:
        return self.spec.protein

    @property
    def n_variants(self) -> int:
        return len(self.variants)


_SSMULA_REVISION = "4926b8e614d426ca567159988834482c23da13a3"
_COMBINGYM_REVISION = "e635a959731d2a14f90269245ccbec4bf2645324"
_GRAPHFLA_REVISION = "9f86def1a12db76d26af16077ba28cb61ee0af5f"


def _ssmula_spec(
    name: str,
    protein: str,
    positions: tuple[int, ...],
    wild_type: str,
    directory: str | None = None,
) -> LandscapeSpec:
    return LandscapeSpec(
        name=name,
        protein=protein,
        source="ssmula",
        relative_path=f"data/{directory or name}/scale2max/{name}.csv",
        fitness_column="fitness",
        sequence_column="AAs",
        depth_column="n_mut",
        positions=positions,
        wild_type=wild_type,
        source_url="https://zenodo.org/records/15203754",
        source_revision=_SSMULA_REVISION,
    )


SSMULA_MANIFEST: tuple[LandscapeSpec, ...] = (
    _ssmula_spec("ParD2", "ParD", (61, 64, 80), "ILK"),
    _ssmula_spec("ParD3", "ParD", (61, 64, 80), "DKE"),
    _ssmula_spec("GB1", "GB1", (39, 40, 41, 54), "VDGV"),
    _ssmula_spec("DHFR", "DHFR", (26, 27, 28), "ADL"),
    _ssmula_spec("T7", "T7", (748, 756, 758), "NRQ"),
    _ssmula_spec("TEV", "TEV", (146, 148, 167, 170), "TDHS"),
    _ssmula_spec("TrpB3A", "TrpB", (104, 105, 106), "AET", "TrpB"),
    _ssmula_spec("TrpB3B", "TrpB", (105, 106, 107), "ETG", "TrpB"),
    _ssmula_spec("TrpB3C", "TrpB", (106, 107, 108), "TGA", "TrpB"),
    _ssmula_spec("TrpB3D", "TrpB", (117, 118, 119), "TAA", "TrpB"),
    _ssmula_spec("TrpB3E", "TrpB", (184, 185, 186), "FGS", "TrpB"),
    _ssmula_spec("TrpB3F", "TrpB", (162, 166, 301), "LIY", "TrpB"),
    _ssmula_spec("TrpB3G", "TrpB", (227, 228, 301), "VSY", "TrpB"),
    _ssmula_spec("TrpB3H", "TrpB", (228, 230, 231), "SGS", "TrpB"),
    _ssmula_spec("TrpB3I", "TrpB", (182, 183, 184), "YVF", "TrpB"),
    _ssmula_spec("TrpB4", "TrpB", (183, 184, 227, 228), "VFVS", "TrpB"),
)


def _combingym_spec(
    name: str,
    protein: str,
    filename: str,
    fitness_column: str,
    phenotype: str,
    depth_column: str = "n_mut",
) -> LandscapeSpec:
    return LandscapeSpec(
        name=name,
        protein=protein,
        source="combingym",
        relative_path=f"Data/DMS/Clean/{filename}",
        fitness_column=fitness_column,
        sequence_column="genotype",
        depth_column=depth_column,
        phenotype=phenotype,
        source_url="https://github.com/sitonglab/CombinGym",
        source_revision=_COMBINGYM_REVISION,
    )


COMBINGYM_MANIFEST: tuple[LandscapeSpec, ...] = (
    _combingym_spec("GB1_binding", "GB1", "GB1_clean.xlsx", "fitness", "binding"),
    _combingym_spec(
        "CR9114_H1_binding", "CR9114", "bnAbs_CR9114_H1_clean.xlsx", "h1_mean", "binding"
    ),
    _combingym_spec(
        "CR6261_H1_binding", "CR6261", "bnAbs_CR6261_H1_clean.xlsx", "h1_mean", "binding"
    ),
    _combingym_spec(
        "CR6261_H9_binding", "CR6261", "bnAbs_CR6261_H9_clean.xlsx", "h9_mean", "binding"
    ),
    _combingym_spec(
        "CreiLOV_fluorescence",
        "CreiLOV",
        "CreiLOV_clean.xlsx",
        "mean_log",
        "fluorescence",
    ),
    _combingym_spec("eqFP611_blue", "eqFP611", "eqFP611_clean.xlsx", "blue", "fluorescence"),
    _combingym_spec("eqFP611_red", "eqFP611", "eqFP611_clean.xlsx", "red", "fluorescence"),
    _combingym_spec(
        "eqFP611_combined", "eqFP611", "eqFP611_clean.xlsx", "combined", "fluorescence"
    ),
    _combingym_spec("SpCas9_activity", "SpCas9", "SpCas9_clean.xlsx", "mean", "activity"),
    _combingym_spec("SaCas9_activity", "SaCas9", "SaCas9_clean.xlsx", "Mean", "activity"),
    _combingym_spec("HIV1_activity", "HIV-1", "HIV-1_clean.xlsx", "fitness", "activity"),
    _combingym_spec("RhlA_RLs", "RhlA", "RhlA_clean.xlsx", "RLs_log", "activity", "num_mut"),
    _combingym_spec(
        "RhlA_fraction",
        "RhlA",
        "RhlA_clean.xlsx",
        "%Rha-(C8-C10)_log",
        "activity",
        "num_mut",
    ),
    _combingym_spec(
        "RhlA_product",
        "RhlA",
        "RhlA_clean.xlsx",
        "Rha-(C8-C10)_log",
        "activity",
        "num_mut",
    ),
)


def _graphfla_spec(filename: str, protein: str, reference_sequence: str) -> LandscapeSpec:
    return LandscapeSpec(
        name=f"GraphFLA_{Path(filename).stem}",
        protein=protein,
        source="graphfla",
        relative_path=f"data/BioSequence/{filename}",
        fitness_column="fitness",
        sequence_column="sequences",
        depth_column=None,
        reference_sequence=reference_sequence,
        source_url="https://github.com/COLA-Laboratory/GraphFLA",
        source_revision=_GRAPHFLA_REVISION,
    )


# One condition per protein family, selected before the external transfer run.
# Each table is a complete Cartesian landscape with at least 512 variants.
GRAPHFLA_EXTERNAL_MANIFEST: tuple[LandscapeSpec, ...] = (
    _graphfla_spec("Lunzer2005_fitness.csv", "IMDH", "DDFAGR"),
    _graphfla_spec("Michael2024_LamB.csv", "phage-lambda-J", "000000000"),
    _graphfla_spec("Wu2020_Bk79.csv", "H3N2-haemagglutinin", "EEFDFA"),
    _graphfla_spec("Bank2016a.csv", "Hsp90", "QFGFSAAAE"),
    _graphfla_spec("Schulz2025.csv", "COV107-23", "0000000000"),
    _graphfla_spec("Poelwijk2019.csv", "eqFP611", "0000000000000"),
    _graphfla_spec("Moulana2023_S309.csv", "SARS-CoV-2-RBD", "000000000000000"),
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported landscape table: {path}")


def _find_full_wild_type(raw: pd.DataFrame, sequence_column: str, depth_column: str) -> str:
    depths = pd.to_numeric(raw[depth_column], errors="coerce")
    candidates = raw.loc[depths == 0, sequence_column].dropna().astype(str)
    if candidates.empty:
        raise ValueError("dataset has no depth-zero wild-type sequence")
    sequences = candidates.str.rstrip("*").unique()
    if len(sequences) != 1:
        raise ValueError(f"dataset has {len(sequences)} distinct depth-zero sequences")
    return str(sequences[0])


def _normalise_full_sequences(raw: pd.DataFrame, spec: LandscapeSpec) -> LandscapeDataset:
    assert spec.depth_column is not None
    wt = _find_full_wild_type(raw, spec.sequence_column, spec.depth_column)
    rows: list[dict[str, object]] = []

    columns = raw[[spec.sequence_column, spec.fitness_column]]
    for sequence_value, fitness_value in columns.itertuples(index=False, name=None):
        if pd.isna(sequence_value) or pd.isna(fitness_value):
            continue
        sequence = str(sequence_value).rstrip("*")
        if len(sequence) != len(wt):
            continue
        mutations = {
            i: mutant
            for i, (parent, mutant) in enumerate(zip(wt, sequence), start=1)
            if mutant != parent
        }
        rows.append(
            {
                "sequence": sequence,
                "fitness": float(fitness_value),
                "depth": len(mutations),
                "mutations": mutations,
            }
        )

    variants = pd.DataFrame(rows).drop_duplicates("sequence", keep="first").reset_index(drop=True)
    if variants.empty:
        raise ValueError(f"{spec.name} has no finite variants")

    mutable_positions = tuple(
        sorted({position for mutations in variants["mutations"] for position in mutations})
    )
    alphabets = {
        position: tuple(sorted({wt[position - 1], *variants["sequence"].str[position - 1]}))
        for position in mutable_positions
    }
    return LandscapeDataset(spec, variants, wt, mutable_positions, alphabets)


def _normalise_site_sequences(raw: pd.DataFrame, spec: LandscapeSpec) -> LandscapeDataset:
    if spec.positions is None or spec.wild_type is None:
        raise ValueError(f"{spec.name} requires positions and wild_type metadata")

    rows: list[dict[str, object]] = []
    columns = raw[[spec.sequence_column, spec.fitness_column]]
    for sequence_value, fitness_value in columns.itertuples(index=False, name=None):
        if pd.isna(sequence_value) or pd.isna(fitness_value):
            continue
        sequence = str(sequence_value).strip()
        if len(sequence) != len(spec.wild_type) or "*" in sequence:
            continue
        mutations = {
            index: mutant
            for index, (parent, mutant) in enumerate(zip(spec.wild_type, sequence), start=1)
            if mutant != parent
        }
        rows.append(
            {
                "sequence": sequence,
                "fitness": float(fitness_value),
                "depth": len(mutations),
                "mutations": mutations,
            }
        )

    variants = pd.DataFrame(rows).drop_duplicates("sequence", keep="first").reset_index(drop=True)
    if variants.empty:
        raise ValueError(f"{spec.name} has no finite variants")
    alphabets = {
        position: tuple(sorted(variants["sequence"].str[position - 1].unique()))
        for position in range(1, len(spec.positions) + 1)
    }
    mutable_positions = tuple(range(1, len(spec.positions) + 1))
    return LandscapeDataset(spec, variants, spec.wild_type, mutable_positions, alphabets)


def _normalise_graphfla(
    raw: pd.DataFrame, spec: LandscapeSpec, *, require_complete: bool = True
) -> LandscapeDataset:
    """Normalise a GraphFLA table around its declared reference genotype."""
    position_columns = sorted(
        (column for column in raw if column.startswith("pos") and column[3:].isdigit()),
        key=lambda column: int(column[3:]),
    )
    if not position_columns:
        raise ValueError(f"{spec.name} has no position columns")
    if spec.reference_sequence is None:
        raise ValueError(f"{spec.name} has no declared reference sequence")
    if len(spec.reference_sequence) != len(position_columns):
        raise ValueError(f"{spec.name} reference length does not match its position columns")
    finite = raw[position_columns + [spec.fitness_column]].dropna()
    finite = finite.drop_duplicates(position_columns, keep="first").reset_index(drop=True)
    if finite.empty:
        raise ValueError(f"{spec.name} has no finite variants")

    if require_complete:
        expected = 1
        for column in position_columns:
            expected *= int(finite[column].nunique())
        if len(finite) != expected:
            raise ValueError(f"{spec.name} is not a complete Cartesian landscape")

    reference_values: list[object] = []
    reference_mask = pd.Series(True, index=finite.index)
    for column, reference_symbol in zip(position_columns, spec.reference_sequence):
        matches = finite[column].map(str) == reference_symbol
        if not bool(matches.any()):
            raise ValueError(f"{spec.name} reference state is absent at {column}")
        reference_value = finite.loc[matches, column].iloc[0]
        reference_values.append(reference_value)
        reference_mask &= finite[column] == reference_value
    if int(reference_mask.sum()) != 1:
        raise ValueError(f"{spec.name} reference genotype is not unique")

    symbols = "ACDEFGHIKLMNPQRSTVWY"
    encodings: list[dict[object, str]] = []
    kept_columns: list[str] = []
    for column, reference in zip(position_columns, reference_values):
        states = list(pd.unique(finite[column]))
        if len(states) <= 1:
            continue
        if len(states) > len(symbols):
            raise ValueError(f"{spec.name} position {column} has more than 20 states")
        ordered = [reference, *sorted((state for state in states if state != reference), key=str)]
        encodings.append(dict(zip(ordered, symbols)))
        kept_columns.append(column)

    if not kept_columns:
        raise ValueError(f"{spec.name} has no mutable positions")
    wild_type = "A" * len(kept_columns)
    rows: list[dict[str, object]] = []
    for values in finite[kept_columns + [spec.fitness_column]].itertuples(index=False, name=None):
        sequence = "".join(encoding[value] for encoding, value in zip(encodings, values[:-1]))
        mutations = {
            position: residue
            for position, residue in enumerate(sequence, start=1)
            if residue != "A"
        }
        rows.append(
            {
                "sequence": sequence,
                "fitness": float(values[-1]),
                "depth": len(mutations),
                "mutations": mutations,
            }
        )

    variants = pd.DataFrame(rows)
    alphabets: dict[int, tuple[str, ...]] = {
        position: tuple(symbols[: len(encoding)])
        for position, encoding in enumerate(encodings, start=1)
    }
    mutable_positions = tuple(alphabets)
    expected_singles = sum(len(alphabet) - 1 for alphabet in alphabets.values())
    if require_complete and int((variants["depth"] == 1).sum()) != expected_singles:
        raise ValueError(f"{spec.name} reference row lacks a complete single-mutant shell")
    return LandscapeDataset(spec, variants, wild_type, mutable_positions, alphabets)


def load_landscape(
    root: str | Path, spec: LandscapeSpec, *, require_complete: bool = True
) -> LandscapeDataset:
    """Load one landscape from an SSMuLA, CombinGym or GraphFLA checkout.

    ``require_complete`` applies to GraphFLA tables. Setting it to ``False``
    admits a partially measured library, whose missing genotypes simply do not
    appear in the returned variants.
    """
    path = Path(root) / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(f"missing {spec.name} source table: {path}")
    raw = _read_table(path)
    required = {spec.fitness_column}
    if spec.source != "graphfla":
        required.add(spec.sequence_column)
    if spec.depth_column is not None:
        required.add(spec.depth_column)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{spec.name} is missing columns: {', '.join(missing)}")
    if spec.source == "graphfla":
        return _normalise_graphfla(raw, spec, require_complete=require_complete)
    if spec.source == "ssmula":
        return _normalise_site_sequences(raw, spec)
    return _normalise_full_sequences(raw, spec)


def _load_manifest(
    root: str | Path,
    manifest: tuple[LandscapeSpec, ...],
    *,
    strict: bool,
) -> list[LandscapeDataset]:
    datasets: list[LandscapeDataset] = []
    for spec in manifest:
        try:
            datasets.append(load_landscape(root, spec))
        except FileNotFoundError:
            if strict:
                raise
    return datasets


def load_ssmula_landscapes(root: str | Path, *, strict: bool = True) -> list[LandscapeDataset]:
    """Load the sixteen SSMuLA landscapes from a data checkout."""
    return _load_manifest(root, SSMULA_MANIFEST, strict=strict)


def load_combingym_landscapes(root: str | Path, *, strict: bool = True) -> list[LandscapeDataset]:
    """Load the fourteen CombinGym assay landscapes from a checkout."""
    return _load_manifest(root, COMBINGYM_MANIFEST, strict=strict)


def load_graphfla_landscapes(root: str | Path, *, strict: bool = True) -> list[LandscapeDataset]:
    """Load the fixed external GraphFLA validation panel from a checkout."""
    return _load_manifest(root, GRAPHFLA_EXTERNAL_MANIFEST, strict=strict)
