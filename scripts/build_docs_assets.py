"""Build the figures and API facts embedded in the documentation site."""

import argparse
import inspect
import json
import re
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mutascape import (
    Campaign,
    CampaignResult,
    DMSLandscape,
    MutationDomain,
    Oracle,
    from_dms,
    load_oracle,
    run_campaign,
)
from mutascape.api import LandscapeGenerator

ROOT = Path(__file__).resolve().parents[1]
CREILOV_DMS = ROOT / "data" / "creilov" / "creilov.csv"
CREILOV_FASTA = ROOT / "data" / "creilov" / "creilov_wt.fasta"
ASSET_DIR = ROOT / "web" / "assets" / "docs"
DATA_PATH = ROOT / "web" / "data" / "docs.generated.json"

GREEN = "#27ae60"
DARK = "#2d2d35"
GREY = "#9a9ba2"
GRID = "#e5e5e8"


def random_search(campaign: Campaign) -> None:
    """Spend each round on a uniform sample of unmeasured candidates."""
    while campaign.remaining:
        size = min(campaign.batch_size, campaign.remaining)
        campaign.measure(campaign.sample(size, seed=100 + campaign.round))


def _paragraphs(value: object) -> list[str]:
    doc = inspect.getdoc(value) or ""
    blocks = [block.strip() for block in doc.split("\n\n")]
    return [block for block in blocks if block]


# Annotations render as `mutascape.model.DMSLandscape` and similar. The module
# path is noise in a reference where every name is already an export.
_QUALIFIED = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)+([A-Za-z_][A-Za-z0-9_]*)\b")

# `NDArray[np.float64]` expands to its two-argument form, and the shape argument says
# nothing a reader of this reference needs.
_NDARRAY = re.compile(r"ndarray\[tuple\[Any, \.\.\.\], dtype\[([A-Za-z0-9_]+)\]\]")


def _short_type(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return ""
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return _NDARRAY.sub(r"NDArray[\1]", _QUALIFIED.sub(r"\1", text))


def _parameters(signature: inspect.Signature) -> list[dict[str, str | bool]]:
    rows: list[dict[str, str | bool]] = []
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        rows.append(
            {
                "name": parameter.name,
                "annotation": _short_type(parameter.annotation),
                "default": (
                    "" if parameter.default is inspect.Parameter.empty else repr(parameter.default)
                ),
                "keywordOnly": parameter.kind is inspect.Parameter.KEYWORD_ONLY,
            }
        )
    return rows


def _prose(text: str) -> str:
    """Normalise one docstring paragraph for the site.

    RST marks code with double backticks. The site renders single backticks, so the pair
    would otherwise show up as literal characters around every name.
    """
    return " ".join(line.strip() for line in text.splitlines()).replace("``", "`")


def _note(block: str) -> dict[str, object]:
    lines = [line.strip() for line in block.splitlines()]
    if all(line.startswith("- ") for line in lines):
        return {"kind": "list", "items": [_prose(line[2:]) for line in lines]}
    return {"kind": "text", "text": _prose(block)}


def _render_signature(name: str, signature: inspect.Signature) -> str:
    parts: list[str] = []
    starred = False
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            starred = True
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not starred:
            parts.append("*")
            starred = True
        prefix = {
            inspect.Parameter.VAR_POSITIONAL: "*",
            inspect.Parameter.VAR_KEYWORD: "**",
        }.get(parameter.kind, "")
        text = f"{prefix}{parameter.name}"
        annotation = _short_type(parameter.annotation)
        if annotation:
            text += f": {annotation}"
        if parameter.default is not inspect.Parameter.empty:
            text += f" = {parameter.default!r}"
        parts.append(text)
    returns = _short_type(signature.return_annotation)
    return f"{name}({', '.join(parts)})" + (f" -> {returns}" if returns else "")


def _entry(name: str, value: object) -> dict[str, object]:
    signature = inspect.signature(value)
    blocks = _paragraphs(value)
    return {
        "name": name,
        "signature": _render_signature(name, signature),
        "summary": _prose(blocks[0]) if blocks else "",
        "notes": [_note(block) for block in blocks[1:]],
        "parameters": _parameters(signature),
        "returns": _short_type(signature.return_annotation),
    }


def _api_reference() -> list[dict[str, object]]:
    groups: list[tuple[str, list[tuple[str, object]]]] = [
        (
            "Load data",
            [
                ("from_dms", from_dms),
                ("load_oracle", load_oracle),
            ],
        ),
        (
            "Sample an oracle",
            [
                ("LandscapeGenerator.draw", LandscapeGenerator.draw),
                ("MutationDomain.count", MutationDomain.count),
                ("MutationDomain.sample", MutationDomain.sample),
            ],
        ),
        (
            "Query an oracle",
            [
                ("Oracle.score_many", Oracle.score_many),
                ("Oracle.evaluate", Oracle.evaluate),
            ],
        ),
        (
            "Run a campaign",
            [
                ("run_campaign", run_campaign),
                ("Campaign.sample", Campaign.sample),
                ("Campaign.measure", Campaign.measure),
                ("Campaign.encode", Campaign.encode),
                ("CampaignResult.to_dict", CampaignResult.to_dict),
            ],
        ),
        (
            "Advanced construction",
            [("DMSLandscape.from_singles", DMSLandscape.from_singles)],
        ),
    ]
    return [
        {"title": title, "entries": [_entry(name, value) for name, value in members]}
        for title, members in groups
    ]


def _make_generator() -> LandscapeGenerator:
    return from_dms(
        CREILOV_DMS,
        fasta=CREILOV_FASTA,
        maximum_depth=4,
    )


def _profile_data(generator: LandscapeGenerator) -> dict[str, object]:
    oracle = generator.draw(seed=42)
    baseline = generator.draw(seed=42, strength=0.0)
    depths: dict[str, dict[str, list[float]]] = {}
    for depth in range(1, 5):
        variants = generator.domain.sample(depth=depth, n=160, seed=depth)
        depths[str(depth)] = {
            "baseline": baseline.evaluate_batch(variants).tolist(),
            "oracle": oracle.evaluate_batch(variants).tolist(),
        }
    labels = ["WT", "A29H", "R60D", "A29H+R60D"]
    return {
        "depths": depths,
        "labels": labels,
        "scores": [float(value) for value in oracle.score_many(labels)],
        "domain": repr(generator.domain),
        "domainSize": generator.domain.count(),
        "seed": 42,
    }


def _campaign_data(generator: LandscapeGenerator) -> dict[str, object]:
    oracle = generator.draw(seed=42)
    candidates = [
        variant
        for depth in range(1, 5)
        for variant in generator.domain.sample(depth=depth, n=250, seed=20 + depth)
    ]
    result = run_campaign(
        random_search,
        oracle,
        candidates,
        budget=192,
        batch_size=48,
        seed=7,
        noise=0.0,
    )
    history = result.history
    return {
        "budget": 192,
        "batchSize": 48,
        "candidateCount": len(candidates),
        "bestFound": result.best_found,
        "optimum": result.optimum,
        "simpleRegret": result.simple_regret,
        "queries": result.queries,
        "rounds": result.rounds,
        "fitness": history["fitness"].astype(float).tolist(),
    }


def _plot_profile(data: dict[str, object], path: Path) -> None:
    depths = data["depths"]
    assert isinstance(depths, dict)
    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    positions = np.arange(1, 5, dtype=float)
    for offset, key, colour, label in [
        (-0.16, "baseline", GREY, "Single-effect baseline"),
        (0.16, "oracle", GREEN, "Fixed synthetic oracle"),
    ]:
        values = [depths[str(depth)][key] for depth in range(1, 5)]
        artists = axis.boxplot(
            values,
            positions=positions + offset,
            widths=0.26,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": DARK, "linewidth": 1.4},
            whiskerprops={"color": colour, "linewidth": 1.0},
            capprops={"color": colour, "linewidth": 1.0},
            boxprops={"facecolor": colour, "edgecolor": colour, "alpha": 0.72},
        )
        artists["boxes"][0].set_label(label)
    axis.axhline(0, color=DARK, linewidth=0.8, alpha=0.45)
    axis.set(
        xlabel="Mutation depth",
        ylabel="Fitness relative to wild type",
        xticks=positions,
        xticklabels=["1", "2", "3", "4"],
    )
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(frameon=False, loc="lower left", ncols=2)
    _finish_figure(fig, path)


def _plot_campaign(data: dict[str, object], path: Path) -> None:
    fitness = np.asarray(data["fitness"], dtype=float)
    queries = np.arange(1, fitness.size + 1)
    best = np.maximum.accumulate(fitness)
    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    axis.step(queries, best, where="post", color=GREEN, linewidth=2.2, label="Best found")
    axis.axhline(
        float(data["optimum"]),
        color=DARK,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label="Best candidate",
    )
    for boundary in range(int(data["batchSize"]), int(data["budget"]), int(data["batchSize"])):
        axis.axvline(boundary, color=GRID, linewidth=0.8)
    axis.set(
        xlabel="Queries spent",
        ylabel="Best fitness found",
        xlim=(1, int(data["budget"])),
    )
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(frameon=False, loc="lower right")
    _finish_figure(fig, path)


def _finish_figure(fig: Figure, path: Path) -> None:
    fig.patch.set_facecolor("white")
    for axis in fig.axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#c8c9ce")
        axis.tick_params(colors="#55565d")
        axis.xaxis.label.set_color(DARK)
        axis.yaxis.label.set_color(DARK)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def build(asset_dir: Path, data_path: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "mutascape-docs-v1",
        }
    )
    generator = _make_generator()
    profile = _profile_data(generator)
    campaign = _campaign_data(generator)
    _plot_profile(profile, asset_dir / "oracle-profile.svg")
    _plot_campaign(campaign, asset_dir / "campaign-trace.svg")
    c2st = _plot_c2st(asset_dir / "creilov-c2st.svg")
    payload = {
        "api": _api_reference(),
        "oracle": {key: value for key, value in profile.items() if key != "depths"},
        "campaign": {key: value for key, value in campaign.items() if key != "fitness"},
        "c2st": c2st,
    }
    data_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def check() -> None:
    with tempfile.TemporaryDirectory(prefix="mutascape-docs-") as directory:
        root = Path(directory)
        generated_assets = root / "assets"
        generated_data = root / "docs.generated.json"
        build(generated_assets, generated_data)
        expected = {
            ASSET_DIR / "oracle-profile.svg": generated_assets / "oracle-profile.svg",
            ASSET_DIR / "campaign-trace.svg": generated_assets / "campaign-trace.svg",
            ASSET_DIR / "creilov-c2st.svg": generated_assets / "creilov-c2st.svg",
            DATA_PATH: generated_data,
        }
        stale = [
            str(target.relative_to(ROOT))
            for target, source in expected.items()
            if not target.exists() or target.read_bytes() != source.read_bytes()
        ]
        if stale:
            joined = ", ".join(stale)
            raise SystemExit(f"documentation assets are stale: {joined}; run `just docs-assets`")


def _plot_c2st(path: Path) -> list[dict]:
    frame = pd.read_csv(ROOT / "results/creilov_c2st/metrics.csv")
    pairs = [("rep1", "rep2"), ("rep1", "rep3"), ("rep2", "rep3"), ("measured", "synthetic_0")]
    fig, axis = plt.subplots(figsize=(8.4, 3.8))
    summaries = []
    for offset, classifier, colour in (
        (-0.1, "logistic", "#555555"),
        (0.1, "histogram", "#6095b0"),
    ):
        for index, (left, right) in enumerate(pairs):
            values = frame.loc[
                (frame.left == left) & (frame.right == right) & (frame.classifier == classifier),
                "auc",
            ]
            middle = float(values.median())
            axis.errorbar(
                middle,
                index + offset,
                xerr=[[middle - values.min()], [values.max() - middle]],
                fmt="o",
                color=colour,
                capsize=3,
                label=classifier if index == 0 else None,
            )
            summaries.append(
                {"left": left, "right": right, "classifier": classifier, "median_auc": middle}
            )
    axis.axvline(0.5, color="#aaaaaa", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Held-out classifier AUC",
        xlim=(0.47, 1.02),
        yticks=range(4),
        yticklabels=[
            "Replicate 1 / 2",
            "Replicate 1 / 3",
            "Replicate 2 / 3",
            "Measured / synthetic",
        ],
    )
    axis.invert_yaxis()
    axis.legend(frameon=False, loc="lower left")
    _finish_figure(fig, path)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when tracked assets are stale")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        build(ASSET_DIR, DATA_PATH)


if __name__ == "__main__":
    main()
