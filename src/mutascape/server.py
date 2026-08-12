"""FastAPI oracle for the legacy alpha web viz at /web/.

Exposes POST /api/score so the browser can ask the real mutascape Landscape
for fitness predictions instead of running a random-coupling model in JS.
Also serves the static frontend so the whole app runs on a single port.

This demo intentionally keeps the old alpha/pairwise UI model for interactive
visualization. It is not the frozen publication reference pipeline; use
mutascape.validation.reference_configs for paper-grade outputs.

Usage
    uv run uvicorn mutascape.server:app --reload --port 8000
    # then browse http://localhost:8000/
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mutascape.contacts import ContactMap
from mutascape.epistasis import RandomEpistasis
from mutascape.landscape import Landscape

REPO_ROOT = Path(__file__).resolve().parents[2]
GB1_DIR = REPO_ROOT / "data" / "gb1"
WEB_ROOT = REPO_ROOT / "web"


def _load_gb1_wt() -> str:
    lines = (GB1_DIR / "gb1_wt.fasta").read_text().splitlines()
    return "".join(line for line in lines if not line.startswith(">"))


GB1_WT = _load_gb1_wt()


class Site(BaseModel):
    position: int
    altAA: str = Field(min_length=1, max_length=1)


class ScoreRequest(BaseModel):
    wt: str
    sites: list[Site]
    alpha: float = 1.0
    seed: int = 42
    masks: list[int] | None = None


class ScoreEntry(BaseModel):
    mask: int
    fitness: float


class ScoreResponse(BaseModel):
    backend: str
    scores: list[ScoreEntry]


def _gb1_contacts() -> ContactMap:
    data = np.load(GB1_DIR / "gb1_distances.npz")
    return ContactMap.from_distance_matrix(
        distances=data["distances"], positions=data["positions"].tolist(), threshold=8.0
    )


@lru_cache(maxsize=16)
def _gb1_landscape(alpha: float, seed: int) -> Landscape:
    return Landscape.from_dms(
        dms_path=GB1_DIR / "gb1.csv",
        contact_map=_gb1_contacts(),
        epistasis=RandomEpistasis(sigma=0.3, seed=seed),
        alpha=alpha,
        seed=seed,
        wt_fasta=GB1_DIR / "gb1_wt.fasta",
    )


def _synthetic_landscape(wt: str, site_positions: list[int], alpha: float, seed: int) -> Landscape:
    """For pasted FASTAs we have no DMS and no structure. Build a tiny
    landscape with random single-site effects at the selected positions and
    a uniform all-contacts map over those same positions.
    """
    rng = np.random.default_rng(seed)
    wild_type: dict[int, str] = {p: wt[p - 1] for p in site_positions}
    singles: dict[int, dict[str, float]] = {}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for pos in site_positions:
        wt_aa = wt[pos - 1]
        effects = {wt_aa: 0.0}
        for aa in alphabet:
            if aa == wt_aa:
                continue
            effects[aa] = float(rng.normal(0, 1))
        singles[pos] = effects

    n = len(site_positions)
    contact_matrix = np.ones((n, n), dtype=bool)
    np.fill_diagonal(contact_matrix, False)
    contacts = ContactMap(
        matrix=contact_matrix,
        positions=tuple(site_positions),
        threshold=float("inf"),
        distances=None,
    )
    epistasis = RandomEpistasis(sigma=0.3, seed=seed)
    couplings = epistasis.generate(contact_map=contacts, single_site_effects=singles)
    return Landscape(
        wild_type=wild_type,
        single_site_effects=singles,
        contact_map=contacts,
        couplings=couplings,
        alpha=alpha,
        seed=seed,
        wt_sequence=wt,
    )


def _score_variants(landscape: Landscape, sites: list[Site], masks: list[int]) -> list[float]:
    scores: list[float] = []
    for m in masks:
        mutations: dict[int, str] = {}
        for i, s in enumerate(sites):
            if (m >> i) & 1:
                mutations[s.position] = s.altAA
        scores.append(float(landscape.evaluate(mutations)))
    return scores


app = FastAPI(title="mutascape oracle", docs_url="/api/docs")


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "oracle": True,
        "mode": "legacy-alpha-demo",
        "datasets": ["gb1"],
        "hard_max_k": 20,
    }


@app.post("/api/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    k = len(req.sites)
    if k < 1 or k > 20:
        raise HTTPException(400, f"k must be in [1, 20], got {k}")
    if not req.wt:
        raise HTTPException(400, "wt sequence required")
    for s in req.sites:
        if not (1 <= s.position <= len(req.wt)):
            raise HTTPException(
                400, f"position {s.position} out of range for WT of length {len(req.wt)}"
            )
        if req.wt[s.position - 1] == s.altAA:
            raise HTTPException(
                400, f"altAA at {s.position} matches WT; pick a different amino acid"
            )

    masks = req.masks if req.masks is not None else list(range(1 << k))

    if req.wt == GB1_WT:
        backend = "gb1-legacy-alpha"
        lscape = _gb1_landscape(req.alpha, req.seed)
    else:
        backend = "synthetic-legacy-alpha"
        lscape = _synthetic_landscape(req.wt, [s.position for s in req.sites], req.alpha, req.seed)

    scores = _score_variants(lscape, req.sites, masks)
    return ScoreResponse(
        backend=backend,
        scores=[ScoreEntry(mask=m, fitness=f) for m, f in zip(masks, scores)],
    )


# Static frontend --------------------------------------------------------


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(WEB_ROOT / "styles.css", media_type="text/css")


@app.get("/data/{name}")
def data_file(name: str) -> FileResponse:
    path = WEB_ROOT / "data" / name
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="application/json")


app.mount("/dist", StaticFiles(directory=WEB_ROOT / "dist"), name="dist")
