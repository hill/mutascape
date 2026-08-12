"""FastAPI service and static host for the mutascape benchmark oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mutascape.web_models import (
    BenchmarkParameters,
    LandscapeRegistry,
    ModelStore,
    build_custom_landscape,
    build_reference_landscape,
    evaluate_entries,
    parse_custom_source,
    preview_cache_key,
    preview_custom,
    preview_reference,
    reference_catalogue_entries,
    reference_detail,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "web"
LOGO_PATH = REPO_ROOT / "docs" / "logo_ai_image.png"
MAX_REQUEST_BYTES = 25_000_000


def _camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.title() for part in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=_camel, populate_by_name=True)


class ParameterInput(ApiModel):
    noise_sigma: float = Field(default=0.3, ge=0, le=5)
    alpha: float = Field(default=1.5, ge=0, le=10)
    pairwise_sigma: float = Field(default=0, ge=0, le=5)
    combinatorial_upside: float = Field(default=3, ge=1, le=10)
    seed: int = 42

    def domain(self) -> BenchmarkParameters:
        return BenchmarkParameters(
            noise_sigma=self.noise_sigma,
            alpha=self.alpha,
            pairwise_sigma=self.pairwise_sigma,
            combinatorial_upside=self.combinatorial_upside,
            seed=self.seed,
        )


class DraftCreate(ApiModel):
    dms_csv: str = Field(min_length=1)
    wt_fasta: str = Field(min_length=1)
    structure_text: str | None = None
    structure_format: Literal["pdb", "cif", "mmcif"] | None = None
    chain_id: str | None = Field(default=None, max_length=16)
    structure_inference: Literal["none", "esmfold"] = "none"


class PreviewSource(ApiModel):
    kind: Literal["reference", "draft", "model"]
    id: str = Field(min_length=1, max_length=128)


class PreviewSample(ApiModel):
    size: int = Field(default=10_000, ge=1, le=100_000)
    depth_weights: dict[int, Annotated[float, Field(ge=0)]]

    @model_validator(mode="after")
    def check_positive_weight(self) -> PreviewSample:
        if not any(value > 0 for value in self.depth_weights.values()):
            raise ValueError("depthWeights needs at least one positive value")
        return self


class PreviewRequest(ApiModel):
    source: PreviewSource
    parameters: ParameterInput
    sample: PreviewSample


class PublishRequest(ApiModel):
    draft_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=80)
    parameters: ParameterInput


class EvaluationEntry(ApiModel):
    mutation_label: str | None = Field(default=None, max_length=1024)
    compact_sequence: str | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def check_one_input(self) -> EvaluationEntry:
        if bool(self.mutation_label) == bool(self.compact_sequence):
            raise ValueError("provide one mutationLabel or compactSequence")
        return self


class EvaluationRequest(ApiModel):
    entries: list[EvaluationEntry] = Field(min_length=1, max_length=10_000)


class InteractiveEvaluationRequest(ApiModel):
    source: PreviewSource
    parameters: ParameterInput
    entries: list[EvaluationEntry] = Field(min_length=1, max_length=1_000)


class LeaderboardCreate(ApiModel):
    name: str = Field(min_length=1, max_length=80)
    landscape_id: str = Field(min_length=1, max_length=128)
    query_budget: int = Field(ge=1, le=10_000_000)


class LeaderboardEntryCreate(ApiModel):
    model_name: str = Field(min_length=1, max_length=120)
    successes: int = Field(ge=0, le=1_000_000)
    trials: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def check_successes(self) -> LeaderboardEntryCreate:
        if self.successes > self.trials:
            raise ValueError("successes cannot exceed trials")
        return self


def create_app(database_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(
        title="mutascape benchmark oracle",
        version="1.0.0",
        docs_url="/api/docs",
    )
    app.state.model_store = ModelStore(database_path)
    app.state.landscapes = LandscapeRegistry()

    @app.middleware("http")
    async def bound_request_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        try:
            request_size = int(content_length) if content_length is not None else None
        except ValueError:
            return JSONResponse({"detail": "invalid content-length header"}, status_code=400)
        if request_size is not None and request_size > MAX_REQUEST_BYTES:
            return JSONResponse(
                {"detail": "request body must be at most 25 MB"},
                status_code=413,
            )
        return await call_next(request)

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "oracle": True,
            "mode": "benchmark-studio",
            "references": ["gb1_headline", "trpb_headline"],
            "previewMaxVariants": 100_000,
            "evaluationMaxEntries": 10_000,
            "structureInference": ["esmfold"],
        }

    @app.get("/api/v1/models")
    def list_models(request: Request) -> dict[str, Any]:
        store = _store(request)
        custom = [_public_model(model) for model in store.list_models()]
        return {"models": [*reference_catalogue_entries(), *custom]}

    @app.get("/api/v1/models/{model_id}")
    def get_model(model_id: str, request: Request) -> dict[str, Any]:
        reference = reference_detail(model_id)
        if reference is not None:
            return {
                **reference,
                "examples": _examples(
                    model_id,
                    reference["positions"],
                    reference["wtCompact"],
                ),
            }
        model = _store(request).get_model(model_id)
        if model is None:
            raise HTTPException(404, "model not found")
        public = _public_model(model)
        return {
            **public,
            "examples": _examples(
                model_id,
                model["positions"],
                model["wtCompact"],
                model=model,
            ),
        }

    @app.post("/api/v1/drafts", status_code=201)
    def create_draft(payload: DraftCreate, request: Request) -> dict[str, Any]:
        try:
            source, summary = parse_custom_source(
                dms_csv=payload.dms_csv,
                wt_fasta=payload.wt_fasta,
                structure_text=payload.structure_text,
                structure_format=payload.structure_format,
                chain_id=payload.chain_id,
                structure_inference=payload.structure_inference,
                structure_cache_dir=_store(request).path.parent / "structures",
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        uploads = {
            "dmsCsv": payload.dms_csv,
            "wtFasta": payload.wt_fasta,
            "structureText": payload.structure_text,
            "structureFormat": payload.structure_format,
            "chainId": payload.chain_id,
            "structureInference": payload.structure_inference,
        }
        draft = _store(request).create_draft(source, uploads)
        return {
            "id": draft["id"],
            "expiresAt": draft["expiresAt"],
            "summary": summary,
        }

    @app.post("/api/v1/previews")
    def preview(payload: PreviewRequest, request: Request) -> dict[str, Any]:
        store = _store(request)
        parameters = payload.parameters.domain()
        draft_source: dict[str, Any] | None = None
        published_model: dict[str, Any] | None = None
        if payload.source.kind == "reference":
            if reference_detail(payload.source.id) is None:
                raise HTTPException(404, "reference model not found")
        elif payload.source.kind == "draft":
            draft = store.get_draft(payload.source.id)
            if draft is None:
                raise HTTPException(404, "draft not found or expired")
            draft_source = draft["source"]
        else:
            published_model = store.get_model(payload.source.id)
            if published_model is None:
                raise HTTPException(404, "model not found")
            fixed_parameters = BenchmarkParameters(**published_model["parameters"])
            if parameters != fixed_parameters:
                raise HTTPException(422, "published model parameters are fixed")
            parameters = fixed_parameters

        request_data = payload.model_dump(by_alias=True)
        request_data["parameters"] = parameters.as_public_json()
        cache_key = preview_cache_key(request_data)
        cached = store.get_cached_preview(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

        weights = {int(key): value for key, value in payload.sample.depth_weights.items()}
        try:
            if payload.source.kind == "reference":
                result = preview_reference(
                    payload.source.id,
                    parameters,
                    payload.sample.size,
                    weights,
                )
            elif payload.source.kind == "draft":
                assert draft_source is not None
                result = preview_custom(
                    draft_source,
                    parameters,
                    payload.sample.size,
                    weights,
                )
            else:
                assert published_model is not None
                result = preview_custom(
                    published_model["source"],
                    parameters,
                    payload.sample.size,
                    weights,
                    model_id=published_model["id"],
                    name=published_model["name"],
                )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        result["cached"] = False
        store.cache_preview(cache_key, result)
        return result

    @app.post("/api/v1/models", status_code=201)
    def publish(payload: PublishRequest, request: Request) -> JSONResponse:
        parameters = payload.parameters.domain()
        name = payload.name.strip()
        if not name:
            raise HTTPException(422, "name cannot be blank")
        try:
            parameters.validate()
            model, created = _store(request).publish(
                payload.draft_id,
                name,
                parameters,
            )
        except KeyError as exc:
            raise HTTPException(404, "draft not found or expired") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        body = {
            **_public_model(model),
            "created": created,
            "examples": _examples(
                model["id"],
                model["positions"],
                model["wtCompact"],
                model=model,
            ),
        }
        return JSONResponse(body, status_code=201 if created else 200)

    @app.post("/api/v1/models/{model_id}/evaluate")
    def evaluate(
        model_id: str,
        payload: EvaluationRequest,
        request: Request,
    ) -> dict[str, Any]:
        registry: LandscapeRegistry = request.app.state.landscapes
        if reference_detail(model_id) is not None:
            landscape = registry.reference(model_id)
        else:
            model = _store(request).get_model(model_id)
            if model is None:
                raise HTTPException(404, "model not found")
            landscape = registry.custom(model)
        entries = [entry.model_dump(by_alias=True) for entry in payload.entries]
        try:
            results = evaluate_entries(landscape, entries)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"modelId": model_id, "results": results}

    @app.post("/api/v1/evaluations")
    def evaluate_landscape(
        payload: InteractiveEvaluationRequest,
        request: Request,
    ) -> dict[str, Any]:
        store = _store(request)
        parameters = payload.parameters.domain()
        try:
            if payload.source.kind == "reference":
                if reference_detail(payload.source.id) is None:
                    raise HTTPException(404, "reference model not found")
                landscape = build_reference_landscape(payload.source.id, parameters)
            elif payload.source.kind == "draft":
                draft = store.get_draft(payload.source.id)
                if draft is None:
                    raise HTTPException(404, "draft not found or expired")
                landscape = build_custom_landscape(draft["source"], parameters)
            else:
                model = store.get_model(payload.source.id)
                if model is None:
                    raise HTTPException(404, "model not found")
                fixed_parameters = BenchmarkParameters(**model["parameters"])
                if parameters != fixed_parameters:
                    raise HTTPException(422, "published model parameters are fixed")
                landscape = build_custom_landscape(model["source"], fixed_parameters)
            entries = [entry.model_dump(by_alias=True) for entry in payload.entries]
            results = evaluate_entries(landscape, entries)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"source": payload.source.model_dump(), "results": results}

    @app.get("/api/v1/leaderboards")
    def list_leaderboards(request: Request) -> dict[str, Any]:
        return {"leaderboards": _store(request).list_leaderboards()}

    @app.post("/api/v1/leaderboards", status_code=201)
    def create_leaderboard(
        payload: LeaderboardCreate,
        request: Request,
    ) -> dict[str, Any]:
        name = payload.name.strip()
        if not name:
            raise HTTPException(422, "name cannot be blank")
        if reference_detail(payload.landscape_id) is None:
            if _store(request).get_model(payload.landscape_id) is None:
                raise HTTPException(404, "landscape not found")
        return _store(request).create_leaderboard(
            name,
            payload.landscape_id,
            payload.query_budget,
        )

    @app.post("/api/v1/leaderboards/{board_id}/entries", status_code=201)
    def add_leaderboard_entry(
        board_id: str,
        payload: LeaderboardEntryCreate,
        request: Request,
    ) -> dict[str, Any]:
        model_name = payload.model_name.strip()
        if not model_name:
            raise HTTPException(422, "model name cannot be blank")
        try:
            return _store(request).add_leaderboard_entry(
                board_id,
                model_name,
                payload.successes,
                payload.trials,
            )
        except KeyError as exc:
            raise HTTPException(404, "leaderboard not found") from exc

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/landscapes")
    @app.get("/leaderboards")
    def routed_page() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/landscapes/{landscape_id}")
    def landscape_page(landscape_id: str) -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/styles.css")
    def styles() -> FileResponse:
        return FileResponse(WEB_ROOT / "styles.css", media_type="text/css")

    @app.get("/assets/logo.png")
    def logo() -> FileResponse:
        return FileResponse(LOGO_PATH, media_type="image/png")

    app.mount("/dist", StaticFiles(directory=WEB_ROOT / "dist"), name="dist")
    return app


def _store(request: Request) -> ModelStore:
    return request.app.state.model_store


def _public_model(model: dict[str, Any]) -> dict[str, Any]:
    source = model["source"]
    return {
        "id": model["id"],
        "name": model["name"],
        "kind": "custom",
        "description": "Published custom single-site DMS benchmark oracle.",
        "positions": model["positions"],
        "wtCompact": model["wtCompact"],
        "parameters": BenchmarkParameters(**model["parameters"]).as_public_json(),
        "createdAt": model["createdAt"],
        "sourceSummary": {
            "wtLength": len(source["wtSequence"]),
            "measuredRows": len(source["dmsRows"]),
            "hasStructure": source["hasStructure"],
            "structureMethod": source.get(
                "structureMethod",
                "uploaded" if source["hasStructure"] else "none",
            ),
            "contactCount": len(source["contactPairs"]),
            "canFitNonlinearity": source["canFitNonlinearity"],
        },
    }


def _examples(
    model_id: str,
    positions: list[int],
    wt_compact: str,
    model: dict[str, Any] | None = None,
) -> dict[str, str]:
    wildtype = wt_compact[0] if wt_compact else "A"
    mutant = "G" if wildtype == "A" else "A"
    mutation = f"{wildtype}{positions[0]}{mutant}" if positions else "A1G"
    if model_id.endswith("_headline"):
        local = (
            "from mutascape.validation.reference_configs import get_reference_config\n\n"
            f"landscape = get_reference_config({model_id!r}).build_landscape()\n"
            f"fitness = landscape[{mutation!r}]"
        )
    else:
        assert model is not None
        source = model["source"]
        parameters = BenchmarkParameters(**model["parameters"])
        structure_method = source.get(
            "structureMethod",
            "uploaded" if source["hasStructure"] else "none",
        )
        if structure_method == "esmfold":
            contact_setup = (
                'with open("wt.fasta") as fasta:\n'
                '    sequence = "".join(\n'
                '        line.strip() for line in fasta if not line.startswith(">")\n'
                "    )\n"
                "contact_map = ContactMap.from_sequence(sequence)"
            )
        elif structure_method == "uploaded":
            contact_setup = 'contact_map = ContactMap.from_pdb("structure.pdb")'
        else:
            position_tuple = tuple(int(position) for position in positions)
            contact_setup = (
                f"positions = {position_tuple!r}\n"
                "contact_map = ContactMap(\n"
                "    matrix=np.zeros((len(positions), len(positions)), dtype=bool),\n"
                "    positions=positions,\n"
                "    threshold=8.0,\n"
                ")"
            )
        if source["canFitNonlinearity"]:
            readout_setup = (
                'dms = pd.read_csv("singles.csv")\n'
                "single_effects = {}\n"
                "for row in dms.itertuples(index=False):\n"
                "    effects = single_effects.setdefault(\n"
                "        int(row.position), {str(row.wildtype): 0.0}\n"
                "    )\n"
                "    effects[str(row.mutant)] = float(row.fitness)\n"
                "readout = GlobalEpistasisSigmoid.fit_from_singles(\n"
                "    single_effects,\n"
                f"    combinatorial_upside={parameters.combinatorial_upside!r},\n"
                ")"
            )
        else:
            readout_setup = "readout = Identity()"
        local = (
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "from mutascape import Landscape\n"
            "from mutascape.contacts import ContactMap\n"
            "from mutascape.epistasis import RandomEpistasis\n"
            "from mutascape.nonlinearity import GlobalEpistasisSigmoid, Identity\n\n"
            f"{contact_setup}\n\n"
            f"{readout_setup}\n\n"
            "landscape = Landscape.from_dms(\n"
            '    dms_path="singles.csv",\n'
            '    wt_fasta="wt.fasta",\n'
            "    contact_map=contact_map,\n"
            "    epistasis=RandomEpistasis(\n"
            f"        sigma={parameters.pairwise_sigma!r}, seed={parameters.seed!r}\n"
            "    ),\n"
            f"    alpha={parameters.alpha!r},\n"
            f"    noise_sigma={parameters.noise_sigma!r},\n"
            f"    seed={parameters.seed!r},\n"
            "    nonlinearity=readout,\n"
            ")\n"
            f"fitness = landscape[{mutation!r}]"
        )
    hosted = (
        "import requests\n\n"
        'base_url = "https://your-mutascape-host.example"\n'
        f'response = requests.post(\n    f"{{base_url}}/api/v1/models/{model_id}/evaluate",\n'
        f'    json={{"entries": [{{"mutationLabel": {mutation!r}}}]}},\n'
        "    timeout=30,\n"
        ")\n"
        "response.raise_for_status()\n"
        'fitness = response.json()["results"][0]["fitness"]'
    )
    return {"localPython": local, "hostedPython": hosted}


app = create_app()
