default:
    @just --list

sync:
    uv sync --extra dev

test *ARGS:
    uv run pytest {{ARGS}}

lint:
    uv run ruff check src tests

fmt:
    uv run ruff check --fix src tests
    uv run ruff format src tests

# Download pinned SSMuLA and CombinGym sources outside Git.
benchmark-data *ARGS:
    uv run python scripts/fetch_benchmark_data.py {{ARGS}}

# Run grouped sparse-to-complete evaluation.
benchmark-completion *ARGS:
    uv run python scripts/run_completion_benchmark.py {{ARGS}}

# Export aggregate benchmark figures for a manuscript.
benchmark-figures *ARGS:
    uv run python scripts/export_completion_figures.py {{ARGS}}

# Regenerate web/data/gb1.json from the GB1 CSVs (run after updating data/gb1/).
web-data:
    uv run python -m mutascape.viz_export

# Install frontend deps (bun) and build the TypeScript bundle into web/dist/.
web-build:
    cd web && bun install --silent && bun run build

# Frontend-only dev server (Bun). No oracle; client-side synthetic couplings.
web-dev:
    cd web && bun install --silent && bun run dev

# Run the FastAPI oracle. Serves both the frontend and POST /api/score.
# Rebuild the frontend first so web/dist/ exists.
api-dev: web-build
    uv run uvicorn mutascape.server:app --reload --port 8000

# Full stack dev (alias). http://localhost:8000
dev: api-dev
