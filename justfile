default:
    @just --list

sync:
    uv sync --extra dev --extra web

test *ARGS:
    uv run pytest {{ARGS}}

lint:
    uv run ruff check .
    uv run ruff format --check .
    cd web && bun run lint

fmt:
    uv run ruff check --fix .
    uv run ruff format .
    cd web && bun run format

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

# Check, test, and build the web application.
web-check:
    uv run python scripts/build_docs_assets.py --check
    cd web && bun install --silent && bun run lint && bun run typecheck && bun test && bun run build

# Rebuild figures, outputs, and API facts embedded in the documentation site.
docs-assets:
    uv run python scripts/build_docs_assets.py

# Frontend-only dev server with client-side synthetic couplings.
web-dev:
    cd web && bun install --silent && bun run dev

# Run FastAPI and serve the built studio.
# Rebuild the frontend first so web/dist/ exists.
api-dev: web-build
    uv run uvicorn web.server:app --reload --port 8000

# Full stack dev (alias). http://localhost:8000
dev: api-dev
