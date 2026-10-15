default:
    @just --list

[group('setup')]
sync:
    uv sync --extra dev --extra web

[group('check')]
test *ARGS:
    uv run pytest {{ARGS}}

[group('check')]
typecheck:
    uv run ty check

[group('check')]
lint: typecheck
    uv run ruff check .
    uv run ruff format --check .
    cd web && bun run lint

# Formatting must finish even when lint findings remain, so the fix pass cannot
# abort the run.
[group('check')]
fmt:
    -uv run ruff check --fix .
    uv run ruff format .
    cd web && bun run format

# Download pinned SSMuLA and CombinGym sources outside Git.
[group('data')]
benchmark-data *ARGS:
    uv run python scripts/fetch_benchmark_data.py {{ARGS}}

# Run grouped sparse-to-complete evaluation.
[group('data')]
benchmark-completion *ARGS:
    uv run python scripts/run_completion_benchmark.py {{ARGS}}

# Rebuild figures, outputs, and API facts embedded in the documentation site.
[group('data')]
docs-assets:
    uv run python scripts/build_docs_assets.py

# Install frontend deps (bun) and build the TypeScript bundle into web/dist/.
[group('web')]
web-build:
    cd web && bun install --silent && bun run build

# Check, test, and build the web application.
[group('web')]
web-check:
    uv run python scripts/build_docs_assets.py --check
    cd web && bun install --silent && bun run lint && bun run typecheck && bun test && bun run build

# Frontend-only dev server with client-side synthetic couplings.
[group('web')]
web-dev:
    cd web && bun install --silent && bun run dev

# Run FastAPI and serve the built studio.
# Rebuild the frontend first so web/dist/ exists.
[group('web')]
api-dev: web-build
    uv run uvicorn web.server:app --reload --port 8000

# Full stack dev (alias). http://localhost:8000
[group('web')]
dev: api-dev
