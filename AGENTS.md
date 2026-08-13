# Repository Guidelines

## Project Structure and Module Organisation

The Python package lives in `src/mutascape/`. Core modules cover landscape generation, posterior completion, scoring, and validation.

Tests live in `tests/` and follow the package layout. Prepared protein datasets are under `data/<protein>/`. Put repeatable experiments in `scripts/`.

The React studio lives in `web/`. FastAPI routes live in `src/mutascape/server.py`. Bundled empirical priors live in `src/mutascape/data/priors/`.

## Build, Test, and Development Commands

### Python and validation

- `uv sync --extra dev` installs Python 3.12 dependencies and developer tools.
- `just test` runs the full pytest suite. Pass a target with `just test tests/test_landscape.py`.
- `just lint` checks Python and TypeScript with Ruff and Biome.
- `just fmt` applies Ruff and Biome fixes.

### Web application

- `just web-data` rebuilds `web/data/gb1.json` from prepared GB1 data.
- `just web-build` installs Bun packages and builds the browser bundle.
- `just web-check` lints, type-checks, tests, and builds the studio.
- `just dev` builds the web app and starts FastAPI at `http://localhost:8000`.

Run `./.venv/bin/python scripts/rebuild_reference_outputs.py` when publication reference outputs change.

## Coding Style and Naming Conventions

Use four spaces for Python and two spaces for TypeScript. Ruff targets Python 3.12 with a 100-character line limit. Biome checks `web/src/`.

Use `snake_case` for Python functions and modules. Use `PascalCase` for classes and React components. Use `camelCase` for TypeScript values. Add type hints to public Python APIs.

Do not embed new experiment logic only in notebooks. Move reusable work into package modules or scripts.

## Testing Guidelines

Pytest discovers files named `test_*.py` and functions named `test_*`. Bun discovers files named `*.test.ts` and `*.test.tsx`.

Add focused unit tests for model changes and regression tests for frozen metrics. Run `just test`, `just lint`, and `just web-check` before review. No coverage threshold exists.

## Commit and Pull Request Guidelines

Recent commits use short, plain summaries without a required prefix. Write an imperative subject that names the affected area, such as `fix TrpB reference metric`.

Pull requests should state the research goal, list affected datasets, and report test commands. Link related issues. Include comparison figures for visual changes.

Do not commit local caches, virtual environments, model weights, or untracked generated outputs.
