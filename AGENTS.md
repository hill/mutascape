# Repository Guidelines

## Project Structure and Module Organisation

The Python package lives in `src/mutascape/`. Core modules cover landscapes, mutants, contacts, scoring, epistasis, and nonlinearity. Publication metrics and frozen GB1 and TrpB configurations live in `src/mutascape/validation/`.

Tests live in `tests/` and follow the package layout. Prepared protein datasets are under `data/<protein>/`. Keep exploratory work in `notebooks/`, repeatable experiments in `scripts/`, and generated figures in the relevant output directory. The TypeScript visualisation and FastAPI-backed interface live in `web/` and `src/mutascape/server.py`.

## Build, Test, and Development Commands

### Python and validation

- `uv sync --extra dev` installs Python 3.12 dependencies and developer tools.
- `just test` runs the full pytest suite. Pass a target with `just test tests/test_landscape.py`.
- `just lint` checks `src/` and `tests/` with Ruff.
- `just fmt` applies Ruff fixes and formatting.

### Web application

- `just web-data` rebuilds `web/data/gb1.json` from prepared GB1 data.
- `just web-build` installs Bun packages and builds the browser bundle.
- `just dev` builds the web app and starts FastAPI at `http://localhost:8000`.

Run `./.venv/bin/python scripts/rebuild_reference_outputs.py` when publication reference outputs change.

## Coding Style and Naming Conventions

Use four spaces for Python indentation. Ruff targets Python 3.12 with a 100-character line limit. Use `snake_case` for functions and modules, `PascalCase` for classes, and descriptive scientific names for parameters. Add type hints to public APIs. Keep TypeScript strict and use `camelCase` for functions and variables.

Do not embed new experiment logic only in notebooks. Move reusable work into package modules or scripts.

## Testing Guidelines

Pytest discovers files named `test_*.py` and functions named `test_*`. Add focused unit tests for model changes and regression tests for frozen metrics. Changes to reference configurations must preserve documented tolerance bands or explain and update them. Run `just test` and `just lint` before opening a pull request. No formal coverage threshold exists.

## Commit and Pull Request Guidelines

Recent commits use short, plain summaries without a required prefix. Write an imperative subject that names the affected area, such as `fix TrpB reference metric`.

Pull requests should state the research or software goal, list affected datasets or configurations, and report test commands. Link related issues. Include before-and-after figures for visualisation or publication-output changes. Do not commit local caches, virtual environments, model weights, or generated output unless the repository already tracks that output class.
