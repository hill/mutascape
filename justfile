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
