"""Load frozen validation results bundled with Mutascape."""

import json
from pathlib import Path
from typing import Any

VALIDATION_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "validation"


def load_benchmark_validity_summary() -> dict[str, Any]:
    """Return the frozen task-transfer result used by the studio."""
    path = VALIDATION_DATA_ROOT / "benchmark-validity-v1.json"
    return json.loads(path.read_text())
