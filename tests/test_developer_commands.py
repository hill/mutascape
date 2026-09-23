"""Keep developer commands linked to existing scripts."""

import re
from pathlib import Path


def test_justfile_python_scripts_exist():
    root = Path(__file__).parents[1]
    recipe = (root / "justfile").read_text()
    scripts = re.findall(r"uv run python (scripts/[^\s]+\.py)", recipe)
    assert scripts
    for script in scripts:
        assert (root / script).is_file(), f"developer command points to absent script {script}"
