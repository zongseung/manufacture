"""Display-only notebook contract, checked without reading September targets."""

import ast
import json
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("name", ["01_preprocess.ipynb", "02_eda.ipynb", "03_results.ipynb"])
def test_notebook_has_only_read_and_display_operations(name: str) -> None:
    # Given a delivered notebook, when parsing code, then no mutation/training APIs exist.
    notebook = json.loads((NB / name).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(source)
    forbidden = {
        "write_csv",
        "write_json",
        "write_text",
        "write_bytes",
        "write_parquet",
        "savefig",
        "fit",
        "train",
        "dump",
        "to_csv",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(module.split(".")[0] in {"gmst", "torch", "lightgbm"} for module in modules)
    assert source
