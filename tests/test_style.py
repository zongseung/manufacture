import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_paths() -> None:
    import gmst

    assert gmst.ROOT == ROOT
    assert gmst.DATA == ROOT / "5. 자원 최적화 AI 데이터셋"
    assert gmst.RESULTS == ROOT / "results"


def test_dependencies_and_cuda_source() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    names = {re.split(r"[<>=!~\[ ;]", name)[0] for name in config["project"]["dependencies"]}
    assert "lightgbm" in names
    assert not names & {"requests", "scipy", "sklearn", "scikit-learn", "pandas", "pytest"}
    assert any(name.startswith("pytest") for name in config["dependency-groups"]["dev"])
    assert config["tool"]["pytest"]["ini_options"] == {"pythonpath": ["."], "testpaths": ["tests"]}
    assert config["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cu126"}
    assert config["tool"]["uv"]["index"][0]["url"] == "https://download.pytorch.org/whl/cu126"


def test_production_code_has_no_banned_imports_or_credentials() -> None:
    for path in (ROOT / "gmst").rglob("*.py"):
        source = path.read_text()
        assert not re.search(r"\.env\b", source), path
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
                assert not set(names) & {"scipy", "sklearn", "pandas", "requests"}, path
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"scipy", "sklearn", "pandas", "requests"}, path
            if isinstance(node, ast.Call) and path.name != "run_all.py":
                assert not any(keyword.arg == "unseal" and isinstance(keyword.value, ast.Constant)
                               and keyword.value.value is True for keyword in node.keywords), path


def test_ponytail_comments_have_ceiling_and_trigger() -> None:
    for folder in ("gmst", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            for line in path.read_text().splitlines():
                if re.search(r"#\s*ponytail:", line):
                    assert re.search(r"# ?ponytail:\s*[^,]+,\s*\S+", line), (path, line)
