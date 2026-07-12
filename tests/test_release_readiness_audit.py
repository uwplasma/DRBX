from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    module_path = REPO_ROOT / "scripts" / "audit_release_readiness.py"
    spec = importlib.util.spec_from_file_location("audit_release_readiness", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_readiness_audit_passes_current_tree_without_footprint() -> None:
    audit = _load_audit_module()

    result = audit.run_audit(REPO_ROOT, check_footprint=False)

    assert result["passed"], result["failures"]
    assert result["version"] == "2.0.0.dev0"


def test_release_readiness_audit_ignores_python_version_markers_for_pins() -> None:
    audit = _load_audit_module()

    assert audit._is_unpinned_requirement("tomli; python_version < '3.11'")
    assert audit._is_unpinned_requirement("jax")
    assert not audit._is_unpinned_requirement("jax>=0.4")
