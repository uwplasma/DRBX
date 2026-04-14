from __future__ import annotations

import json

from conftest import REPO_ROOT


PUBLIC_RELEASE_FILES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "native_runtime_cli.md",
    REPO_ROOT / "docs" / "restartable_diffusion_tutorial.md",
    REPO_ROOT / "docs" / "validation_gallery.md",
    REPO_ROOT / "docs" / "physics_models.md",
    REPO_ROOT / "docs" / "research_directions.md",
    REPO_ROOT / "docs" / "tokamak_tcv_x21_scaffold_demo.md",
    REPO_ROOT / "examples" / "alfven_wave_meeting_demo.py",
    REPO_ROOT / "examples" / "blob2d_meeting_demo.py",
    REPO_ROOT / "examples" / "tokamak-3D" / "tcv-x21" / "scaffold_demo.py",
    REPO_ROOT / "src" / "jax_drb" / "validation" / "tokamak_tcv_x21_scaffold.py",
)

PUBLIC_RUN_LOGS = (
    REPO_ROOT / "docs" / "data" / "restartable_diffusion_demo_artifacts" / "run_first" / "restartable_diffusion_run_log.json",
    REPO_ROOT / "docs" / "data" / "restartable_diffusion_demo_artifacts" / "run_full" / "restartable_diffusion_full_run_log.json",
    REPO_ROOT / "docs" / "data" / "restartable_diffusion_demo_artifacts" / "run_resumed" / "restartable_diffusion_resumed_run_log.json",
    REPO_ROOT / "docs" / "data" / "tokamak_tcv_x21_scaffold_artifacts" / "data" / "tokamak_tcv_x21_scaffold_manifest.json",
)


def test_public_release_surface_avoids_local_path_leaks() -> None:
    forbidden = ("/Users/", "rogeriojorge", "local/hermes", "local/jax_drb")
    for path in PUBLIC_RELEASE_FILES:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still contains {needle!r}"


def test_public_release_surface_avoids_legacy_branding_in_user_docs() -> None:
    forbidden = ("Hermes-style", "Hermes-3 input deck", "BOUT++")
    for path in PUBLIC_RELEASE_FILES:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still contains {needle!r}"


def test_committed_demo_run_logs_use_sanitized_paths() -> None:
    for path in PUBLIC_RUN_LOGS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(payload, sort_keys=True)
        assert "/Users/" not in text
        assert payload["run_configuration"]["runtime"]["compilation_cache_dir"].startswith("~/")
