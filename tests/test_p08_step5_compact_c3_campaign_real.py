"""Step-5 compact_c3 re-freeze campaign on real HSX data, bounded (slow): the campaign's closure ``preflight_grid`` at N32
with the four pinned operator options (autodiff K, q2 faces, fixed_radius inner support, ``compact_c3`` magnetic
evaluator): every host-vs-JAX policy row passes, no ``uniq_mismatches``, the environment's reference carries the compact
C3 evaluator. The ``compare_to_oracle`` rows are informational (the frozen oracles describe the old spline operator).

Needs the HSX N32 geometry / sidecar and the frozen campaigns' N32 oracle arrays (skipped otherwise). A few minutes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
CAMPAIGNS = ("p05", "p05n_frozen", "p05n_upwind", "p06n", "p06_legacy", "p07", "p07n")
FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
         "bfield_toroidal": "compact_c3"}

pytestmark = pytest.mark.slow

_have = (GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and SIDECAR.is_file()
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


@needs_inputs
def test_campaign_closure_preflight_passes_at_n32_with_the_compact_c3_evaluator(tmp_path):
    from p_shared import owner_closure as oc
    from p_shared.replay_support import DEFAULT_PATHS
    from p08_step5_compact_c3 import campaign

    paths = dict(DEFAULT_PATHS)
    if not oc.oracle_available(paths, CAMPAIGNS, n=N):
        pytest.skip("the frozen campaigns' N32 oracle arrays are unavailable")
    case = campaign.preflight_grid(n=N, input_root=WORKSPACE, sidecar=SIDECAR, output=tmp_path, paths=paths)
    assert case["all_pass"] is True, (case["policy_failures"], case["uniq_mismatches"], case["options_recorded"],
                                      case["bfield"])
    assert case["operator_options"] == FINAL and case["options_recorded"] is True
    assert case["bfield"]["ok"] is True and case["bfield"]["evaluator_toroidal_method"] == "compact_c3"
    assert case["jax"]["backend"] == "cpu" and case["jax"]["x64"] is True and case["policy_rows"] == 29
    report = json.loads(Path(case["closure_report"]).read_text())
    assert (report["curvature"], report["face_quadrature"], report["inner_support"], report["bfield_toroidal"]) == \
           ("autodiff", "q2", "fixed_radius", "compact_c3")
    print("\ncompact_c3 closure preflight N32:", {k: case[k] for k in ("policy_rows", "peak_rss_gib", "seconds")})
