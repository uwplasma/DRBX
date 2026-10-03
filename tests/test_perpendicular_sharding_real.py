"""eta-sharded combined perpendicular RHS on the real N32 owner closure (P08 step 6, stage A; slow).

A bounded closure whose target owners cover every eta plane (two owners per plane on an interior ring and on the wall
ring, with the pinned operator options: autodiff K, q2 faces, fixed_radius inner support, ``compact_c3`` magnetic
evaluator) is lowered into a ``PerpendicularPlan`` by the step-3 gate machinery. In a subprocess with four forced host
devices ``perpendicular_rhs`` on the plan (single device) is compared with ``sharded_perpendicular_rhs`` over 1, 2 and 4
shards for the P06N variants ``main_phi_dirichlet`` and ``main_phi_neumann``: every term of every field at the closure's
target owners agrees to 1e-12 of the single-device maximum (the gate of the design; only the summation order of the
scatter-adds may differ). Needs the HSX N32 geometry / sidecar (skipped otherwise); the P06N owner values are recomputed on the fly with the frozen
routine (``tests/p06n_owner_values_live.py``) as the saved oracle file was removed; about a minute.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
GATE = 1.0e-12
CASE = Path(__file__).resolve().with_name("perpendicular_sharding_real_case.py")

pytestmark = pytest.mark.slow

_have = (GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and SIDECAR.is_file()
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


@needs_inputs
def test_sharded_rhs_equals_single_device_on_the_real_closure_covering_every_plane():
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=4".strip()
    env["JAX_PLATFORMS"] = "cpu"
    done = subprocess.run([sys.executable, str(CASE)], capture_output=True, text=True, env=env, timeout=1800,
                          check=False)
    assert done.returncode == 0, f"subprocess failed ({done.returncode}):\n{done.stderr[-4000:]}"
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["devices"] == 4 and report["owners"] >= 2 * N
    keys = {f"{variant}/Sz{sz}" for variant in ("main_phi_dirichlet", "main_phi_neumann") for sz in (1, 2, 4)}
    assert set(report["results"]) == keys
    for key, entry in report["results"].items():
        assert entry["finite_targets"], key
        bad = {name: m for name, m in entry["fields"].items() if not m["rel"] <= GATE}
        assert not bad, (key, bad)
        assert all(m["scale"] > 0 for m in entry["fields"].values()), key              # no vacuous (all-zero) term
    print(f"\nsharded vs single RHS, N32 closure of {report['owners']} owners: worst rel "
          f"{max(e['worst_rel'] for e in report['results'].values()):.2e}; setup {report['setup_seconds']:.0f} s, "
          f"total {report['seconds']:.0f} s, peak RSS {report['peak_rss_gib']:.2f} GiB")
