"""eta-sharded potential solve on the real N32 / N48 P07 Dirichlet exports (P08 step 6, stage B; slow).

The local step-5 exports (``work/p08-step5-export-.../N{32,48}/p07_dirichlet.npz`` with ``owner_map.npz``; its
``raw_to_owner`` equals ``perpendicular_structured.reconstruction.load_context(n, WORKSPACE).ro``) carry the real
Dirichlet operator. A manufactured problem (smooth ``x_true`` in ring / theta / plane, smooth Dirichlet data ``g``,
``rhs = A x_true + B g``) and a rough one (``rhs = A (white noise)``) are solved with ``rtol = 1e-10`` by the single-device
``solve_phi`` and eta-sharded (N32: 1, 2, 4 shards, N48: 1, 4) in a subprocess with four forced host devices. Gate of the
design: iteration counts equal or within 1 and ``||phi_sharded - phi_single||_M <= 10 rtol ||phi_single||_M``, both
satisfying the recomputed residual; the stacked preconditioner factors equal the global plane slices and one matvec equals
the global one to 1e-15. Skipped when the exports are missing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
EXPORT = WORKSPACE / "work/p08-step5-export-2ff50718-20261001T165909Z-ebcbd2"
CASE = Path(__file__).resolve().with_name("perpendicular_phi_sharding_real_case.py")
RTOL = 1.0e-10

pytestmark = pytest.mark.slow

CASES = ((32, (1, 2, 4)), (48, (1, 4)))


def _run(n: int, shards) -> dict:
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=4".strip()
    env["JAX_PLATFORMS"] = "cpu"
    done = subprocess.run([sys.executable, str(CASE), str(EXPORT), str(n), *map(str, shards)], capture_output=True,
                          text=True, env=env, timeout=3600, check=False)
    assert done.returncode == 0, f"subprocess failed ({done.returncode}):\n{done.stderr[-4000:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("n, shards", CASES)
def test_sharded_phi_solve_equals_single_device_on_the_real_export(n, shards):
    if not (EXPORT / f"N{n}" / "p07_dirichlet.npz").is_file() or not (EXPORT / f"N{n}" / "owner_map.npz").is_file():
        pytest.skip(f"the N{n} P07 export is unavailable")
    report = _run(n, shards)
    assert report["devices"] == 4
    single = report["single"]
    assert single["converged"] and single["relative_residual"] <= RTOL
    lines = [f"N{n}: {report['owners']} owners, single {single['iterations']} it, {single['seconds']:.2f} s "
             f"(compile {single['compile_seconds']:.1f} s)"]
    for sz in shards:
        e = report[f"Sz{sz}"]
        assert e["converged"] and e["relative_residual"] <= RTOL, (sz, e["relative_residual"])
        assert e["host_residual_rel"] <= RTOL * (1 + 1e-6), (sz, e["host_residual_rel"])
        assert abs(e["iterations"] - single["iterations"]) <= 1, (sz, e["iterations"], single["iterations"])
        assert e["dphi_m"] <= 10 * RTOL * e["phi_m"], (sz, e["dphi_m"], e["phi_m"])
        assert e["true_error_rel"] <= 1e-6, (sz, e["true_error_rel"])
        rough, rough_single = e["rough"], report["single_rough"]
        assert rough["converged"] and abs(rough["iterations"] - rough_single["iterations"]) <= 1
        assert rough["dphi_m"] <= 10 * RTOL * rough["phi_m"], (sz, rough["dphi_m"], rough["phi_m"])
        assert e["factor_max_diff"] == 0.0
        assert e["matvec_rel"] <= 1e-15, (sz, e["matvec_rel"])
        lines.append(f"  Sz={sz}: {e['iterations']} it, dphi/phi = {e['dphi_m'] / e['phi_m']:.1e}, "
                     f"{e['seconds']:.2f} s (compile {e['compile_seconds']:.1f} s), rss {report['rss_mb'][f'Sz{sz}']:.0f} MB")
    print("\n" + "\n".join(lines))
