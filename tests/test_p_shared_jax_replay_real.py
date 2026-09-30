"""G1 of P08 step 2b (E6, slow): the JAX owner closure vs the host replay and the frozen oracles, all seven campaign
keys, on the bounded owner closure of real HSX rows.

``run_jax_owner_closure_check`` builds the closure's rows once, runs the host ``assemble_owner_terms`` and the JAX
assembly (``jax_assemble_owner_terms``: plan lowering, boundary data at the plan's tables, ``p05_terms`` /
``p05n_action`` / ``p06_action`` / ``p07`` operators) and applies the uniform tolerance policy of
``p_shared.jax_replay`` (design section 8): non-cancellation terms <= 1e-11 of their scale; cancellation terms
<= 10x their measured one-ulp conditioning floor. ``compare_to_oracle`` runs with the JAX
terms (host MMS references kept), every row must pass.

N32 runs in the default slow set (~1 minute). N48 / N64 (bounded: 11-12 owners, geometry only for the selected
owners) are opt-in: ``DRBX_JAX_REPLAY_GRIDS=48,64``. ``DRBX_JAX_REPLAY_REPORT_DIR`` additionally writes the JSON
report of each grid.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
CAMPAIGNS = ("p05", "p05n_frozen", "p05n_upwind", "p06n", "p06_legacy", "p07", "p07n")
OPT_IN_GRIDS = tuple(int(g) for g in os.environ.get("DRBX_JAX_REPLAY_GRIDS", "").split(",") if g.strip())
REPORT_DIR = os.environ.get("DRBX_JAX_REPLAY_REPORT_DIR")

pytestmark = pytest.mark.slow

_PAYLOADS: dict = {}


def _payload(n: int) -> dict:
    from p_shared import jax_replay as jr
    from p_shared import owner_closure as oc
    from p_shared.replay_support import DEFAULT_PATHS

    if n not in _PAYLOADS:
        directory = GEOMETRY / f"{n}x{n}x{n}"
        if not ((directory / "base_geometry.npz").is_file() and (directory / "rlp_topology.npz").is_file()
                and SIDECAR.is_file()):
            pytest.skip(f"HSX N{n} geometry/sidecar inputs are unavailable")
        if not oc.oracle_available(dict(DEFAULT_PATHS), CAMPAIGNS, n=n):
            pytest.skip(f"the frozen campaigns' N{n} oracle arrays are unavailable")
        output = None if REPORT_DIR is None else Path(REPORT_DIR) / f"N{n}.json"
        _PAYLOADS[n] = jr.run_jax_owner_closure_check(
            n=n, input_root=WORKSPACE, sidecar_path=SIDECAR, paths=dict(DEFAULT_PATHS), campaigns=CAMPAIGNS,
            output=output, curvature="fd", face_quadrature="q3", inner_support="profile7")
    return _PAYLOADS[n]


def _assert_gate(payload: dict) -> None:
    from p_shared import jax_replay as jr

    rows = payload["diff_table"]
    assert payload["uniq_mismatches"] == []                                  # same out structure and uniq arrays
    assert {r["campaign"] for r in rows} >= {"p05", "p05n_frozen", "p05n_upwind", "p06n", "p06_legacy", "p07", "p07n"}
    terms = {r["term"] for r in rows}
    for expected in ("cells.p05_centered", "faces.p05_live_jump_owner_num", "faces.p05_live_jump_values",
                     "cells.p05n_frozen_raw_N", "faces.p05n_upwind_face_D", "cells.p06n_raw_material",
                     "faces.p06n_faces_correction", "cells.p06legacy_raw_centered.total",
                     "faces.p06legacy_faces_correction", "p07.p07_global_N", "p07.p07n_global_D",
                     "cells.q1_evolution_volume"):
        assert expected in terms, expected
    for r in rows:
        assert r["kind"] == jr.classify_term(r["term"])
        if r["kind"] == "cancellation":
            assert r["floor"] is not None and r["floor"] > 0.0, r          # the floor was measured
        if r["kind"] == "host_only":
            assert r["max_abs"] == 0.0, r                                  # host arrays, untouched
        assert r["pass"], r
    # frozen-oracle gate with the JAX terms: every row, and the same rows as the host table
    assert len(payload["oracle_jax"]) == len(payload["oracle_host"]) > 100
    assert [(r["campaign"], r["term"]) for r in payload["oracle_jax"]] == \
           [(r["campaign"], r["term"]) for r in payload["oracle_host"]]
    failed = [(r["campaign"], r["term"], r["max_abs"]) for r in payload["oracle_jax"] if not r["pass"]]
    assert failed == []
    assert payload["oracle_jax_all_pass"] and payload["oracle_host_all_pass"]


def test_jax_owner_closure_n32_meets_the_uniform_policy_and_every_oracle_row():
    payload = _payload(32)
    _assert_gate(payload)
    assert payload["all_diff_pass"] and payload["diff_failures"] == []
    # the JAX P05 antisymmetry diagnostic is roundoff, like the host's
    assert payload["antisymmetry_max"]["jax"] <= 1e-12


@pytest.mark.skipif(not OPT_IN_GRIDS, reason="set DRBX_JAX_REPLAY_GRIDS=48,64 to run the N48/N64 closures")
@pytest.mark.parametrize("n", sorted(set(OPT_IN_GRIDS) - {32}) or [0])
def test_jax_owner_closure_larger_grids(n):
    if n == 0:
        pytest.skip("no opt-in grid")
    payload = _payload(n)
    _assert_gate(payload)
    assert payload["all_diff_pass"] and payload["diff_failures"] == []
