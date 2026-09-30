"""Gates G3.2 and G3.3 of P08 step 3 on the real bounded owner closure (slow).

* **G3.2** (``run_g32``): for all seven campaign keys the combined perpendicular RHS, fed with the campaign's own owner
  values / kinds / boundary data and only that campaign's term (bracket with ``rho_star = 1`` and the campaign's pairs,
  curvature with the campaign's ``tau`` and seam multiplier, diffusion ``-D * P07`` with ``D = 1``), reproduces the E6
  ``JaxOwnerClosure`` terms under the uniform tolerance policy and every ``compare_to_oracle`` row (143).  Strict.
* **G3.3** (``run_g33``): the full combined RHS of the P06N ``main_phi_dirichlet`` / ``main_phi_neumann`` state
  (``rho_star = 0.05``, ``tau = 1``, ``D_f = 1e-2``) against the published-form continuum reference: every term of every
  field has a finite relative L2 error below 1, the least-squares scale of the combined term against the reference is
  1 within 5% (no sign or coefficient error), and with ``DRBX_STEP3_GRIDS=48,64`` no term's error grows by more than
  2x between grids (the closure has 11-12 owners, so the orders are indicative).

N32 runs in the default slow set (~1 minute).  ``DRBX_STEP3_REPORT_DIR`` additionally writes the JSON reports.
"""
from __future__ import annotations

import gc
import math
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
OPT_IN_GRIDS = tuple(int(g) for g in os.environ.get("DRBX_STEP3_GRIDS", "").split(",") if g.strip())
REPORT_DIR = os.environ.get("DRBX_STEP3_REPORT_DIR")
EXPECTED_ORACLE_ROWS = 143

pytestmark = pytest.mark.slow

_REPORTS: dict = {}


def _reports(n: int) -> dict:
    """``{"g32": report, "g33": report}`` of grid ``n`` (one setup, dropped afterwards)."""
    from p_shared import step3_gates as sg
    from p_shared import owner_closure as oc
    from p_shared.replay_support import DEFAULT_PATHS

    if n not in _REPORTS:
        directory = GEOMETRY / f"{n}x{n}x{n}"
        if not ((directory / "base_geometry.npz").is_file() and (directory / "rlp_topology.npz").is_file()
                and SIDECAR.is_file()):
            pytest.skip(f"HSX N{n} geometry/sidecar inputs are unavailable")
        if not oc.oracle_available(dict(DEFAULT_PATHS), CAMPAIGNS, n=n):
            pytest.skip(f"the frozen campaigns' N{n} oracle arrays are unavailable")
        setup = sg.build_setup(n, CAMPAIGNS, curvature="fd", face_quadrature="q3")
        _REPORTS[n] = {"g32": sg.run_g32(n, setup=setup, output_dir=REPORT_DIR),
                       "g33": sg.run_g33(n, setup=setup, output_dir=REPORT_DIR)}
        del setup
        gc.collect()
    return _REPORTS[n]


def _assert_g32(report: dict) -> None:
    assert report["uniq_mismatches"] == []                          # same out structure and uniq arrays
    assert report["bracket_term_mismatch_max"] == 0.0               # the bracket term is the first raw pair, bitwise
    assert set(report["per_campaign"]) == set(CAMPAIGNS)
    for campaign, entry in report["per_campaign"].items():
        assert entry["vs_e6"]["terms"] > 0 and entry["vs_e6"]["failures"] == [], (campaign, entry["vs_e6"])
        assert entry["vs_oracle"]["rows"] > 0 and entry["vs_oracle"]["failures"] == [], (campaign, entry["vs_oracle"])
    failed = [(r["campaign"], r["term"], r["max_abs"]) for r in report["oracle_table"] if not r["pass"]]
    assert failed == []
    assert report["oracle_rows"] == EXPECTED_ORACLE_ROWS == report["oracle_rows_passed"] == report["oracle_rows_accounted"]
    for row in report["diff_table"]:
        assert row["pass"], row
        if row["kind"] == "operator":
            assert row["max_rel_to_scale"] <= 1e-11, row
    assert report["pass"]


def _assert_g33(report: dict) -> None:
    assert report["variants"] == ["main_phi_dirichlet", "main_phi_neumann"]
    for variant, v in report["results"].items():
        assert all(count == 0 for count in v["q3_counters"].values()), (variant, v["q3_counters"])
        for field, entry in v["fields"].items():
            for term, m in entry.items():
                where = (variant, field, term, m)
                assert m["finite"] and m["rel_l2"] is not None and math.isfinite(m["rel_l2"]), where
                assert 0.0 <= m["rel_l2"] < 1.0, where
                assert m["rel_max"] is not None and 0.0 <= m["rel_max"] < 1.0, where
                if m["fit_scale"] is not None:                       # sign / coefficient check
                    assert 0.95 < m["fit_scale"] < 1.05, where
        # the only identically-zero reference: [phi, Te] when phi = Te - 1 (main_phi_neumann)
        degenerate = [(f, t) for f, e in v["fields"].items() for t, m in e.items() if m["degenerate_reference"]]
        assert degenerate == ([("Te", "poisson_bracket")] if variant == "main_phi_neumann" else []), degenerate


def test_g32_n32_combined_path_reproduces_every_campaign_and_every_oracle_row():
    _assert_g32(_reports(32)["g32"])


def test_g33_n32_combined_rhs_matches_the_continuum_reference_at_discretization_level():
    _assert_g33(_reports(32)["g33"])


@pytest.mark.skipif(not OPT_IN_GRIDS, reason="set DRBX_STEP3_GRIDS=48,64 to run the N48/N64 closures")
@pytest.mark.parametrize("n", sorted(set(OPT_IN_GRIDS) - {32}) or [0])
def test_larger_grids_g32_and_g33(n):
    if n == 0:
        pytest.skip("no opt-in grid")
    reports = _reports(n)
    _assert_g32(reports["g32"])
    _assert_g33(reports["g33"])


@pytest.mark.skipif(not OPT_IN_GRIDS, reason="set DRBX_STEP3_GRIDS=48,64 to run the N48/N64 closures")
def test_g33_no_term_error_grows_by_more_than_2x_between_grids():
    from p_shared import step3_gates as sg

    grids = sorted(set(OPT_IN_GRIDS) | {32})
    orders = sg.observed_orders({n: _reports(n)["g33"] for n in grids})
    bad = {}
    for variant, fields in orders.items():
        for field, terms in fields.items():
            for term, entry in terms.items():
                errors = entry["rel_l2"]
                assert entry["finite"] and all(e is not None for e in errors), (variant, field, term, entry)
                if any(b > sg.GROWTH_FLAG * a for a, b in zip(errors, errors[1:])):
                    bad[(variant, field, term)] = errors
    assert bad == {}
    if REPORT_DIR is not None:
        import json
        (Path(REPORT_DIR) / "g33_orders.json").write_text(json.dumps({"grids": grids, "orders": orders}, indent=2,
                                                                     allow_nan=False))
