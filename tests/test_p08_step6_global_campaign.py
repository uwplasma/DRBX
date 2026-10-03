"""Fast tests of the P08 step-6 transverse-wave campaign (``scripts/p08_step6_global``): the pinned contract in the
configuration (and its drift refusal), the field definitions (values / gradients / Hessians against closed forms and
finite differences, including near the axis; the smoothness of the switch ``B(u^2)``; positivity of the transported
fields), the owner averages (chunked vs the frozen one-shot routine, bit for bit), the u-bands, the re-freeze campaign
reading (provenance / identity / options, against synthetic folders and against the local stripped real folder), the
identity and resume refusals, the stage wiring with the heavy functions monkeypatched, the sharding hook, ``validate``
and ``analyze`` on synthetic outputs.

Fully synthetic / ``tmp_path`` based except the optional read of the local stripped re-freeze folder (skipped when
absent); the real bounded preflight is ``tests/test_p08_step6_global_campaign_real.py`` (slow).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from drbx.stencils import artifact as art                                          # noqa: E402
from p_shared import build_artifact as ba                                           # noqa: E402
from p_shared import runner                                                         # noqa: E402
from p08_step4_global import campaign as step4_campaign                             # noqa: E402
from p08_step5_combined import analysis as ana5                                     # noqa: E402
from p08_step5_combined import checkpoints, reduction                               # noqa: E402
from p08_step5_combined import references as ref5                                   # noqa: E402
from p08_step5_compact_c3 import campaign as c3                                     # noqa: E402
from p08_step6_global import analysis, fields, sharding                             # noqa: E402
from p08_step6_global import references as refs6                                    # noqa: E402
from p08_step6_global import campaign                                               # noqa: E402
from tests.test_p08_step5_combined_campaign import _synthetic_references            # noqa: E402
from tests.test_p08_step5_compact_c3_campaign import _fake_world                    # noqa: E402

FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
         "bfield_toroidal": "compact_c3"}
THREE = {k: v for k, v in FINAL.items() if k != "bfield_toroidal"}
GRIDS = (32, 48, 64)
LOCAL_REFREEZE = REPO.parent / "work" / "p08-step5-compact-c3-6c4de657-20261002T150857Z-27485"
RF_ID = "refreeze-identity"
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Configuration and sources
# ---------------------------------------------------------------------------
def test_config_pins_the_contract_the_fields_the_variants_and_the_bands():
    cfg = campaign.config()
    assert cfg["schema"] == "drbx.p08-step6-global-v1" == campaign.SCHEMA
    assert cfg["resolutions"] == [32, 48, 64]
    assert cfg["operator_options"] == FINAL == campaign.operator_options() == campaign.OPERATOR_OPTIONS
    assert cfg["params"] == {"rho_star": 0.05, "tau": 1.0,
                             "diffusion": {"density": 0.01, "Te": 0.01, "Ti": 0.01, "vorticity": 0.01}}
    assert tuple(cfg["variants"]) == ("transverse_dirichlet", "transverse_phi_wave_dirichlet")
    for spec in cfg["variants"].values():
        assert spec["kinds"] == ["dirichlet"] * 5 and spec["constant"] is False
    assert cfg["variants"]["transverse_dirichlet"]["field_set"] == "transverse"
    assert cfg["variants"]["transverse_phi_wave_dirichlet"]["field_set"] == "transverse_phi_wave"
    assert cfg["phi_solve_rtol"] == 1e-11 and cfg["phi_solve_rtol_production_default"] == 1e-8
    assert (cfg["phi_solve_boundary"], cfg["phi_solve_preconditioner"], cfg["phi_solve_factor_dtype"],
            cfg["phi_solve_restart"], cfg["phi_solve_max_restarts"]) == \
           ("dirichlet", "block_jacobi_eta_plane", "float32", 50, 40)
    assert cfg["u_band_edges"] == [0.0, 0.06, 0.12, 0.21, 0.27, 0.40, 1.0]
    assert cfg["fields"] == fields.PINNED_FIELDS and {k: v["fields"] for k, v in cfg["field_sets"].items()} == \
        fields.PINNED_FIELD_SETS
    assert tuple(cfg["stages"]) == campaign.STAGES == ("owner_values", "references", "jax", "reduce", "sharding")
    assert cfg["order_criterion"]["informational"] is True
    assert "DEVIATION" in cfg["fields_note"]


def test_the_scientific_contract_is_the_refreeze_contract_except_the_catalogue():
    ours, theirs = campaign.config(), c3.config()
    assert ours["operator_options"] == theirs["operator_options"]
    for key in ("params", "phi_solve_boundary", "phi_solve_preconditioner", "phi_solve_factor_dtype", "phi_solve_rtol",
                "phi_solve_rtol_production_default", "phi_solve_restart", "phi_solve_max_restarts", "owner_chunk_size",
                "boundary_batch", "device_put_plan", "consistency_tolerance", "degenerate_fraction", "order_criterion"):
        assert ours[key] == theirs[key], key
    assert (campaign.OPERATOR_OPTIONS, campaign.PARAMS, campaign.PHI_SOLVE) == (c3.OPERATOR_OPTIONS, c3.PARAMS, c3.PHI_SOLVE)


@pytest.mark.parametrize("name,value", [
    ("OPERATOR_OPTIONS", THREE),
    ("OPERATOR_OPTIONS", {**FINAL, "bfield_toroidal": "spline"}),
    ("PARAMS", {"rho_star": 0.1, "tau": 1.0, "diffusion": {}}),
    ("VARIANTS", ("transverse_dirichlet",)),
    ("PHI_SOLVE", {**campaign.PHI_SOLVE, "phi_solve_rtol": 1e-8}),
    ("FIELDS", {**fields.PINNED_FIELDS, "n": {**fields.PINNED_FIELDS["n"], "amplitude": 0.4}}),
    ("FIELD_SETS", {**fields.PINNED_FIELD_SETS, "transverse": ["n", "Te", "Ti", "omega", "phi_wave"]}),
    ("U_BAND_EDGES", [0.0, 0.1, 1.0]),
    ("STAGES", ("owner_values", "references", "jax", "reduce")),
    ("REFREEZE_CAMPAIGN", "p08_step5_combined"),
    ("FIELD_SETS_OF_VARIANTS", {"transverse_dirichlet": "transverse_phi_wave",
                                "transverse_phi_wave_dirichlet": "transverse_phi_wave"})])
def test_config_refuses_drift_from_the_pinned_literals(monkeypatch, name, value):
    monkeypatch.setattr(campaign, name, value)
    with pytest.raises(ValueError, match="differ from the pinned|differs from the pinned|differ from the pinned|"
                                         "differs from the pinned definition|refreeze block"):
        campaign.config()


def test_source_hashes_cover_the_package_the_refreeze_package_and_the_grid_context_sources():
    hashes = campaign.source_hashes()
    assert len(hashes) == len(campaign.SOURCE_FILES) == len(set(campaign.SOURCE_FILES))
    for rel, value in hashes.items():
        assert (REPO / rel).is_file(), rel
        assert len(value) == 64
    for needed in (*c3.SOURCE_FILES, "scripts/p08_step6_global/campaign.py", "scripts/p08_step6_global/fields.py",
                   "scripts/p08_step6_global/sharding.py", "scripts/p08_step6_global/configuration.json",
                   "scripts/perpendicular_structured/reconstruction.py", "scripts/p08_step5_combined/reduction.py"):
        assert needed in hashes


def test_folder_layout_matches_what_the_5_3_reduction_reads(tmp_path):
    assert campaign.sidecar_path(tmp_path) == tmp_path / "localized_sidecar.json"
    assert campaign.inputs_path(tmp_path) == tmp_path / "provenance" / "inputs.json"
    assert campaign.sharding_path(tmp_path, 48) == tmp_path / "N48" / "sharding.json"
    assert ref5.references_dir(tmp_path, 48) == tmp_path / "N48" / "references"
    assert refs6.owner_values_path(tmp_path, 48) == tmp_path / "N48" / "owner_values.npz"


# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------
def _points(count=64, seed=3, u_max=1.0):
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(0.0, u_max, count), rng.uniform(0.0, TWO_PI, count), rng.uniform(0.0, TWO_PI, count)))


def _columns(period=TWO_PI):
    return fields.ColumnSet(fields.PINNED_FIELDS, fields.PINNED_FIELD_SETS, period)


def _wave_numpy(p, direction_deg, lam, part, offset=0.0, amplitude=1.0, eta_phase=0.0):
    """Independent closed-form value, gradient (d_u, d_theta, d_eta) of Q's wave convention."""
    u, th, eta = p.T
    a, k = math.radians(direction_deg), TWO_PI / lam
    xa = u * np.cos(th - a)                                            # = x cos a + y sin a
    phase = k * xa + eta + eta_phase
    trig, dtrig = (np.cos(phase), -np.sin(phase)) if part == "re" else (np.sin(phase), np.cos(phase))
    grad = amplitude * dtrig[:, None] * np.column_stack((k * np.cos(th - a), -k * u * np.sin(th - a), np.ones(len(p))))
    return offset + amplitude * trig, grad


def test_waves_match_closed_forms_and_q_waves():
    p = _points()
    state = _columns().state("transverse")
    v, g = state.values_gradients(p)
    for slot, (deg, lam, part) in enumerate(((0.0, 2.0, "re"), (60.0, 2.0, "im"), (120.0, 4.0, "re"))):
        ev, eg = _wave_numpy(p, deg, lam, part, offset=1.0, amplitude=0.5)
        assert np.max(np.abs(v[slot] - ev)) < 1e-13 and np.max(np.abs(g[slot] - eg)) < 1e-12
    # Q's plane waves exp(i (2 pi x/lambda + eta)) in direction 'x' (q_fci_layered_global.fields) are the same function
    from q_fci_layered_global import fields as qf
    qv, qg = qf.field("wave_a60_lambda2", p)
    ev, eg = _wave_numpy(p, 60.0, 2.0, "im")
    assert np.max(np.abs(qv.imag - ev)) < 1e-13 and np.max(np.abs(qg.imag - eg)) < 1e-12
    xv, _xg = qf.evaluate(p, "x_lambda2_m1")
    cs = fields.ColumnSet({"w": {"kind": "wave", "direction_deg": 0.0, "wavelength": 2.0, "part": "re"}},
                          {"a": ["w"] * 5}, TWO_PI)
    assert np.max(np.abs(cs.values(p)[:, 0] - xv.real)) < 1e-13


def _switch_numpy(p, harmonic, eta_phase, u_s=0.21, w2=0.06):
    u, th, eta = p.T
    x, y = u * np.cos(th), u * np.sin(th)
    bump = np.exp(-(((x * x + y * y) - u_s ** 2) / w2) ** 2)
    poly = (x ** 4 - 6 * x * x * y * y + y ** 4) if harmonic == "cos4" else 4 * (x ** 3 * y - x * y ** 3)
    return bump * poly / u_s ** 4 * np.cos(eta + eta_phase)


def test_switch_harmonic_fields_match_the_definition_and_the_harmonic_identity():
    p = _points()
    cs = _columns()
    values = cs.values(p)
    names = list(cs.names)
    omega = values[:, names.index("omega")]
    assert np.max(np.abs(omega - _switch_numpy(p, "cos4", 0.0))) < 1e-13
    ev, _ = _wave_numpy(p, 30.0, 2.0, "re", amplitude=0.5)
    phi = values[:, names.index("phi_switch")]
    assert np.max(np.abs(phi - (_switch_numpy(p, "sin4", 0.3) + ev))) < 1e-13
    # P4 = u^4 cos(4 theta) / u^4 sin(4 theta): the harmonics are what the brief names
    u, th, eta = p.T
    s = u * u
    bump = np.exp(-((s - 0.21 ** 2) / 0.06) ** 2)
    assert np.max(np.abs(omega - bump * (s / 0.21 ** 2) ** 2 * np.cos(4 * th) * np.cos(eta))) < 1e-12
    ev150, _ = _wave_numpy(p, 150.0, 2.0, "re")
    assert np.max(np.abs(values[:, names.index("phi_wave")] - ev150)) < 1e-13
    # the switch is maximal at u_s^2 and the field is O(1)
    assert abs(float(fields.switch_function(0.21 ** 2)) - 1.0) < 1e-15
    assert 1.0 < float(np.max(np.abs(omega))) < 3.0


def test_derivatives_equal_finite_differences_everywhere_including_near_the_axis():
    cs = _columns()
    pts = np.concatenate([_points(80, seed=5), _points(40, seed=6, u_max=1e-2), _points(20, seed=7, u_max=1e-6),
                          np.array([[0.0, 0.3, 1.0], [0.0, 4.0, 5.0], [1e-12, 1.0, 2.0]])])
    for fs in cs.field_sets:
        d = fields.fd_defects(cs.state(fs), pts, step=1e-5)
        assert d["gradient_fd_rel"] < 1e-7 and d["hessian_fd_rel"] < 1e-6, (fs, d)
        assert d["hessian_asymmetry"] < 1e-12
        v, g, h = cs.state(fs).values_gradients_hessians(pts)
        assert v.shape == (5, len(pts)) and g.shape == (5, len(pts), 3) and h.shape == (5, len(pts), 3, 3)
        assert np.all(np.isfinite(v)) and np.all(np.isfinite(g)) and np.all(np.isfinite(h))
        v1, g1 = cs.state(fs).values_gradients(pts)
        assert np.array_equal(v, v1) and np.array_equal(g, g1)


def test_smoothness_at_the_axis_the_switch_depends_on_u_squared_only():
    """The fields are smooth functions of (x, y): the (u, theta) derivatives equal the chain rule of the Cartesian ones,
    which are finite at x = y = 0; B depends on u^2 only."""
    import jax.numpy as jnp
    spec = fields.PINNED_FIELDS["omega"]
    om = 1.0
    f_polar = fields.build_function(spec, om)

    def f_cart(xy, eta):
        x, y = xy[0], xy[1]
        s = x * x + y * y
        poly = x ** 4 - 6 * x * x * y * y + y ** 4
        return jnp.exp(-((s - 0.21 ** 2) / 0.06) ** 2) * poly / 0.21 ** 4 * jnp.cos(eta)

    rng = np.random.default_rng(1)
    for u in (0.0, 1e-9, 1e-4, 0.05, 0.2, 0.5):
        for th, eta in zip(rng.uniform(0, TWO_PI, 3), rng.uniform(0, TWO_PI, 3)):
            xy = jnp.array([u * math.cos(th), u * math.sin(th)])
            gx = np.asarray(jax.grad(f_cart, argnums=0)(xy, eta))
            hx = np.asarray(jax.hessian(f_cart, argnums=0)(xy, eta))
            assert np.all(np.isfinite(gx)) and np.all(np.isfinite(hx))
            p = jnp.array([u, th, eta])
            g = np.asarray(jax.grad(f_polar)(p))
            h = np.asarray(jax.hessian(f_polar)(p))
            c, s_ = math.cos(th), math.sin(th)
            assert abs(g[0] - (gx[0] * c + gx[1] * s_)) < 1e-10 and abs(g[1] - u * (-gx[0] * s_ + gx[1] * c)) < 1e-10
            # d_uu f = c^2 f_xx + 2 c s f_xy + s^2 f_yy
            assert abs(h[0, 0] - (c * c * hx[0, 0] + 2 * c * s_ * hx[0, 1] + s_ * s_ * hx[1, 1])) < 1e-9
    # B(s): finite, even-analytic (a function of s = u^2, never of u), positive, unit maximum at s = u_s^2
    s = np.linspace(0.0, 1.2, 241)
    b = np.asarray(fields.switch_function(jnp.asarray(s)))
    assert np.all(b > 0.0) and np.all(np.isfinite(b)) and abs(b.max() - 1.0) < 1e-3 and s[b.argmax()] == pytest.approx(0.0441, abs=0.006)
    db = np.asarray(jax.vmap(jax.grad(fields.switch_function))(jnp.asarray(s)))
    assert np.all(np.isfinite(db))
    # the value does not depend on theta at u = 0 and varies by O(u) near it
    state = _columns().state("transverse")
    th_values = state.values(np.array([[0.0, th, 1.0] for th in np.linspace(0, TWO_PI, 7)]))
    assert np.max(np.abs(th_values - th_values[:, :1])) < 1e-15


def test_the_transported_fields_are_positive_and_the_variants_differ_only_in_phi():
    cs = _columns()
    p = _points(400, seed=11)
    a, b = cs.state("transverse"), cs.state("transverse_phi_wave")
    va, vb = a.values(p), b.values(p)
    assert np.all(va[:3] >= 0.5 - 1e-12) and np.all(va[:3] <= 1.5 + 1e-12)       # n, Te, Ti: the model needs n > 0
    assert np.array_equal(va[:4], vb[:4]) and not np.allclose(va[4], vb[4])
    assert a.fields == ("n", "Te", "Ti", "omega", "phi_switch") and b.fields[-1] == "phi_wave"
    # the three transported fields have positive curvature-matrix denominators on the whole disk
    assert float(va[0].min()) > 0.4


def test_the_eta_phase_uses_the_period():
    p = _points(10, seed=2)
    one = fields.ColumnSet(fields.PINNED_FIELDS, fields.PINNED_FIELD_SETS, TWO_PI).values(p)
    q = p.copy()
    q[:, 2] = p[:, 2] + TWO_PI                                            # one full period of eta
    assert np.max(np.abs(fields.ColumnSet(fields.PINNED_FIELDS, fields.PINNED_FIELD_SETS, TWO_PI).values(q) - one)) < 1e-12
    half = fields.ColumnSet(fields.PINNED_FIELDS, fields.PINNED_FIELD_SETS, math.pi)
    q[:, 2] = p[:, 2] + math.pi                                           # one period of a pi-periodic grid
    assert np.max(np.abs(half.values(q) - half.values(p))) < 1e-12
    with pytest.raises(ValueError, match="unknown field kind"):
        fields.build_function({"kind": "bogus"}, 1.0)
    with pytest.raises(ValueError, match="five|known fields"):
        fields.ColumnSet(fields.PINNED_FIELDS, {"x": ["n"]}, TWO_PI)


# ---------------------------------------------------------------------------
# Owner averages and bands (a synthetic grid context with the P06N layout: order / starts / ro / rv / vol / pts)
# ---------------------------------------------------------------------------
def _synthetic_context(n_raw=600, seed=0):
    rng = np.random.default_rng(seed)
    ro = np.sort(rng.integers(0, 90, n_raw))
    ro = np.unique(ro, return_inverse=True)[1]                              # owners 0..m-1, every owner non-empty
    ro = ro[rng.permutation(n_raw)]                                         # raw ids are not sorted by owner
    pts = np.column_stack((rng.uniform(0.0, 1.0, n_raw), rng.uniform(0.0, TWO_PI, n_raw), rng.uniform(0.0, TWO_PI, n_raw)))
    rv = rng.uniform(0.5, 1.5, n_raw)
    order = np.argsort(ro, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(ro))]
    vol = np.bincount(ro, weights=rv)
    return SimpleNamespace(n=8, ro=ro, rv=rv, vol=vol, pts=pts, order=order, starts=starts)


def test_chunked_owner_averages_equal_the_frozen_one_shot_routine_bit_for_bit():
    from p05n_field_derived_global import operator as p05n_operator
    t = _synthetic_context()
    cs = _columns()
    n_owners = len(t.vol)
    chunked = np.concatenate([fields.owner_average_chunk(t, cs.values, s, min(s + 17, n_owners))
                              for s in range(0, n_owners, 17)])
    frozen = p05n_operator.live_observations(t, cs.values(t.pts))          # what P06N's observations_all calls
    assert np.array_equal(chunked, frozen) and np.array_equal(fields.owner_average(t, cs.values), frozen)
    exact = fields.owner_average_exact_sum(t, cs.values, np.arange(n_owners))
    assert np.max(np.abs(exact - frozen)) < 1e-13
    # it is the raw-volume weighted average of the midpoint values, not the unweighted mean
    o = 5
    members = t.order[t.starts[o]:t.starts[o + 1]]
    v = cs.values(t.pts[members])
    assert np.allclose(frozen[o], (t.rv[members, None] * v).sum(0) / t.vol[o], atol=1e-14)
    assert not np.allclose(frozen[o], v.mean(0), atol=1e-6)


def test_owner_values_stage_chunks_merge_and_resume(tmp_path, monkeypatch):
    t = _synthetic_context()
    cfg = campaign.config()
    cs = _columns()
    import perpendicular_structured.reconstruction as pr
    t.g = SimpleNamespace(eta_period=TWO_PI)
    monkeypatch.setattr(pr, "load_context", lambda n, root: t)
    inline = {}

    def fake_run_stage(output, stage, units, identity, *, compute, initializer, initargs, workers, **kw):
        initializer(*initargs)
        for unit in units:
            if not runner.valid_unit(output, unit, identity):
                compute(unit)
        inline["calls"] = inline.get("calls", 0) + 1
        return {"executed_units": len(units)}

    monkeypatch.setattr(runner, "run_stage", fake_run_stage)
    monkeypatch.setattr(runner, "require_cpu_backend", lambda: "cpu")
    cfg = {**cfg, "owner_chunk_size": 11}
    out = tmp_path / "out"
    first = refs6.owner_values_stage(n=32, cfg=cfg, input_root="/x", output=out, identity="ID", workers=1)
    assert first["skipped"] is False and first["n_owners"] == len(t.vol) and first["columns"] == list(cs.names)
    values, names, manifest = refs6.load_owner_values(out, 32, "ID")
    assert np.array_equal(values, fields.owner_average(t, cs.values)) and names == list(cs.names)
    assert manifest["sha256"] == runner.sha256_file(refs6.owner_values_path(out, 32))
    second = refs6.owner_values_stage(n=32, cfg=cfg, input_root="/x", output=out, identity="ID", workers=1)
    assert second["skipped"] is True and inline["calls"] == 1
    with pytest.raises(ValueError, match="no valid owner values"):
        refs6.load_owner_values(out, 32, "OTHER")
    np.savez(refs6.owner_values_path(out, 32), values=values + 1.0, columns=np.asarray(names))   # corrupted file
    with pytest.raises(ValueError, match="no valid owner values"):
        refs6.load_owner_values(out, 32, "ID")


def test_u_bands_use_the_ring_centre_partition_the_owners_and_refuse_multi_ring_owners():
    t = SimpleNamespace(pts=np.column_stack((np.repeat([0.0156, 0.0781, 0.2031, 0.2344, 0.5, 0.99], 3), np.zeros(18), np.zeros(18))),
                        order=np.arange(18), starts=np.arange(0, 19, 3))
    u = refs6.owner_u(t)
    assert u.tolist() == [0.0156, 0.0781, 0.2031, 0.2344, 0.5, 0.99]
    masks = refs6.band_masks(u, campaign.U_BAND_EDGES)
    assert list(masks) == ["uband_0.00-0.06", "uband_0.06-0.12", "uband_0.12-0.21", "uband_0.21-0.27",
                           "uband_0.27-0.40", "uband_0.40-1.00"]
    assert [int(m.sum()) for m in masks.values()] == [1, 1, 1, 1, 0, 2]
    assert sum(m.astype(int) for m in masks.values()).tolist() == [1] * 6
    edge = refs6.band_masks(np.array([0.06, 0.12, 0.21, 1.0]), campaign.U_BAND_EDGES)         # [lo, hi), last closed
    assert [int(m.sum()) for m in edge.values()] == [0, 1, 1, 1, 0, 1]
    with pytest.raises(ValueError, match="increasing"):
        refs6.band_masks(u, [0.0, 0.5, 0.2])
    t.pts[1, 0] = 0.3                                                        # an owner spanning two rings
    with pytest.raises(ValueError, match="several radial rings"):
        refs6.owner_u(t)


def test_load_owner_context_adds_the_bands_to_the_p06n_regions(monkeypatch):
    import p06n_field_derived_global.core as p06n_core
    import perpendicular_structured.reconstruction as pr
    t = _synthetic_context()
    t.pts[:, 0] = (t.ro % 10 + 0.5) / 10.0                                  # every owner on one ring
    monkeypatch.setattr(pr, "load_context", lambda n, root: t)
    monkeypatch.setattr(p06n_core, "regional_masks", lambda tt: {"interior": np.ones(len(tt.vol), bool),
                                                                 "physical_wall": np.zeros(len(tt.vol), bool)})
    vol, regions = refs6.load_owner_context(32, "/x", campaign.U_BAND_EDGES)
    assert np.array_equal(vol, t.vol)
    assert set(regions) == {"interior", "physical_wall", *refs6.band_masks(np.zeros(1), campaign.U_BAND_EDGES)}
    monkeypatch.setattr(p06n_core, "regional_masks", lambda tt: {"uband_0.00-0.06": np.ones(len(tt.vol), bool)})
    with pytest.raises(ValueError, match="region name clash"):
        refs6.load_owner_context(32, "/x", campaign.U_BAND_EDGES)
    with pytest.raises(ValueError, match="do not partition"):
        refs6.load_owner_context(32, "/x", [0.0, 0.5, 0.9])                 # owners at u > 0.9 fall outside


# ---------------------------------------------------------------------------
# The re-freeze campaign (input): synthetic folders
# ---------------------------------------------------------------------------
def _identity_json(options=FINAL, **extra):
    return {"policy": ba.build_policy(**options), "n": 32, **extra}


def _fake_refreeze(root, *, options=FINAL, identity=RF_ID, tweak=None, summary=True):
    """A complete synthetic p08_step5_compact_c3 campaign folder: three artifacts (identity, manifest, row/geometry
    files), provenance/inputs.json, validation.json, the catalogue summary and a localized sidecar."""
    root = Path(root)
    grids, artifacts = {}, {}
    for n in GRIDS:
        build = _identity_json(options, n=n)
        gdir = root / "artifact" / f"N{n}"
        (gdir / "rows").mkdir(parents=True, exist_ok=True)
        (gdir / "rows" / "cells_00000.npz").write_bytes(b"abcd")
        (gdir / "geometry.npz").write_bytes(b"geo")
        (gdir / "census.npz").write_bytes(b"cen")
        (gdir / "build_identity.json").write_text(json.dumps(build))
        manifest = {"schema": art.SCHEMA, "identity": art._json_safe(build),
                    "chunks": {"cells": [{"file": "rows/cells_00000.npz", "bytes": 4, "sha256": "x", "sources": 1, "targets": 1}]}}
        (gdir / "manifest.json").write_text(json.dumps(manifest))
        grids[str(n)] = {"artifact_dir": f"/remote/artifact/N{n}", "artifact_identity_sha256": runner.digest(build),
                         "manifest_sha256": runner.sha256_file(gdir / "manifest.json")}
        artifacts[str(n)] = {"artifact_identity_sha256": runner.digest(build), "policy": ba.build_policy(**options)}
    (root / "provenance").mkdir(exist_ok=True)
    inputs = {"campaign": "p08_step5_compact_c3", "identity": identity, "operator_options": options,
              "configuration": {"schema": "drbx.p08-step5-compact-c3-v1", "operator_options": options}, "grids": grids}
    validation = {"identity": identity, "operator_options": options, "operational_complete": True, "grids_pass": True,
                  "solver_gates_pass": True, "artifacts": artifacts}
    if tweak:
        tweak(inputs, validation)
    (root / "provenance" / "inputs.json").write_text(json.dumps(inputs))
    (root / "validation.json").write_text(json.dumps(validation))
    (root / "localized_sidecar.json").write_text(json.dumps({"schema": "x", "metric_cache": {"path": "/remote/m"},
                                                              "makegrid": {"path": "/remote/g"},
                                                              "artifact": {"path": "/remote/a"},
                                                              "metric_query_batch_size": 4096}))
    if summary:
        (root / "summary").mkdir(exist_ok=True)
        (root / "summary" / "step5_combined_summary.json").write_text(json.dumps(_catalogue_summary(identity)))
    return root


def _catalogue_summary(identity):
    results = {}
    for n in GRIDS:
        results[n] = {}
        for variant in ("main_phi_dirichlet", "dirichlet_rich"):
            for cmp in ("presc_vs_ref", "solved_vs_ref", "solved_vs_presc"):
                for field in reduction.FIELDS:
                    results[n][reduction.row_key(variant, cmp, field, "total", "global")] = {
                        "rel_l2": 1e-4 * (32 / n) ** 5, "l2": 1e-2 * (32 / n) ** 5, "max_abs": 1.0, "degenerate": False,
                        "finite": True}
    orders = ana5.orders(results)
    clean = {k: {kk: vv for kk, vv in v.items()} for k, v in orders.items()}
    return {"identity": identity, "orders": clean,
            "phi_error": {str(n): {v: {"global": {"rel_l2": 1e-6}} for v in ("main_phi_dirichlet", "dirichlet_rich")}
                          for n in GRIDS},
            "psi_diffusion": {str(n): {v: {"N_minus_O": {"global": {"rel_l2": 1e-7}}}
                                       for v in ("main_phi_dirichlet", "dirichlet_rich")} for n in GRIDS}}


def test_read_refreeze_records_identities_and_checks_the_files(tmp_path):
    root = _fake_refreeze(tmp_path / "rf")
    rf = campaign.read_refreeze(root, grids=[32, 48])
    assert rf["identity"] == RF_ID and rf["grids_pass"] is True and set(rf["grids"]) == {"32", "48", "64"}
    assert rf["grids"]["32"]["checked_on_disk"] and rf["grids"]["32"]["files_checked"]
    assert not rf["grids"]["64"]["checked_on_disk"]                     # N64 not requested: identity only
    assert rf["grids"]["48"]["artifact_dir"] == str((root / "artifact" / "N48").resolve())    # local path, not the remote one
    assert rf["grids"]["32"]["artifact_identity_sha256"] == runner.digest(
        json.loads((root / "artifact/N32/build_identity.json").read_text()))
    # a missing row file / wrong size / missing geometry is refused for a real run, tolerated with metadata_only
    (root / "artifact" / "N48" / "rows" / "cells_00000.npz").write_bytes(b"abc")
    with pytest.raises(ValueError, match="row file.*missing or of the wrong size"):
        campaign.read_refreeze(root, grids=[48])
    assert campaign.read_refreeze(root, grids=[48], metadata_only=True)["grids"]["48"]["files_checked"] is False
    (root / "artifact" / "N32" / "geometry.npz").unlink()
    with pytest.raises(ValueError, match="missing artifact file"):
        campaign.read_refreeze(root, grids=[32])
    campaign.read_refreeze(root, grids=[32], metadata_only=True)


def test_read_refreeze_refuses_other_options_identities_and_incomplete_campaigns(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        campaign.read_refreeze(tmp_path / "nowhere", grids=[32])
    with pytest.raises(ValueError, match="missing input file"):
        campaign.read_refreeze(tmp_path, grids=[32])

    def check(name, match, tweak=None, **kw):
        root = _fake_refreeze(tmp_path / name, tweak=tweak, **kw)
        with pytest.raises(ValueError, match=match):
            campaign.read_refreeze(root, grids=[32], metadata_only=True)

    check("spline", "differ from the pinned", options=THREE)                           # three-key (spline) campaign
    check("wrongcampaign", "is not a p08_step5_compact_c3", tweak=lambda i, v: i.update(campaign="p08_step5_combined"))
    check("wrongschema", "is not a p08_step5_compact_c3", tweak=lambda i, v: i["configuration"].update(schema="x"))
    check("valopts", "validation.json operator options",
          tweak=lambda i, v: v.update(operator_options={**FINAL, "face_quadrature": "q3"}))
    check("ids", "different campaign identities", tweak=lambda i, v: v.update(identity="other"))
    check("incomplete", "not operationally complete", tweak=lambda i, v: v.update(operational_complete=False))
    check("missinggrid", "no artifact record", tweak=lambda i, v: i["grids"].pop("64"))
    check("artids", "disagree on the artifact identity",
          tweak=lambda i, v: v["artifacts"]["48"].update(artifact_identity_sha256="0" * 64))
    check("policy", "records the artifact policy",
          tweak=lambda i, v: v["artifacts"]["32"].update(policy=ba.build_policy(**THREE)))
    check("nosummary", "missing input file", summary=False)
    # the build identity on disk differs from the recorded one / the artifact was built with other options
    root = _fake_refreeze(tmp_path / "disk")
    (root / "artifact/N32/build_identity.json").write_text(json.dumps(_identity_json(THREE, n=32)))
    with pytest.raises(ValueError, match="not the pinned|identity mismatch"):
        campaign.read_refreeze(root, grids=[32], metadata_only=True)
    root = _fake_refreeze(tmp_path / "disk2")
    gdir = root / "artifact/N48"
    changed = {**_identity_json(FINAL, n=48), "x": 1}
    (gdir / "build_identity.json").write_text(json.dumps(changed))
    manifest = json.loads((gdir / "manifest.json").read_text())
    manifest["identity"] = art._json_safe(changed)
    (gdir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from provenance|manifest.json differs"):
        campaign.read_refreeze(root, grids=[48], metadata_only=True)
    root = _fake_refreeze(tmp_path / "disk3")
    (root / "artifact/N32/manifest.json").write_text(json.dumps({"schema": art.SCHEMA, "identity": {}, "chunks": {}}))
    with pytest.raises(ValueError, match="identity mismatch"):
        campaign.read_refreeze(root, grids=[32], metadata_only=True)
    root = _fake_refreeze(tmp_path / "summaryid")
    (root / "summary/step5_combined_summary.json").write_text(json.dumps({"identity": "other"}))
    with pytest.raises(ValueError, match="catalogue summary belongs to another"):
        campaign.read_refreeze(root, grids=[32], metadata_only=True)


#: the small metadata files of the real local re-freeze folder that the stripped fixture copies (no rows, geometry,
#: census, plan or build receipt: the test never depends on whether ``work/`` still holds the bulk artifact)
_REFREEZE_METADATA = ("provenance/inputs.json", "validation.json", "localized_sidecar.json",
                      "summary/step5_combined_summary.json",
                      *(f"artifact/N{n}/{name}" for n in GRIDS for name in ("build_identity.json", "manifest.json")))


@pytest.mark.skipif(not all((LOCAL_REFREEZE / name).is_file() for name in _REFREEZE_METADATA),
                    reason="the local re-freeze folder's metadata files are unavailable")
def test_the_local_stripped_refreeze_folder_is_read_by_identity_and_provenance(tmp_path):
    import shutil
    stripped = tmp_path / "stripped_refreeze"                    # own stripped copy: only the small metadata files
    for name in _REFREEZE_METADATA:
        (stripped / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LOCAL_REFREEZE / name, stripped / name)
    assert not list(stripped.glob("artifact/N*/geometry.npz")) and not list(stripped.glob("artifact/N*/rows"))
    rf = campaign.read_refreeze(stripped, grids=list(GRIDS), metadata_only=True)
    inputs = json.loads((stripped / "provenance" / "inputs.json").read_text())
    assert rf["identity"] == inputs["identity"] and rf["grids_pass"] is True
    for n in GRIDS:
        rec = rf["grids"][str(n)]
        assert rec["artifact_identity_sha256"] == inputs["grids"][str(n)]["artifact_identity_sha256"]
        assert rec["artifact_identity_sha256"] == runner.digest(
            json.loads((stripped / "artifact" / f"N{n}" / "build_identity.json").read_text()))
        assert rec["artifact_dir"] == str((stripped / "artifact" / f"N{n}").resolve())
        assert rec["checked_on_disk"] is True and rec["files_checked"] is False
    with pytest.raises(ValueError, match="missing artifact file"):                  # no rows / geometry in the stripped copy
        campaign.read_refreeze(stripped, grids=[32])
    catalogue = analysis.read_catalogue_summary(stripped, rf["identity"])
    key = reduction.row_key("main_phi_dirichlet", "presc_vs_ref", "density", "total", "global")
    assert catalogue["orders"][key]["n"] == [32, 48, 64]
    assert reduction.row_key("dirichlet_rich", "solved_vs_ref", "Te", "total", "global") in catalogue["orders"]


# ---------------------------------------------------------------------------
# verify-inputs: identity, resume refusals
# ---------------------------------------------------------------------------
def _world(tmp_path, monkeypatch, **kw):
    workspace = _fake_world(tmp_path, monkeypatch, **kw)
    rf = _fake_refreeze(tmp_path / "refreeze")
    return workspace, rf


def _verify(workspace, rf, output, grids=GRIDS, **kw):
    return campaign.verify_inputs(refreeze=rf, input_root=workspace, oracle_root=kw.pop("oracle_root", None),
                                  output=output, grids=list(grids), **kw)


def test_verify_inputs_records_provenance_and_the_identity_is_stable(tmp_path, monkeypatch):
    workspace, rf = _world(tmp_path, monkeypatch, input_files=[("a/in.bin", b"abc")])
    identity, record = _verify(workspace, rf, tmp_path / "out")
    assert _verify(workspace, rf, tmp_path / "out")[0] == identity
    # the identity does not depend on which grids are requested / verified
    assert _verify(workspace, rf, tmp_path / "out", grids=(32,), oracle_grids=[])[0] == identity
    saved = json.loads((tmp_path / "out" / "provenance" / "inputs.json").read_text())
    assert saved["identity"] == identity and saved["campaign"] == "p08_step6_global"
    assert saved["operator_options"] == FINAL and saved["refreeze"]["identity"] == RF_ID
    assert set(saved["grids"]) == {"32", "48", "64"} and saved["metadata_only"] is False
    assert saved["grids"]["32"]["artifact_identity_sha256"] == record["grids"]["32"]["artifact_identity_sha256"]
    assert saved["jax"]["backend"] == "cpu" and saved["jax"]["x64"] is True
    assert (tmp_path / "out" / "localized_sidecar.json").is_file()
    assert saved["localized_sidecar_sha256"] == runner.sha256_file(tmp_path / "out" / "localized_sidecar.json")


def test_identity_depends_on_configuration_sources_refreeze_and_artifacts(tmp_path, monkeypatch):
    workspace, rf = _world(tmp_path, monkeypatch)
    one = _verify(workspace, rf, tmp_path / "one")[0]
    real = campaign.config
    monkeypatch.setattr(campaign, "config", lambda: {**real(), "consistency_tolerance": 1e-3})
    two = _verify(workspace, rf, tmp_path / "two")[0]
    with pytest.raises(ValueError, match="campaign identity changed"):
        _verify(workspace, rf, tmp_path / "one")
    monkeypatch.setattr(campaign, "config", real)
    real_sources = campaign.SOURCE_FILES
    monkeypatch.setattr(campaign, "SOURCE_FILES", real_sources[:-1])
    three = _verify(workspace, rf, tmp_path / "three")[0]
    monkeypatch.setattr(campaign, "SOURCE_FILES", real_sources)
    other = _fake_refreeze(tmp_path / "rf_other", identity="another-refreeze")
    four = _verify(workspace, other, tmp_path / "four")[0]
    # another artifact identity inside the same refreeze identity
    changed = _fake_refreeze(tmp_path / "rf_art", tweak=lambda i, v: (i["grids"]["64"].update(artifact_identity_sha256="1" * 64),
                                                                       v["artifacts"]["64"].update(artifact_identity_sha256="1" * 64)))
    assert len({one, two, three, four}) == 4
    with pytest.raises(ValueError, match="differs from provenance|artifact identity of N64"):
        _verify(workspace, changed, tmp_path / "five", grids=(64,))


def test_resume_refusals_oracle_root_artifact_identity_inputs_and_backend(tmp_path, monkeypatch):
    workspace, rf = _world(tmp_path, monkeypatch, input_files=[("a/in.bin", b"abc")])
    out = tmp_path / "out"
    _verify(workspace, rf, out)
    (workspace / "a" / "in.bin").write_bytes(b"abd")
    with pytest.raises(ValueError, match="missing or changed immutable input"):
        _verify(workspace, rf, tmp_path / "o1")
    (workspace / "a" / "in.bin").write_bytes(b"abc")
    with pytest.raises(ValueError, match="oracle verification failed"):
        _verify(workspace, rf, tmp_path / "o2", oracle_root=tmp_path / "elsewhere")
    import shutil
    shutil.copytree(workspace / "work", tmp_path / "elsewhere" / "work")
    with pytest.raises(ValueError, match="oracle root changed"):
        _verify(workspace, rf, out, oracle_root=tmp_path / "elsewhere")
    _verify(workspace, rf, tmp_path / "o3", oracle_grids=[])               # no oracle file is read by a run
    shutil.rmtree(workspace / "work")
    _verify(workspace, rf, tmp_path / "o4", oracle_grids=[])
    with pytest.raises(ValueError, match="oracle verification failed"):
        _verify(workspace, rf, tmp_path / "o5", oracle_grids=[32])
    with pytest.raises(ValueError, match="not a directory"):
        _verify(tmp_path / "nowhere", rf, tmp_path / "o6")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(ValueError, match="CPU backend required"):
        _verify(workspace, rf, tmp_path / "o7", oracle_grids=[])


def test_the_localized_sidecar_must_agree_with_the_refreeze_one_up_to_paths(tmp_path, monkeypatch):
    workspace, rf = _world(tmp_path, monkeypatch)
    _verify(workspace, rf, tmp_path / "ok")                              # paths differ (remote vs local): accepted
    side = json.loads((rf / "localized_sidecar.json").read_text())
    side["schema"] = "y"                                                  # a different sidecar content
    (rf / "localized_sidecar.json").write_text(json.dumps(side))
    with pytest.raises(ValueError, match="differs from the re-freeze campaign's beyond"):
        _verify(workspace, rf, tmp_path / "bad")


def test_metadata_only_is_recorded_and_the_artifact_rows_are_required_otherwise(tmp_path, monkeypatch):
    workspace, rf = _world(tmp_path, monkeypatch)
    (rf / "artifact" / "N48" / "geometry.npz").unlink()
    with pytest.raises(ValueError, match="missing artifact file"):
        _verify(workspace, rf, tmp_path / "real")
    _, record = _verify(workspace, rf, tmp_path / "meta", metadata_only=True)
    assert record["metadata_only"] is True and record["grids"]["48"]["files_checked"] is False
    # the stage functions refuse inputs verified without the rows
    with pytest.raises(ValueError, match="--metadata-only"):
        campaign.jax_stage(n=48, args=_stage_args(tmp_path / "meta"), identity="ID", inputs=record)
    for command in ("run", "validate", "analyze", "run-stage"):
        with pytest.raises(ValueError, match="--metadata-only is only allowed"):
            campaign.main([command, "--refreeze-campaign", str(rf), "--input-root", str(workspace), "--output",
                           str(tmp_path / "m2"), "--metadata-only"])


# ---------------------------------------------------------------------------
# CLI and stage wiring
# ---------------------------------------------------------------------------
def test_cli_parsing():
    base = ["--refreeze-campaign", "/r", "--input-root", "/x", "--output", "/y"]
    a = campaign.parse(["run", *base])
    assert a.resolutions == [32, 48, 64] and a.workers == 4 and a.oracle_root is None and a.metadata_only is False
    assert a.refreeze_campaign == Path("/r")
    a = campaign.parse(["preflight", *base, "--workers", "96", "--memory-budget-gib", "400", "--worker-memory-gib", "3.5",
                        "--memory-reserve-gib", "8", "--max-tasks-per-worker", "50", "--resolutions", "32", "--metadata-only"])
    assert a.workers == 96 and a.resolutions == [32] and a.metadata_only is True and a.max_tasks_per_worker == 50
    for stage in ("owner_values", "references", "jax", "reduce", "sharding"):
        assert campaign.parse(["run-stage", *base, "--stage", stage, "--n", "48"]).stage == stage
    for command in ("verify-inputs", "validate", "analyze"):
        assert campaign.parse([command, *base]).command == command
    for bad in (["run", "--resolutions", "16"], ["frobnicate"], ["run-stage", "--stage", "artifact"],
                ["run", "--step4-campaign", "/s"]):
        with pytest.raises(SystemExit):
            campaign.parse([*bad, *base])
    with pytest.raises(SystemExit):
        campaign.parse(["run", "--input-root", "/x", "--output", "/y"])         # --refreeze-campaign is required


def _stage_args(output, workspace="/ws", **over):
    base = dict(output=Path(output), input_root=Path(workspace), oracle_root=None, workers=3, memory_budget_gib=None,
                worker_memory_gib=None, memory_reserve_gib=1.0, max_tasks_per_worker=7, resolutions=[32, 48, 64],
                refreeze_campaign=Path("/rf"))
    return SimpleNamespace(**{**base, **over})


def _inputs(rf="/rf", metadata_only=False):
    return {"grids": {str(n): {"artifact_identity_sha256": f"art{n}"} for n in GRIDS},
            "refreeze": {"path": rf, "identity": RF_ID, "grids_pass": True, "solver_gates_pass": True},
            "metadata_only": metadata_only}


def test_stage_calls_get_the_pinned_options_the_refreeze_artifacts_and_this_campaign_folder(tmp_path, monkeypatch):
    from p08_step6_global import jaxstage
    out = tmp_path / "out"
    seen = {}
    monkeypatch.setattr(refs6, "owner_values_stage", lambda **kw: seen.setdefault("ov", kw) and {"n_owners": 1, "skipped": False})
    monkeypatch.setattr(refs6, "references_stage", lambda **kw: seen.setdefault("refs", kw) and {"n_owners": 1, "skipped": False})
    monkeypatch.setattr(jaxstage, "jax_stage", lambda **kw: seen.setdefault("jax", kw) and {"computed": [], "resumed": []})
    monkeypatch.setattr(reduction, "reduce_grid", lambda **kw: seen.setdefault("reduce", kw) and {"gates": {}})
    args = _stage_args(out, oracle_root=Path("/oracles"))
    inputs = _inputs()
    campaign.owner_values_stage(n=64, args=args, identity="ID")
    campaign.references_stage(n=64, args=args, identity="ID")
    campaign.jax_stage(n=64, args=args, identity="ID", inputs=inputs)
    campaign.reduce_stage(n=64, args=args, identity="ID", inputs=inputs)
    o, r, j, d = seen["ov"], seen["refs"], seen["jax"], seen["reduce"]
    cfg = campaign.config()
    assert o["cfg"] == r["cfg"] == j["cfg"] == d["cfg"] == cfg
    assert r["options"] == j["options"] == FINAL and o["workers"] == r["workers"] == 3 and r["max_tasks_per_worker"] == 7
    assert r["sidecar_path"] == j["sidecar_path"] == out / "localized_sidecar.json"
    assert j["artifact_root"] == Path("/rf") / "artifact" and j["inputs"] is inputs and d["inputs"] is inputs
    assert o["output"] == r["output"] == j["output"] == d["output"] == out
    with pytest.raises(ValueError, match="not recorded"):
        campaign.jax_stage(n=64, args=args, identity="ID", inputs={"grids": {}})
    with pytest.raises(ValueError, match="not recorded"):
        campaign.reduce_stage(n=64, args=args, identity="ID", inputs={"grids": {}})


def _stub_stages(monkeypatch, order, *, grid_pass=lambda n: True, shard="not_implemented"):
    monkeypatch.setattr(campaign, "owner_values_stage", lambda *, n, args, identity:
                        order.append(("owner_values", n)) or {"n_owners": 5, "skipped": False})
    monkeypatch.setattr(campaign, "references_stage", lambda *, n, args, identity:
                        order.append(("references", n)) or {"n_owners": 5, "skipped": False})
    monkeypatch.setattr(campaign, "jax_stage", lambda *, n, args, identity, inputs:
                        order.append(("jax", n)) or {"computed": ["a"], "resumed": []})
    monkeypatch.setattr(campaign, "reduce_stage", lambda *, n, args, identity, inputs:
                        order.append(("reduce", n)) or {"gates": {"grid_pass": grid_pass(n), "solver_gates_pass": True,
                                                                  "all_finite": True}})
    monkeypatch.setattr(campaign, "sharding_stage", lambda *, n, args, identity, inputs:
                        order.append(("sharding", n)) or {"status": shard})


def test_run_grid_order_ends_with_the_sharding_hook(tmp_path, monkeypatch):
    order = []
    _stub_stages(monkeypatch, order)
    grid = campaign.run_grid(n=32, args=_stage_args(tmp_path), identity="ID", inputs={})
    assert order == [("owner_values", 32), ("references", 32), ("jax", 32), ("reduce", 32), ("sharding", 32)]
    assert grid["grid_pass"] is True and grid["status"] == "complete" and grid["sharding_status"] == "not_implemented"
    assert grid["variants_computed"] == ["a"]


def _verified_output(tmp_path, monkeypatch, *, preflight=True):
    workspace, rf = _world(tmp_path, monkeypatch)
    out = tmp_path / "out"
    identity, _ = _verify(workspace, rf, out)
    if preflight:
        runner.write_json(out / "preflight.json", {"identity": identity, "all_pass": True, "cases": {"32": {"all_pass": True}}})
    argv = ["--refreeze-campaign", str(rf), "--input-root", str(workspace), "--output", str(out)]
    return argv, out, identity


def test_main_arguments_are_checked_verify_inputs_locks_and_records(tmp_path, monkeypatch, capsys):
    workspace, rf = _world(tmp_path, monkeypatch)
    argv = ["--refreeze-campaign", str(rf), "--input-root", str(workspace)]
    with pytest.raises(ValueError, match="--stage and --n are required"):
        campaign.main(["run-stage", *argv, "--output", str(tmp_path / "o1")])
    with pytest.raises(ValueError, match="unique and ascending"):
        campaign.main(["verify-inputs", *argv, "--output", str(tmp_path / "o1"), "--resolutions", "48", "32"])
    with pytest.raises(ValueError, match="not among the verified"):
        campaign.main(["run-stage", *argv, "--output", str(tmp_path / "o1"), "--stage", "jax", "--n", "64",
                       "--resolutions", "32"])
    out = tmp_path / "out"
    campaign.main(["verify-inputs", *argv, "--output", str(out)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "verify-inputs" and payload["status"] == "complete"
    assert (out / ".runner.lock").exists() and list((out / "invocations").glob("*_verify-inputs.json"))
    assert json.loads((out / "last_exit.json").read_text())["status"] == "complete"


def test_preflight_command_is_resumable_and_run_is_gated(tmp_path, monkeypatch, capsys):
    argv, out, _ = _verified_output(tmp_path, monkeypatch, preflight=False)
    calls = []

    def fake_preflight_grid(*, n, input_root, sidecar, output, paths):
        calls.append((n, sidecar))
        return {"n": n, "all_pass": len(calls) > 1}

    monkeypatch.setattr(campaign, "preflight_grid", fake_preflight_grid)
    with pytest.raises(SystemExit):
        campaign.main(["preflight", *argv])
    assert calls == [(32, out / "localized_sidecar.json")]
    assert json.loads((out / "last_exit.json").read_text())["status"] == "failed_gates"
    with pytest.raises(ValueError, match="preflight did not pass"):
        campaign.main(["run", *argv, "--resolutions", "32"])
    capsys.readouterr()
    campaign.main(["preflight", *argv])                                    # failed one is recomputed
    campaign.main(["preflight", *argv])                                    # passed one is not redone
    assert len(calls) == 2 and json.loads((out / "preflight.json").read_text())["all_pass"] is True
    with pytest.raises(ValueError, match="requires a matching preflight"):
        campaign.main(["run", *argv[:-1], str(tmp_path / "fresh"), "--resolutions", "32"])


def test_run_flow_runs_the_grids_in_order_stops_after_a_failing_grid_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    argv, out, _ = _verified_output(tmp_path, monkeypatch)
    order, done = [], []
    _stub_stages(monkeypatch, order, grid_pass=lambda n: n != 48)
    monkeypatch.setattr(campaign, "validate", lambda **kw: done.append(("validate", kw["args"].resolutions))
                        or {"solver_gates_pass": True, "all_finite": True, "sharding": {"status": "pending"}})
    monkeypatch.setattr(campaign, "analyze", lambda **kw: done.append(("analyze", kw["args"].resolutions)))
    with pytest.raises(SystemExit):
        campaign.main(["run", *argv])
    assert [o for o in order if o[0] == "owner_values"] == [("owner_values", 32), ("owner_values", 48)]   # N64 not run
    last = json.loads((out / "last_exit.json").read_text())
    assert last["status"] == "failed_gates" and last["summary"]["operational_complete"] is False
    assert [g["n"] for g in last["summary"]["grids"]] == [32, 48]
    assert done == [("validate", [32, 48]), ("analyze", [32, 48])]
    order.clear()
    done.clear()
    _stub_stages(monkeypatch, order)
    campaign.main(["run", *argv])
    assert [o for o in order if o[0] == "sharding"] == [("sharding", 32), ("sharding", 48), ("sharding", 64)]
    assert done == [("validate", [32, 48, 64]), ("analyze", [32, 48, 64])]
    assert json.loads((out / "last_exit.json").read_text())["summary"]["operational_complete"] is True
    # a failing sharding record stops the run like a failing grid
    order.clear()
    done.clear()
    _stub_stages(monkeypatch, order, shard="fail")
    with pytest.raises(SystemExit):
        campaign.main(["run", *argv])
    assert [o for o in order if o[0] == "sharding"] == [("sharding", 32)]
    capsys.readouterr()


def test_run_stage_dispatch(tmp_path, monkeypatch, capsys):
    argv, out, _ = _verified_output(tmp_path, monkeypatch, preflight=False)
    order = []
    _stub_stages(monkeypatch, order)
    for stage in campaign.STAGES:
        campaign.main(["run-stage", *argv, "--stage", stage, "--n", "48"])
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["stage"] == stage
    assert order == [(s, 48) for s in campaign.STAGES]
    _stub_stages(monkeypatch, order, grid_pass=lambda n: False)
    with pytest.raises(SystemExit):
        campaign.main(["run-stage", *argv, "--stage", "reduce", "--n", "48"])
    _stub_stages(monkeypatch, order, shard="fail")
    with pytest.raises(SystemExit):
        campaign.main(["run-stage", *argv, "--stage", "sharding", "--n", "48"])
    assert json.loads((out / "last_exit.json").read_text())["status"] == "failed_gates"


# ---------------------------------------------------------------------------
# The sharding hook (stored record)
# ---------------------------------------------------------------------------
def test_the_sharding_record_is_stored_loaded_and_the_status_checked(tmp_path, monkeypatch):
    args = _stage_args(tmp_path)
    monkeypatch.setattr(sharding, "run", lambda **kw: {"status": "not_implemented"})
    assert campaign.sharding_stage(n=32, args=args, identity="ID", inputs={}) == {"status": "not_implemented"}
    assert json.loads(campaign.sharding_path(tmp_path, 32).read_text()) == {"identity": "ID", "n": 32,
                                                                             "status": "not_implemented"}
    assert campaign.load_sharding(tmp_path, 32, "ID") == {"identity": "ID", "n": 32, "status": "not_implemented",
                                                          "recorded": True}
    assert campaign.load_sharding(tmp_path, 48, "ID") == {"status": "not_implemented", "recorded": False}
    with pytest.raises(ValueError, match="another campaign identity"):
        campaign.load_sharding(tmp_path, 32, "OTHER")
    monkeypatch.setattr(sharding, "run", lambda **kw: {"status": "maybe"})
    with pytest.raises(ValueError, match="unknown status"):
        campaign.sharding_stage(n=32, args=args, identity="ID", inputs={})
    received = {}
    monkeypatch.setattr(sharding, "run", lambda **kw: received.update(kw) or {"status": "pass", "shards": 4})
    campaign.sharding_stage(n=64, args=args, identity="ID", inputs={"x": 1})
    assert set(received) == {"n", "args", "identity", "inputs", "cfg"} and received["cfg"] == campaign.config()


# ---------------------------------------------------------------------------
# The sharding stage (subprocess monkeypatched; the worker itself is the slow functional test)
# ---------------------------------------------------------------------------
def _scfg():
    return campaign.config()["sharding"]


def test_the_sharding_configuration_is_pinned_and_a_drift_is_refused(tmp_path, monkeypatch):
    scfg = _scfg()
    assert {k: v for k, v in scfg.items() if k != "note"} == campaign.SHARDING
    assert scfg["shard_counts"] == [2, 4, 8] and scfg["devices"] == 8 and scfg["min_block_planes"] == 3
    assert scfg["rhs_gate"] == 1e-12 and scfg["phi_iteration_tolerance"] == 1 and scfg["phi_solution_rtol_factor"] == 10.0
    assert scfg["transverse_variant"] == "transverse_dirichlet" and scfg["catalogue_variant"] == "main_phi_dirichlet"
    assert scfg["max_shards"] is None and scfg["plan_halo"] == 3 and scfg["phi_halo"] == 2
    real = (campaign.HERE / "configuration.json").read_text()
    drifted = json.loads(real)
    drifted["sharding"]["shard_counts"] = [2, 4]
    fake = tmp_path / "p08_step6_global"
    fake.mkdir()
    (fake / "configuration.json").write_text(json.dumps(drifted))
    monkeypatch.setattr(campaign, "HERE", fake)
    with pytest.raises(ValueError, match="sharding block"):
        campaign.config()


def test_shard_counts_divide_n_with_a_block_of_at_least_three_planes_and_honour_the_cap():
    scfg = _scfg()
    for n in GRIDS:
        assert sharding.shard_counts(n, scfg) == ([2, 4, 8], [])
    assert sharding.shard_counts(24, scfg) == ([2, 4, 8], [])                    # 24 / 8 = 3 planes
    assert sharding.shard_counts(16, scfg) == ([2, 4], [])                       # 16 / 8 = 2 < 3
    assert sharding.shard_counts(30, scfg) == ([2], [])                          # 4 and 8 do not divide 30
    assert sharding.shard_counts(48, scfg, max_shards=4) == ([2, 4], [8])
    assert sharding.shard_counts(48, scfg, max_shards=1) == ([], [2, 4, 8])


def test_the_worker_environment_forces_eight_host_devices_and_keeps_the_campaign_settings():
    env = sharding.worker_environment(8, {"XLA_FLAGS": "--xla_cpu_foo=1 --xla_force_host_platform_device_count=2",
                                          "OMP_NUM_THREADS": "3", "JAX_PLATFORMS": "gpu", "PATH": "/bin"})
    assert env["XLA_FLAGS"].split() == ["--xla_cpu_foo=1", "--xla_force_host_platform_device_count=8"]
    assert env["JAX_PLATFORMS"] == "cpu" and env["JAX_ENABLE_X64"] == "true" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == "3" and env["OPENBLAS_NUM_THREADS"] == "1" and env["PATH"] == "/bin"
    assert sharding.worker_environment(8, {})["XLA_FLAGS"] == "--xla_force_host_platform_device_count=8"


def _raw_result(counts=(2, 4, 8), *, rel=1e-15, bitwise=False, single_it=7, it=7, dpsi_rel=1e-13, rtol=1e-11,
                converged=True, finite=True):
    """A worker result of two variants with the raw metrics only."""
    rhs, phi = {}, {}
    for variant in ("transverse_dirichlet", "main_phi_dirichlet"):
        rhs[variant] = {"single": {"first_seconds": 3.0, "seconds": 1.0}}
        phi[variant] = {"single": {"iterations": single_it, "converged": True, "psi_m": 2.0, "seconds": 1.0,
                                   "first_seconds": 2.0, "rhs_source": "reference_O"}}
        for sz in counts:
            rhs[variant][f"Sz{sz}"] = {"worst_rel": rel, "bitwise": bitwise, "finite": finite, "lowering_seconds": 1.0,
                                       "first_seconds": 3.0, "seconds": 1.0, "fields": {}}
            phi[variant][f"Sz{sz}"] = {"iterations": it, "converged": converged, "dpsi_m": 2.0 * dpsi_rel, "psi_m": 2.0,
                                       "lowering_seconds": 1.0, "first_seconds": 3.0, "seconds": 1.0}
    return {"n": 32, "identity": "ID", "shard_counts": list(counts), "phi_rtol": rtol, "rhs": rhs, "phi": phi,
            "peak_rss_gib": {"final": 1.5}, "devices": 8}


def test_assess_passes_when_every_gate_of_every_shard_count_holds_and_does_not_mutate_the_input():
    raw = _raw_result()
    before = json.dumps(raw, sort_keys=True)
    rec = sharding.assess(raw, _scfg())
    assert json.dumps(raw, sort_keys=True) == before
    assert rec["status"] == "pass" and rec["failures"] == []
    assert rec["gates"]["rhs"] and rec["gates"]["phi"] and rec["gates"]["complete"] and rec["gates"]["rhs_bitwise"] is False
    assert rec["rhs"]["main_phi_dirichlet"]["Sz4"]["pass"] is True and rec["rhs"]["main_phi_dirichlet"]["Sz4"]["gate"] == 1e-12
    e = rec["phi"]["transverse_dirichlet"]["Sz8"]
    assert e["pass"] is True and e["iteration_difference"] == 0
    assert e["solution_tolerance_m"] == pytest.approx(10 * 1e-11 * 2.0) and all(e["checks"].values())
    assert sharding.assess(_raw_result(bitwise=True), _scfg())["gates"]["rhs_bitwise"] is True


def test_assess_fails_on_each_gate():
    scfg = _scfg()
    gate = scfg["rhs_gate"]
    rec = sharding.assess(_raw_result(rel=gate * 1.0000001), scfg)
    assert rec["status"] == "fail" and rec["gates"]["rhs"] is False and rec["gates"]["phi"] is True
    assert sharding.assess(_raw_result(rel=gate), scfg)["status"] == "pass"           # the gate is inclusive
    assert sharding.assess(_raw_result(rel=float("nan")), scfg)["status"] == "fail"
    assert sharding.assess(_raw_result(finite=False), scfg)["gates"]["rhs"] is False
    rec = sharding.assess(_raw_result(it=9), scfg)                                     # |9 - 7| = 2 > 1
    assert rec["status"] == "fail" and rec["gates"]["phi"] is False
    assert rec["phi"]["main_phi_dirichlet"]["Sz2"]["checks"] == {"converged": True, "iterations": False, "solution": True}
    assert sharding.assess(_raw_result(it=8), scfg)["status"] == "pass"               # |8 - 7| = 1 <= 1
    rec = sharding.assess(_raw_result(dpsi_rel=1.01e-10), scfg)                        # 10 rtol = 1e-10 relative
    assert rec["status"] == "fail" and rec["phi"]["transverse_dirichlet"]["Sz4"]["checks"]["solution"] is False
    assert sharding.assess(_raw_result(dpsi_rel=0.99e-10), scfg)["status"] == "pass"
    rec = sharding.assess(_raw_result(converged=False), scfg)
    assert rec["status"] == "fail" and rec["phi"]["transverse_dirichlet"]["Sz2"]["checks"]["converged"] is False
    raw = _raw_result()
    raw["phi"]["transverse_dirichlet"]["single"]["converged"] = False
    assert sharding.assess(raw, scfg)["status"] == "fail"
    raw = _raw_result()
    del raw["rhs"]["main_phi_dirichlet"]["Sz4"]
    rec = sharding.assess(raw, scfg)
    assert rec["status"] == "fail" and "rhs/main_phi_dirichlet/Sz4: missing" in rec["failures"]
    raw = _raw_result()
    del raw["phi"]["main_phi_dirichlet"]["Sz8"]
    assert "phi/main_phi_dirichlet/Sz8: missing" in sharding.assess(raw, scfg)["failures"]
    empty = _raw_result()
    empty["rhs"] = {}
    assert sharding.assess(empty, scfg)["status"] == "fail"


def test_assess_is_pending_when_the_cap_left_shard_counts_out_and_nothing_failed():
    scfg = _scfg()
    rec = sharding.assess(_raw_result(counts=(2, 4)), scfg, requested=[2, 4, 8])
    assert rec["status"] == "pending" and rec["gates"]["complete"] is False and rec["gates"]["rhs"] and rec["gates"]["phi"]
    assert rec["shard_counts_requested"] == [2, 4, 8]
    assert sharding.assess(_raw_result(counts=(2, 4), rel=1e-3), scfg, requested=[2, 4, 8])["status"] == "fail"


def _fake_worker(monkeypatch, tmp_path, *, make=None, returncode=0, calls=None, write=True):
    """Replace ``subprocess.run`` of the sharding module by a fake worker that logs to the handle and writes the result."""
    calls = [] if calls is None else calls

    def fake(command, stdout=None, stderr=None, env=None, check=None):
        spec_path = Path(command[command.index("--worker") + 1])
        result_path = Path(command[command.index("--result") + 1])
        spec = json.loads(spec_path.read_text())
        calls.append({"command": command, "env": env, "spec": spec, "stdout": stdout.name, "stderr": stderr})
        stdout.write(f"fake worker N{spec['n']} counts {spec['shard_counts']}\n".encode())
        if returncode == 0 and write:
            raw = (make or (lambda sp: _raw_result(tuple(sp["shard_counts"]))))(spec)
            raw.update(n=spec["n"], identity=spec["identity"])
            result_path.write_text(json.dumps(raw))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(sharding, "subprocess", SimpleNamespace(run=fake, STDOUT=-2))
    return calls


def test_run_starts_the_worker_in_a_subprocess_with_forced_devices_and_records_the_assessment(tmp_path, monkeypatch):
    calls = _fake_worker(monkeypatch, tmp_path)
    args = _stage_args(tmp_path, oracle_root=Path("/oracles"))
    inputs, cfg = _inputs(), campaign.config()
    rec = sharding.run(n=48, args=args, identity="ID", inputs=inputs, cfg=cfg, log=lambda m: None)
    (call,) = calls
    assert call["command"][0] == sys.executable and call["command"][1].endswith("p08_step6_global/sharding.py")
    assert call["env"]["XLA_FLAGS"].endswith("--xla_force_host_platform_device_count=8")
    assert call["env"]["JAX_PLATFORMS"] == "cpu" and call["env"]["JAX_ENABLE_X64"] == "true"
    assert call["stdout"] == str(tmp_path / "logs" / "sharding_N48.log") and call["stderr"] == sharding.subprocess.STDOUT
    assert "fake worker N48 counts [2, 4, 8]" in (tmp_path / "logs" / "sharding_N48.log").read_text()
    spec = call["spec"]
    assert spec["n"] == 48 and spec["identity"] == "ID" and spec["shard_counts"] == [2, 4, 8] and spec["cfg"] == cfg
    assert spec["inputs"] == inputs and spec["output"] == str(tmp_path) and spec["oracle_root"] == "/oracles"
    assert spec["input_root"] == "/ws"
    assert rec["status"] == "pass" and rec["shard_counts"] == [2, 4, 8] and rec["shard_counts_skipped"] == []
    assert rec["shard_counts_requested"] == [2, 4, 8] and rec["gates"]["complete"] is True
    assert rec["log"] == str(tmp_path / "logs" / "sharding_N48.log")
    assert not campaign.sharding_path(tmp_path, 48).exists()                               # stored by the campaign wrapper
    stored = campaign.sharding_stage(n=48, args=args, identity="ID", inputs=inputs)
    assert stored["status"] == "pass" and len(calls) == 2
    saved = json.loads(campaign.sharding_path(tmp_path, 48).read_text())
    assert saved["identity"] == "ID" and saved["n"] == 48 and saved["status"] == "pass"
    again = campaign.sharding_stage(n=48, args=args, identity="ID", inputs=inputs)         # the stored record skips
    assert again["status"] == "pass" and len(calls) == 2


def test_run_resumes_from_a_final_record_of_the_same_identity_only(tmp_path, monkeypatch):
    calls = _fake_worker(monkeypatch, tmp_path)
    args, cfg, inputs = _stage_args(tmp_path), campaign.config(), _inputs()
    path = campaign.sharding_path(tmp_path, 32)
    for status in ("pass", "fail"):
        runner.write_json(path, {"identity": "ID", "n": 32, "status": status, "marker": status})
        rec = sharding.run(n=32, args=args, identity="ID", inputs=inputs, cfg=cfg, log=lambda m: None)
        assert rec == {"status": status, "marker": status} and calls == []
    for stored in ({"identity": "OTHER", "n": 32, "status": "pass"}, {"identity": "ID", "n": 32, "status": "pending"},
                   {"identity": "ID", "n": 32, "status": "not_implemented"}):
        runner.write_json(path, stored)
        assert sharding.run(n=32, args=args, identity="ID", inputs=inputs, cfg=cfg, log=lambda m: None)["status"] == "pass"
    assert len(calls) == 3


def test_run_with_a_failing_gate_records_fail_and_the_cap_records_pending(tmp_path, monkeypatch):
    calls = _fake_worker(monkeypatch, tmp_path, make=lambda sp: _raw_result(tuple(sp["shard_counts"]), rel=1e-6))
    args, cfg, inputs = _stage_args(tmp_path), campaign.config(), _inputs()
    rec = sharding.run(n=32, args=args, identity="ID", inputs=inputs, cfg=cfg, log=lambda m: None)
    assert rec["status"] == "fail" and rec["gates"]["rhs"] is False and rec["failures"]
    out = tmp_path / "capped"
    _fake_worker(monkeypatch, tmp_path, calls=calls)
    rec = sharding.run(n=64, args=_stage_args(out, sharding_max_shards=4), identity="ID", inputs=inputs, cfg=cfg,
                       log=lambda m: None)
    assert calls[-1]["spec"]["shard_counts"] == [2, 4]
    assert rec["status"] == "pending" and rec["shard_counts"] == [2, 4] and rec["shard_counts_skipped"] == [8]
    assert rec["shard_counts_requested"] == [2, 4, 8] and rec["max_shards"] == 4
    with pytest.raises(ValueError, match="no shard count"):
        sharding.run(n=64, args=_stage_args(tmp_path / "x", sharding_max_shards=1), identity="ID", inputs=inputs, cfg=cfg,
                     log=lambda m: None)


def test_run_raises_with_the_log_tail_when_the_worker_crashes_or_returns_another_grid(tmp_path, monkeypatch):
    args, cfg, inputs = _stage_args(tmp_path), campaign.config(), _inputs()
    _fake_worker(monkeypatch, tmp_path, returncode=1)
    with pytest.raises(RuntimeError, match=r"(?s)failed \(exit 1\).*fake worker N32"):
        sharding.run(n=32, args=args, identity="ID", inputs=inputs, cfg=cfg, log=lambda m: None)
    assert not campaign.sharding_path(tmp_path, 32).exists()
    _fake_worker(monkeypatch, tmp_path, write=False)
    with pytest.raises(RuntimeError, match="failed"):
        sharding.run(n=32, args=args, identity="ID", inputs=inputs, cfg=cfg, log=lambda m: None)

    def fake(command, stdout=None, stderr=None, env=None, check=None):
        raw = _raw_result((2, 4, 8))
        raw.update(n=32, identity="ELSE")
        Path(command[command.index("--result") + 1]).write_text(json.dumps(raw))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sharding, "subprocess", SimpleNamespace(run=fake, STDOUT=-2))
    with pytest.raises(ValueError, match="another grid / identity"):
        sharding.run(n=32, args=args, identity="ID", inputs=inputs, cfg=cfg, log=lambda m: None)


def test_the_compare_metric_is_relative_to_the_single_device_maximum_in_global_owner_order():
    perm = np.array([2, 0, 1])                                 # plane-major position of each old owner
    single = {("n", "total"): np.array([1.0, -4.0, 2.0]), ("Te", "curvature"): np.zeros(3)}
    pm = np.array([-4.0, 2.0, 1.0])                            # in plane-major order: old [1, 2, 0]
    same = sharding._compare(single, {("n", "total"): pm, ("Te", "curvature"): np.zeros(3)}, perm, None)
    assert same["worst_rel"] == 0.0 and same["bitwise"] is True and same["finite"] is True
    off = sharding._compare(single, {("n", "total"): pm + np.array([0.0, 0.0, 4e-12]), ("Te", "curvature"): np.zeros(3)},
                            perm, None)
    assert off["fields"]["n/total"]["rel"] == pytest.approx(1e-12) and off["bitwise"] is False
    assert off["fields"]["n/total"]["scale"] == 4.0 and off["fields"]["Te/curvature"]["rel"] == 0.0
    nonzero = sharding._compare(single, {("n", "total"): pm, ("Te", "curvature"): np.array([0.0, 1e-30, 0.0])}, perm, None)
    assert nonzero["worst_rel"] == float("inf")                # a nonzero difference against an all-zero term
    bad = sharding._compare(single, {("n", "total"): np.array([np.nan, 2.0, 1.0]), ("Te", "curvature"): np.zeros(3)},
                            perm, None)
    assert bad["finite"] is False and bad["worst_rel"] == float("inf")
    mask = np.array([True, False, True])                       # the metric only sees the masked owners
    masked = sharding._compare({("n", "total"): np.array([1.0, 99.0, 2.0])},
                               {("n", "total"): np.array([2.0, 1.0, 5.0])}, np.array([1, 2, 0]), mask)
    assert masked["worst_rel"] == 0.0


def test_case_arrays_build_the_rhs_and_phi_cases_from_the_adapter_columns(monkeypatch):
    """The worker's case builder: state / phi columns of the variant, boundary data of 5.3's functions, the psi right-hand
    side (reference ``O`` or the discrete ``A psi_bar + B g``)."""
    import scipy.sparse as sp

    from drbx.native import fci_perpendicular_phi_solver as phi_solver
    from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData
    from drbx.native.fci_perpendicular_rhs import FIELDS, PHI
    from p08_step5_combined import jaxstage as jax5

    n_owners, q = 6, 4
    rng = np.random.default_rng(3)
    values = rng.normal(size=(n_owners, 7))
    columns = np.array([5, 1, 2, 0, 4])                                        # n, Te, Ti, omega, phi of the variant
    kinds = ("neumann", "neumann", "neumann", "dirichlet", "dirichlet")
    adapter = SimpleNamespace(owner_values=values, reconstructions={"v": SimpleNamespace(columns=columns, field_kinds=kinds)})
    bc = BoundaryData(rng.normal(size=(q, 5)), rng.normal(size=(q, 2, 5)), rng.normal(size=(2, 5)))
    matrix = sp.csr_matrix(rng.normal(size=(n_owners, n_owners)))
    solver = SimpleNamespace(op=SimpleNamespace(matrix=matrix))
    monkeypatch.setattr(jax5, "variant_boundary_data", lambda prep, variant: bc)
    monkeypatch.setattr(phi_solver, "phi_boundary_term", lambda s_, b: np.asarray(b.dirichlet_value)[:, 0].sum() * np.ones(n_owners))
    prep = SimpleNamespace(adapter=adapter, solver=solver)
    cfg = {"params": {"tau": 1.0}}
    rhs_case, phi_case, got_kinds = sharding._case_arrays(prep, cfg, "v", None)
    assert got_kinds == kinds and rhs_case.name == phi_case.name == "v"
    assert [rhs_case.state[f].tolist() for f in FIELDS] == [values[:, c].tolist() for c in columns[:4]]
    assert rhs_case.phi.tolist() == values[:, 4].tolist() and rhs_case.bc is bc
    assert rhs_case.kinds == dict(zip((*FIELDS, PHI), kinds))
    psi_bar = values[:, 4] + values[:, columns[2]]
    boundary = np.asarray(phi_case.bc.dirichlet_value)[:, 0].sum()
    assert phi_case.rhs_source == "discrete_operator"
    np.testing.assert_allclose(phi_case.rhs, matrix @ psi_bar + boundary, rtol=1e-14)
    assert np.asarray(phi_case.bc.dirichlet_value).shape == (q, 1)               # psi data = phi + tau Ti
    np.testing.assert_allclose(np.asarray(phi_case.bc.dirichlet_value)[:, 0], bc.dirichlet_value[:, 4] + bc.dirichlet_value[:, 2])
    reference = rng.normal(size=n_owners)
    _r, phi_ref, _k = sharding._case_arrays(prep, cfg, "v", reference)
    assert phi_ref.rhs_source == "reference_O" and np.array_equal(phi_ref.rhs, reference)


def test_the_cli_takes_the_sharding_cap():
    base = ["--refreeze-campaign", "/r", "--input-root", "/x", "--output", "/y"]
    assert campaign.parse(["run", *base]).sharding_max_shards is None
    assert campaign.parse(["run-stage", *base, "--stage", "sharding", "--n", "64", "--sharding-max-shards", "4"]
                          ).sharding_max_shards == 4


# ---------------------------------------------------------------------------
# validate / analyze on synthetic outputs
# ---------------------------------------------------------------------------
def _write_grid6(output, cfg, n, *, err_presc, err_solved, band_scale=None, err_psi=1e-3, gates_ok=True, n_owners=60,
                 inputs=None, identity="ID"):
    """A synthetic reduced-ready grid of the step-6 layout: context with P06N-like regions and the six u-bands, merged
    references of the two field sets, per-variant checkpoints. Errors are proportional to the reference; ``band_scale``
    (per band name) multiplies them in that band."""
    inputs = inputs or _inputs()
    volume = 1.0 + 0.1 * np.arange(n_owners)
    u = (np.arange(n_owners) + 0.5) / n_owners
    regions = {"physical_wall": np.arange(n_owners) >= n_owners - 8, "interior": np.arange(n_owners) < n_owners - 8,
               **refs6.band_masks(u, cfg["u_band_edges"])}
    ref5.save_context(output, n, volume, regions)
    refs = _synthetic_references(cfg, n_owners)
    folder = ref5.references_dir(output, n)
    runner.save_npz(folder / ref5.MERGED, **refs)
    runner.write_json(folder / ref5.MANIFEST, {"identity": identity, "sha256": runner.sha256_file(folder / ref5.MERGED),
                                               "n_owners": n_owners})
    jid = checkpoints.jax_identity(identity, n, checkpoints.artifact_sha(inputs, n))
    scale = np.ones(n_owners)
    for name, factor in (band_scale or {}).items():
        scale[regions[name]] = factor
    for variant, spec in cfg["variants"].items():
        fs = spec["field_set"]
        arrays = {}
        for field in reduction.FIELDS:
            for term in ref5.REF_TERMS:
                ref = refs[ref5.ref_key(fs, field, term)]
                arrays[checkpoints.arm_key("presc", field, term)] = ref * (1.0 + err_presc * scale)
                arrays[checkpoints.arm_key("solved", field, term)] = ref * (1.0 + err_solved * scale)
        phi_bar = 0.3 + 0.05 * np.sin(0.2 * np.arange(n_owners))
        arrays.update(phi_bar=phi_bar, ti_bar=np.full(n_owners, 1.0), psi_bar=phi_bar + 1.0,
                      phi_h=phi_bar * (1.0 + err_solved), psi_h=phi_bar * (1.0 + err_solved) + 1.0,
                      psi_N=refs[ref5.psi_key(fs, "O")] * (1.0 + err_psi * scale))
        info = {"variant": variant, "field_set": fs, "constant": False,
                "consistency": {"relative_error_M": 1e-12 if gates_ok else 1e-3, "tolerance": 1e-8, "pass": gates_ok,
                                "solve": {"iterations": 30, "converged": True}},
                "n_minus_o_psi": {"relative": err_psi, "l2": 1.0, "rhs_source": "reference_O"},
                "solved_solve": {"iterations": 55, "converged": True, "seconds": 2.5, "relative_residual": 1e-12,
                                 "residual_norm": 1e-12, "rhs_norm": 1.0},
                "gates": {"consistency": gates_ok, "converged": True, "finite": True}, "gates_pass": gates_ok}
        runner.write_unit(ref5.work_dir(output), checkpoints.variant_unit(n, variant), jid, chunks={"chunk": arrays},
                          started=0.0, extra={"info": info})
    return inputs


def _synthetic_run(output, rf, *, exponent=2.0, gates_ok=True, band_scale=None, shard=None):
    cfg = campaign.config()
    inputs = _inputs(str(rf))
    for n in GRIDS:
        e = 1e-2 * (32.0 / n) ** exponent
        _write_grid6(output, cfg, n, err_presc=e, err_solved=2 * e, err_psi=1e-3 * (32.0 / n) ** 2, gates_ok=gates_ok,
                     inputs=inputs, band_scale=band_scale)
        reduction.reduce_grid(n=n, cfg=cfg, output=output, identity="ID", inputs=inputs)
        if shard:
            runner.write_json(campaign.sharding_path(output, n), {"identity": "ID", "n": n, "status": shard})
    runner.write_json(Path(output) / "preflight.json", {"identity": "ID", "all_pass": True, "cases": {"32": {"all_pass": True}}})
    return inputs


def _vargs(output, grids=GRIDS):
    return SimpleNamespace(output=Path(output), resolutions=list(grids))


def test_reduction_reduces_the_u_bands_like_any_region(tmp_path):
    inputs = _synthetic_run(tmp_path / "o", _fake_refreeze(tmp_path / "rf"))
    summary, results = reduction.load_grid(tmp_path / "o", 48, "ID")
    bands = analysis.band_names(campaign.config())
    assert set(bands) <= set(summary["regions"]) and "global" in summary["regions"]
    assert sum(summary["region_owner_counts"][b] for b in bands) == summary["n_owners"]
    for band in bands:
        key = reduction.row_key("transverse_dirichlet", "presc_vs_ref", "density", "total", band)
        assert results[key]["rel_l2"] == pytest.approx(1e-2 * (32 / 48) ** 2)
    assert inputs["grids"]["32"]["artifact_identity_sha256"] == "art32"


def test_validate_records_gates_pending_sharding_and_the_informational_order(tmp_path):
    rf = _fake_refreeze(tmp_path / "rf")
    inputs = _synthetic_run(tmp_path / "o", rf)
    payload = campaign.validate(args=_vargs(tmp_path / "o"), identity="ID", inputs=inputs)
    saved = json.loads((tmp_path / "o" / "validation.json").read_text())
    assert saved == json.loads(json.dumps(checkpoints.jsonable(payload)))
    assert saved["identity"] == "ID" and saved["acceptance_gate"] is None and saved["user_decides_acceptance"] is True
    assert saved["operator_options"] == FINAL and saved["resolutions"] == [32, 48, 64]
    assert saved["solver_gates_pass"] is True and saved["all_finite"] is True and saved["grids_pass"] is True
    assert saved["preflight_pass"] is True and saved["headline_order_criterion"]["informational"] is True
    assert saved["headline_order_criterion"]["pass"] is True
    # the sharding hook was never run: pending, and it does not fail the grids
    assert saved["sharding"]["status"] == "pending" and saved["sharding"]["pending"] is True
    assert saved["stages_pending"] == ["sharding"]
    assert all(rec["recorded"] is False for rec in saved["sharding"]["grids"].values())
    assert saved["refreeze"]["identity"] == RF_ID and saved["refreeze"]["grids_pass"] is True
    assert saved["grids"]["48"]["solver"]["transverse_dirichlet"]["iterations"] == 55
    assert saved["grids"]["64"]["artifact_identity_sha256"] == "art64"


def test_validate_with_a_stored_sharding_record_pass_pending_or_fail(tmp_path):
    rf = _fake_refreeze(tmp_path / "rf")
    inputs = _synthetic_run(tmp_path / "pass", rf, shard="pass")
    saved = campaign.validate(args=_vargs(tmp_path / "pass"), identity="ID", inputs=inputs)
    assert saved["sharding"]["status"] == "pass" and saved["stages_pending"] == [] and saved["grids_pass"] is True
    inputs = _synthetic_run(tmp_path / "stub", rf, shard="not_implemented")
    saved = campaign.validate(args=_vargs(tmp_path / "stub"), identity="ID", inputs=inputs)
    assert saved["sharding"]["status"] == "pending" and saved["grids_pass"] is True
    inputs = _synthetic_run(tmp_path / "fail", rf, shard="fail")
    saved = campaign.validate(args=_vargs(tmp_path / "fail"), identity="ID", inputs=inputs)
    assert saved["sharding"]["status"] == "fail" and saved["grids_pass"] is False and saved["solver_gates_pass"] is True


def test_validate_requires_the_preflight_and_a_matching_reduction(tmp_path):
    rf = _fake_refreeze(tmp_path / "rf")
    inputs = _synthetic_run(tmp_path / "a", rf)
    with pytest.raises(ValueError, match="requires a matching preflight"):
        campaign.validate(args=_vargs(tmp_path / "a"), identity="OTHER", inputs=inputs)
    runner.write_json(tmp_path / "a" / "preflight.json", {"identity": "ID", "all_pass": False, "cases": {"32": {"all_pass": False}}})
    with pytest.raises(ValueError, match="preflight did not pass"):
        campaign.validate(args=_vargs(tmp_path / "a"), identity="ID", inputs=inputs)
    inputs = _synthetic_run(tmp_path / "b", rf)
    changed = {**inputs, "grids": {**inputs["grids"], "64": {"artifact_identity_sha256": "different"}}}
    with pytest.raises(ValueError, match="computed on another artifact"):
        campaign.validate(args=_vargs(tmp_path / "b"), identity="ID", inputs=changed)
    with pytest.raises(ValueError, match="not recorded"):
        campaign.validate(args=_vargs(tmp_path / "b", (48,)), identity="ID", inputs={"grids": {}})
    saved = campaign.validate(args=_vargs(tmp_path / "b", (48, 64)), identity="ID", inputs=inputs)
    assert saved["resolutions"] == [48, 64]


def test_validate_records_failing_solver_gates_and_a_failing_order_without_raising(tmp_path):
    rf = _fake_refreeze(tmp_path / "rf")
    inputs = _synthetic_run(tmp_path / "gates", rf, gates_ok=False)
    saved = campaign.validate(args=_vargs(tmp_path / "gates"), identity="ID", inputs=inputs)
    assert saved["solver_gates_pass"] is False and saved["grids_pass"] is False and saved["all_finite"] is True
    inputs = _synthetic_run(tmp_path / "order", rf, exponent=1.0)
    saved = campaign.validate(args=_vargs(tmp_path / "order"), identity="ID", inputs=inputs)
    assert saved["grids_pass"] is True and saved["headline_order_criterion"]["pass"] is False


def test_analyze_writes_the_report_the_band_tables_and_the_catalogue_comparison(tmp_path):
    rf = _fake_refreeze(tmp_path / "rf")
    weak = "uband_0.12-0.21"
    inputs = _synthetic_run(tmp_path / "o", rf, band_scale={weak: 30.0})
    payload = campaign.analyze(args=_vargs(tmp_path / "o"), identity="ID", inputs=inputs)
    report = (tmp_path / "o" / "summary" / "step6_report.md").read_text()
    saved = json.loads((tmp_path / "o" / "summary" / "step6_summary.json").read_text())
    assert report.startswith("# P08 step 6") and "P08 step 5.3" not in report.split("\n")[0]
    for heading in ("## u-band errors", "## psi diffusion by u-band", "## Comparison with the re-freeze catalogue",
                    "## Gates", "compact_c3"):
        assert heading in report
    assert saved["schema"] == "drbx.p08-step6-summary.v1" and saved["identity"] == "ID" and saved["grids"] == [32, 48, 64]
    assert saved["operator_options"] == FINAL and saved["u_band_edges"] == campaign.U_BAND_EDGES
    assert set(analysis.band_names(campaign.config())) <= set(saved["regions"])
    # band table: 2 variants x 2 arms x 4 fields x 6 bands, errors scale with the band factor
    rows = saved["u_band_table"]
    assert len(rows) == 2 * 2 * 4 * 6
    by = {(r["variant"], r["cmp"], r["field"], r["band"]): r for r in rows}
    weak_row = by[("transverse_dirichlet", "presc_vs_ref", "density", weak)]
    other = by[("transverse_dirichlet", "presc_vs_ref", "density", "uband_0.27-0.40")]
    assert weak_row["rel_l2"][0] == pytest.approx(30 * other["rel_l2"][0]) and weak_row["order_rel_l2"] == pytest.approx([2.0, 2.0])
    assert len(saved["psi_band_table"]) == 2 * 3 * 6
    # comparison with the catalogue: both catalogue variants, ratio transverse / catalogue
    comparison = saved["catalogue_comparison"]
    assert comparison["catalogue_variants"] == ["main_phi_dirichlet", "dirichlet_rich"] and comparison["grids"] == [32, 48, 64]
    assert len(comparison["rows"]) == 2 * 2 * 2 * 4
    row = next(r for r in comparison["rows"] if r["variant"] == "transverse_dirichlet" and r["catalogue_variant"] ==
               "main_phi_dirichlet" and r["cmp"] == "presc_vs_ref" and r["field"] == "Te")
    assert row["rel_l2_catalogue"] == pytest.approx([1e-4, 1e-4 * (32 / 48) ** 5, 1e-4 * (32 / 64) ** 5])
    assert row["rel_l2_ratio"][0] == pytest.approx(row["rel_l2_transverse"][0] / 1e-4) and row["rel_l2_transverse"][0] > 1e-2
    assert row["order_rel_l2_catalogue"] == pytest.approx([5.0, 5.0])
    assert comparison["phi_psi"][0]["phi_error_rel_l2_catalogue"] == [1e-6] * 3
    assert payload["headline_order_criterion"]["pass"] is True
    with pytest.raises(ValueError, match="another campaign identity"):
        campaign.analyze(args=_vargs(tmp_path / "o"), identity="OTHER", inputs=inputs)
    with pytest.raises(ValueError, match="catalogue summary belongs to another"):
        campaign.analyze(args=_vargs(tmp_path / "o"), identity="ID", inputs={**inputs, "refreeze": {**inputs["refreeze"],
                                                                                                  "identity": "other"}})


def test_catalogue_comparison_tolerates_a_catalogue_without_the_row(tmp_path):
    rf = _fake_refreeze(tmp_path / "rf")
    inputs = _synthetic_run(tmp_path / "o", rf)
    catalogue = json.loads((rf / "summary" / "step5_combined_summary.json").read_text())
    catalogue["orders"] = {}
    catalogue["psi_diffusion"] = {}
    summaries = {n: reduction.load_grid(tmp_path / "o", n, "ID")[0] for n in GRIDS}
    results = {n: reduction.load_grid(tmp_path / "o", n, "ID")[1] for n in GRIDS}
    comparison = analysis.catalogue_comparison(catalogue, ana5.orders(results), summaries, campaign.config())
    assert comparison["rows"][0]["rel_l2_catalogue"] == [None, None, None] and comparison["rows"][0]["rel_l2_ratio"] == [None] * 3
    assert comparison["phi_psi"][0]["n_minus_o_rel_l2_catalogue"] == [None] * 3
    assert inputs["grids"]["32"]["artifact_identity_sha256"] == "art32"


# ---------------------------------------------------------------------------
# references stage, JAX stage and the adapter that feeds 5.3's run_variant
# ---------------------------------------------------------------------------
def _serial_run_stage(calls):
    def run_stage(output, stage, units, identity, *, compute, initializer, initargs, workers, parts=("chunk",),
                  max_tasks_per_worker=None, max_units=None):
        todo = [u for u in units if not runner.valid_unit(output, u, identity, parts=parts)]
        if todo:
            initializer(*initargs)
        for u in todo:
            calls.append((stage, u["start"]))
            compute(u)
        return {"stage": stage, "executed_units": len(todo), "resumed_units": len(units) - len(todo),
                "total_units": len(units), "workers": workers}
    return run_stage


def test_references_stage_runs_5_3_chunks_on_the_transverse_states_and_writes_the_5_3_layout(tmp_path, monkeypatch):
    cfg = {**campaign.config(), "owner_chunk_size": 4}
    n_owners = 10
    volume = 1.0 + 0.1 * np.arange(n_owners)
    regions = {"interior": np.ones(n_owners, bool)}
    seen = {"chunks": [], "edges": None}
    monkeypatch.setattr(refs6, "load_owner_context", lambda n, root, edges: seen.update(edges=edges) or (volume, regions))
    monkeypatch.setattr(refs6, "init_worker", lambda s: refs6._WORKER.update(
        env=None, states=refs6.columns_from_config(s["cfg"], TWO_PI).states(), params=s["cfg"]["params"], work=s["work"],
        identity=s["identity"]))

    def fake_chunk(env, owners, *, states, params):
        seen["chunks"].append((list(owners), sorted(states)))
        out = {"owners": np.asarray(owners)}
        for fs in states:
            for field in ref5.FIELDS:
                for term in ref5.REF_TERMS:
                    out[ref5.ref_key(fs, field, term)] = owners * 1.0
            out[ref5.psi_key(fs, "O")] = owners + 0.5
            out[ref5.psi_key(fs, "R_mid")] = owners + 0.25
        return out

    monkeypatch.setattr(ref5, "reference_chunk", fake_chunk)
    calls = []
    monkeypatch.setattr(runner, "run_stage", _serial_run_stage(calls))
    out = tmp_path / "out"
    manifest = refs6.references_stage(n=32, cfg=cfg, options=FINAL, input_root="/x", sidecar_path="/s", output=out,
                                      identity="ID", workers=2)
    assert manifest["skipped"] is False and manifest["n_owners"] == n_owners and seen["edges"] == campaign.U_BAND_EDGES
    assert [c[0] for c in seen["chunks"]] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert all(c[1] == ["transverse", "transverse_phi_wave"] for c in seen["chunks"])
    merged = ref5.load_references(out, 32, "ID")                          # 5.3's reader accepts this campaign's files
    assert np.array_equal(merged[ref5.psi_key("transverse", "O")], np.arange(n_owners) + 0.5)
    assert set(manifest["field_sets"]) == {"transverse", "transverse_phi_wave"} and manifest["regions"] == ["interior"]
    vol, reg = ref5.load_context_file(out, 32)
    assert np.array_equal(vol, volume) and set(reg) == {"interior"}
    again = refs6.references_stage(n=32, cfg=cfg, options=FINAL, input_root="/x", sidecar_path="/s", output=out,
                                   identity="ID", workers=2)
    assert again["skipped"] is True and len(calls) == 3
    with pytest.raises(ValueError, match="no valid merged references"):
        ref5.load_references(out, 32, "OTHER")


def test_the_adapter_feeds_run_variant_the_right_columns_and_states(tmp_path, monkeypatch):
    from p08_step5_combined import jaxstage as jax5
    from p08_step6_global import jaxstage
    from tests.test_p08_step5_combined_campaign import _prep, _Recorder
    cfg = campaign.config()
    cs = _columns()
    n_own = 6
    values = 1.0 + np.random.default_rng(1).random((n_own, len(cs.names)))
    adapter = jaxstage.build_adapter(cfg, values, cs.names, TWO_PI)
    assert adapter.reconstructions["transverse_dirichlet"].columns.tolist() == [0, 1, 2, 3, 4]
    assert adapter.reconstructions["transverse_phi_wave_dirichlet"].columns.tolist() == [0, 1, 2, 3, 5]
    assert adapter.reconstructions["transverse_dirichlet"].field_kinds == ("dirichlet",) * 5
    assert adapter.exact_state("transverse_dirichlet").fields[-1] == "phi_switch"
    assert adapter.exact_state("transverse_phi_wave_dirichlet").fields[-1] == "phi_wave"
    with pytest.raises(ValueError, match="differ from the configuration"):
        jaxstage.build_adapter(cfg, values, list(cs.names)[::-1], TWO_PI)
    with pytest.raises(ValueError, match="expected"):
        jaxstage.build_adapter(cfg, values[:, :5], cs.names, TWO_PI)

    _cfg5, prep = _prep()
    prep.adapter = adapter
    prep.exact_state = adapter.exact_state
    recorder = _Recorder()
    monkeypatch.setattr(jax5, "perpendicular_rhs", recorder)
    monkeypatch.setattr(jax5, "solve_phi", prep.fake_solve)
    for variant, phi_col in (("transverse_dirichlet", 4), ("transverse_phi_wave_dirichlet", 5)):
        recorder.calls.clear()
        arrays, info = jaxstage.run_variant(prep, variant, cfg, omega_rhs=None, log=lambda m: None)
        assert np.array_equal(arrays["phi_bar"], values[:, phi_col]) and np.array_equal(arrays["ti_bar"], values[:, 2])
        assert info["kinds"] == ["dirichlet"] * 5 and info["field_set"] == cfg["variants"][variant]["field_set"]
        assert info["gates"]["finite"] is True
        assert np.array_equal(recorder.calls[0]["state"]["vorticity"], values[:, 3])      # the owner average of omega
        assert np.array_equal(recorder.calls[0]["phi"], values[:, phi_col])               # prescribed arm


def _jax_stage_world(tmp_path, monkeypatch, *, owner_volume=None):
    from p08_step6_global import jaxstage
    cfg = campaign.config()
    out = tmp_path / "out"
    n_own = 12
    inputs = _inputs()
    _write_grid6(out, cfg, 32, err_presc=1e-3, err_solved=2e-3, n_owners=n_own, inputs=inputs)
    volume = ref5.load_context_file(out, 32)[0]
    for unit in list(ref5.work_dir(out).glob("_chunks/N32/jax_*")):          # drop the synthetic jax checkpoints
        import shutil
        shutil.rmtree(unit)
    values = np.random.default_rng(2).random((n_own, 6))
    path = refs6.owner_values_path(out, 32)
    runner.save_npz(path, values=values, columns=np.asarray(_columns().names))
    runner.write_json(refs6.owner_values_manifest_path(out, 32), {"identity": "ID", "sha256": runner.sha256_file(path),
                                                                 "n_owners": n_own})
    calls = {"prepare": 0, "variants": []}

    def fake_prepare(**kw):
        calls["prepare"] += 1
        calls["prepare_kw"] = kw
        return SimpleNamespace(owner_volume=volume if owner_volume is None else owner_volume, setup_seconds={"x": 1.0},
                               plan_summary={"cells": 1}, artifact_identity_sha256="art32")

    def fake_run_variant(prep, variant, cfg_, *, omega_rhs=None, log=None):
        calls["variants"].append((variant, np.asarray(omega_rhs)))
        arrays = {"phi_bar": np.zeros(n_own)}
        return arrays, {"variant": variant, "gates": {"finite": True}, "gates_pass": True}

    monkeypatch.setattr(jaxstage, "prepare", fake_prepare)
    monkeypatch.setattr(jaxstage, "run_variant", fake_run_variant)
    return cfg, out, inputs, calls, values


def test_jax_stage_checkpoints_every_variant_feeds_the_matching_o_psi_and_resumes(tmp_path, monkeypatch):
    from p08_step6_global import jaxstage
    cfg, out, inputs, calls, values = _jax_stage_world(tmp_path, monkeypatch)
    kw = dict(n=32, cfg=cfg, options=FINAL, artifact_root=Path("/rf/artifact"), sidecar_path=out / "localized_sidecar.json",
              input_root=Path("/ws"), output=out, identity="ID", inputs=inputs, log=lambda m: None)
    summary = jaxstage.jax_stage(**kw)
    assert summary["computed"] == list(cfg["variants"]) and summary["resumed"] == [] and calls["prepare"] == 1
    refs = ref5.load_references(out, 32, "ID")
    for variant, omega_rhs in calls["variants"]:
        assert np.array_equal(omega_rhs, refs[ref5.psi_key(cfg["variants"][variant]["field_set"], "O")])
    pk = calls["prepare_kw"]
    assert pk["artifact_root"] == Path("/rf/artifact") and pk["options"] == FINAL and np.array_equal(pk["owner_values"], values)
    assert pk["column_names"] == list(_columns().names) and pk["inputs"] is inputs
    jid = checkpoints.jax_identity("ID", 32, "art32")
    for variant in cfg["variants"]:
        assert runner.valid_unit(ref5.work_dir(out), checkpoints.variant_unit(32, variant), jid)
    again = jaxstage.jax_stage(**kw)
    assert again["computed"] == [] and again["resumed"] == list(cfg["variants"]) and calls["prepare"] == 1
    saved = json.loads((out / "N32" / "jax_stage.json").read_text())
    assert saved["owner_values_sha256"] == runner.sha256_file(refs6.owner_values_path(out, 32))


def test_jax_stage_refuses_missing_owner_values_and_other_owner_volumes(tmp_path, monkeypatch):
    from p08_step6_global import jaxstage
    cfg, out, inputs, calls, _values = _jax_stage_world(tmp_path, monkeypatch, owner_volume=np.ones(12))
    kw = dict(n=32, cfg=cfg, options=FINAL, artifact_root=Path("/rf/artifact"), sidecar_path=out / "s.json",
              input_root=Path("/ws"), output=out, identity="ID", inputs=inputs, log=lambda m: None)
    with pytest.raises(ValueError, match="owner volumes of the references stage differ"):
        jaxstage.jax_stage(**kw)
    refs6.owner_values_path(out, 32).unlink()
    with pytest.raises(ValueError, match="no valid owner values"):
        jaxstage.jax_stage(**kw)


def test_prepare_requires_a_recorded_matching_artifact_identity(tmp_path, monkeypatch):
    from p_shared import replay_support
    from p08_step2_global import replay
    from p08_step6_global import jaxstage
    cfg = campaign.config()
    monkeypatch.setattr(replay_support, "build_environment", lambda **kw: SimpleNamespace(t=SimpleNamespace(vol=np.ones(3))))
    monkeypatch.setattr(replay, "load_artifact", lambda root, n: {"identity": _identity_json(FINAL, n=n)})
    kw = dict(n=32, cfg=cfg, options=FINAL, artifact_root=tmp_path, sidecar_path=tmp_path / "s.json", input_root=tmp_path,
              owner_values=np.zeros((3, 6)), column_names=_columns().names, log=lambda m: None)
    with pytest.raises(ValueError, match="differs from provenance"):
        jaxstage.prepare(inputs={"grids": {}}, **kw)
    with pytest.raises(ValueError, match="differs from provenance"):
        jaxstage.prepare(inputs={"grids": {"32": {"artifact_identity_sha256": "other"}}}, **kw)
    monkeypatch.setattr(replay, "load_artifact", lambda root, n: {"identity": _identity_json(THREE, n=n)})
    with pytest.raises(ValueError, match="not the pinned"):
        jaxstage.prepare(inputs={"grids": {"32": {"artifact_identity_sha256": "other"}}}, **kw)
