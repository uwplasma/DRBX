"""P10 evolved MMS, chunk C1 (bundle): round trip, identity checks and mismatch rejection on the n = 16 synthetic bundle (no HSX files)."""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from p10_evolved_mms import bundle as B          # noqa: E402
from p10_evolved_mms import synthetic as syn    # noqa: E402


@pytest.fixture(scope="module")
def b():
    return syn.synthetic_bundle()


def _inputs(arm="raw"):
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = syn.synthetic_inputs(16, 8, arm)
    return arm, layout, nodal, copy.deepcopy(nodal_meta), dict(lap), copy.deepcopy(lap_meta), dict(prov)


def _assemble(**kw):
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs(kw.pop("arm", "raw"))
    nodal = kw.get("nodal", nodal)
    lap = kw.get("lap", lap)
    nodal_meta = kw.get("nodal_meta", nodal_meta)
    lap_meta = kw.get("lap_meta", lap_meta)
    return B.assemble_bundle(arm, layout, nodal, nodal_meta, lap, lap_meta, prov, build_preconditioner=False)


def test_synthetic_bundle_contents_and_identity(b):
    E, P, N = b.E, b.P, b.N
    ref = b.ref
    assert (E, P, N) == (8, 254, 16) and ref.points.shape == (E, P, 3) and ref.wall_points.shape == (E, N, 3)
    assert ref.A.shape == (E, P, 3, 3) and ref.divA.shape == ref.h.shape == ref.K.shape == ref.ginv_u.shape == (E, P, 3)
    assert ref.jac.shape == ref.B.shape == (E, P) and ref.wall_A_row.shape == ref.wall_ginv_u.shape == (E, N, 3)
    assert b.ctx.prec is not None and b.ctx.curvature_flux.shape == (E, P, 3)
    idn = b.identity
    assert idn["agreement"] == {"jac": "bitwise", "Hp": "bitwise"}
    assert idn["arm"] == "raw" and idn["arm_identity"] is None and idn["n"] == 16 and idn["family"] == "A"
    for key in ("layout_sha256", "nodal_plan_sha256", "laplacian_plan_sha256", "bundle_sha256", "arrays_sha256",
                "nodal_metric_identity", "laplacian_metric_identity"):
        assert re.fullmatch(r"[0-9a-f]{64}", idn[key]), key
    assert idn["git_commit"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", idn["git_commit"])
    assert idn["preconditioner"]["method"] == "core_schur" and idn["preconditioner"]["factor_dtype"] == "float32"
    # the plans carry the same norm bitwise and the wall points are the layout's wall lattice
    assert np.array_equal(np.asarray(b.ctx.plan.Hp), np.asarray(b.ctx.lplan.Hp))
    np.testing.assert_allclose(np.asarray(b.H), np.asarray(b.ctx.lplan.Hp) * b.ctx.lplan.structure.deta)
    # the logical |J| of the reference arrays is the one of both plans
    assert float(jnp.min(ref.jac)) > 0 and float(jnp.abs(ref.wall_ginv_u[..., 0]).min()) > 0


def test_bundle_is_a_pytree_jit_argument(b):
    f = jax.jit(lambda bundle: jnp.sum(bundle.ref.jac) + jnp.sum(bundle.ctx.curvature_flux))
    v1, v2 = f(b), f(b)
    assert f._cache_size() == 1 and float(v1) == float(v2)
    assert float(v1) == pytest.approx(float(jnp.sum(b.ref.jac) + jnp.sum(b.ctx.curvature_flux)), rel=1e-14)


def test_round_trip(tmp_path, b):
    B.save_bundle(b, tmp_path / "bundle")
    assert (tmp_path / "bundle" / "bundle.npz").exists() and (tmp_path / "bundle" / "identity.json").exists()
    stored = json.loads((tmp_path / "bundle" / "identity.json").read_text())
    assert stored["bundle_sha256"] == b.identity["bundle_sha256"] and stored["synthetic"] is True
    c = B.load_bundle(tmp_path / "bundle", expected_identity=b.identity["bundle_sha256"])
    for name in B.RefArrays._fields:
        assert np.array_equal(np.asarray(getattr(c.ref, name)), np.asarray(getattr(b.ref, name))), name
    for key in ("nodal_plan_sha256", "laplacian_plan_sha256", "layout_sha256", "arrays_sha256", "bundle_sha256", "arm_identity",
                "arm", "n", "agreement", "nodal_metric_identity", "laplacian_metric_identity"):
        assert c.identity[key] == b.identity[key], key
    assert (c.E, c.P, c.N) == (b.E, b.P, b.N)
    # the rebuilt context reproduces the potential solve (deterministic factorization) and the curvature flux
    assert np.array_equal(np.asarray(c.ctx.curvature_flux), np.asarray(b.ctx.curvature_flux))
    from drbx.native.fci_perpendicular_sbp_laplacian_solve import solve_dirichlet
    from drbx.native.fci_perpendicular_sbp_laplacian import LaplacianBoundaryData

    s = jnp.cos(b.ref.points[..., 0] * 3.0 + b.ref.points[..., 2])
    wall = LaplacianBoundaryData(value=(jnp.zeros((b.E, b.N)),))
    xb, ib = solve_dirichlet(b.ctx.lplan, s, wall, b.ctx.prec, rtol=1e-12, maxit=100)
    xc, ic = solve_dirichlet(c.ctx.lplan, s, wall, c.ctx.prec, rtol=1e-12, maxit=100)
    assert bool(ib["converged"]) and bool(ic["converged"])
    np.testing.assert_allclose(np.asarray(xc), np.asarray(xb), rtol=0, atol=1e-10 * float(jnp.abs(xb).max()))


def test_expected_identity_accepted_and_rejected(tmp_path, b):
    out = B.save_bundle(b, tmp_path / "bundle")
    ok = {"arm": "raw", "n": 16, "arm_identity": None, "layout_sha256": b.identity["layout_sha256"],
          "nodal_metric_identity": b.identity["nodal_metric_identity"]}
    B.load_bundle(out, expected_identity=ok, build_preconditioner=False)
    with pytest.raises(ValueError, match="mismatch"):
        B.load_bundle(out, expected_identity={**ok, "arm": "filtered"}, build_preconditioner=False)
    with pytest.raises(ValueError, match="mismatch"):
        B.load_bundle(out, expected_identity={**ok, "arm_identity": {"quantity": "J*B^i"}}, build_preconditioner=False)
    with pytest.raises(ValueError, match="mismatch"):
        B.load_bundle(out, expected_identity={"nodal_metric_identity": "0" * 64}, build_preconditioner=False)
    with pytest.raises(ValueError, match="expected"):
        B.load_bundle(out, expected_identity="0" * 64, build_preconditioner=False)


def test_tampered_bundle_rejected(tmp_path, b):
    out = B.save_bundle(b, tmp_path / "bundle")
    with np.load(out / "bundle.npz") as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    arrays["B"] = arrays["B"] * (1.0 + 1e-9)
    np.savez(out / "bundle.npz", **arrays)
    with pytest.raises(ValueError, match="arrays_sha256"):
        B.load_bundle(out, build_preconditioner=False)
    out2 = B.save_bundle(b, tmp_path / "bundle2")
    idn = json.loads((out2 / "identity.json").read_text())
    idn["arm"] = "filtered"
    (out2 / "identity.json").write_text(json.dumps(idn))
    with pytest.raises(ValueError, match="bundle_sha256"):
        B.load_bundle(out2, build_preconditioner=False)


def test_mismatched_jacobian_rejected_and_tiny_difference_reported():
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs()
    bad = dict(lap)
    bad["J"] = lap["J"] * (1.0 + 1e-6 * np.cos(np.arange(lap["J"].size)).reshape(lap["J"].shape))
    with pytest.raises(ValueError, match=r"\|J\|"):
        B.assemble_bundle(arm, layout, nodal, nodal_meta, bad, lap_meta, prov, build_preconditioner=False)
    near = dict(lap)
    near["J"] = lap["J"] * (1.0 + 1e-14 * np.cos(np.arange(lap["J"].size)).reshape(lap["J"].shape))
    bundle = B.assemble_bundle(arm, layout, nodal, nodal_meta, near, lap_meta, prov, build_preconditioner=False)
    assert "max_rel" in bundle.identity["agreement"]["jac"] and bundle.identity["agreement"]["Hp"] != "bitwise"
    assert np.array_equal(np.asarray(bundle.ctx.plan.Hp), np.asarray(bundle.ctx.lplan.Hp))      # context built, Hp bitwise
    with pytest.raises(ValueError, match="shapes"):
        B.assemble_bundle(arm, layout, nodal, nodal_meta, {**lap, "J": lap["J"][:, :-1]}, lap_meta, prov, build_preconditioner=False)


def test_arm_layout_and_node_mismatches_rejected():
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs("filtered")
    with pytest.raises(ValueError, match="arm identity mismatch"):
        B.assemble_bundle(arm, layout, nodal, nodal_meta, lap, {**lap_meta, "eta_filter": None}, prov, build_preconditioner=False)
    other = {**syn.SYNTHETIC_ARM_FILTER, "arm_sha256": "1" * 64}
    with pytest.raises(ValueError, match="arm identity mismatch"):
        B.assemble_bundle(arm, layout, nodal, nodal_meta, lap, {**lap_meta, "eta_filter": other}, prov, build_preconditioner=False)
    with pytest.raises(ValueError, match="arm"):
        B.assemble_bundle("raw", layout, nodal, nodal_meta, lap, lap_meta, prov, build_preconditioner=False)
    with pytest.raises(ValueError, match="layout"):
        B.assemble_bundle(arm, layout, nodal, nodal_meta, lap, {**lap_meta, "layout_sha256": "0" * 64}, prov, build_preconditioner=False)
    pts = np.array(lap["points"])
    pts[..., 1] += 1e-6
    with pytest.raises(ValueError, match="nodes"):
        B.assemble_bundle(arm, layout, nodal, nodal_meta, {**lap, "points": pts}, lap_meta, prov, build_preconditioner=False)
    wp = np.array(lap["wall_points"])
    wp[..., 2] += 1e-6
    with pytest.raises(ValueError, match="wall points"):
        B.assemble_bundle(arm, layout, nodal, nodal_meta, {**lap, "wall_points": wp}, lap_meta, prov, build_preconditioner=False)


def test_identity_changes_with_the_arm_options():
    raw, filtered = _assemble(arm="raw"), _assemble(arm="filtered")
    assert raw.identity["arm_identity"] is None and filtered.identity["arm_identity"]["quantity"] == "synthetic"
    assert raw.identity["bundle_sha256"] != filtered.identity["bundle_sha256"]
    assert raw.identity["nodal_plan_sha256"] == filtered.identity["nodal_plan_sha256"]       # same synthetic geometry
    # the arm identity (the eta-filter option incl. its arm sha) enters the bundle sha
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs("filtered")
    other = {**syn.SYNTHETIC_ARM_FILTER, "max_harmonic_per_period": 4}
    nodal_meta["eta_filter"] = lap_meta["eta_filter"] = other
    changed = B.assemble_bundle(arm, layout, nodal, nodal_meta, lap, lap_meta, prov, build_preconditioner=False)
    assert changed.identity["bundle_sha256"] != filtered.identity["bundle_sha256"]
    # a different metric identity changes it too
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = _inputs("raw")
    prov["nodal_metric_identity"] = "f" * 64
    c2 = B.assemble_bundle(arm, layout, nodal, nodal_meta, lap, lap_meta, prov, build_preconditioner=False)
    assert c2.identity["bundle_sha256"] != raw.identity["bundle_sha256"]


def test_metric_paths_and_unknown_arm():
    nodal, lap = B.metric_paths("filtered", 48)
    assert nodal.parts[-3:] == ("p09_m3_filtered_20261004", "N48", "nodal_metric.npz")
    assert lap.parts[-4:] == ("p09_m5_laplacian_20261004", "filtered", "N48", "laplacian_metric.npz")
    assert B.metric_paths("raw", 32)[0].parts[-3] == "p09_m3_campaign_20261004"
    with pytest.raises(ValueError, match="arm"):
        B.metric_paths("bogus", 32)
