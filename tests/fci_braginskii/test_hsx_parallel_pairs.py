"""Parallel operator pairs on the real HSX model (set DRBX_TEST_GEOMETRY_BUNDLE).

Adjointness of every pair, the Wiesenberger step-modulated advection along b and the
Stegmeir anisotropic heat blob.  The dense operators come from the production model
captured at its first ``shard_map`` (one device, 16^3).
"""

from __future__ import annotations

import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse.linalg as ssl
from jax.sharding import PartitionSpec as P

BUNDLE = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(BUNDLE is None, reason="set DRBX_TEST_GEOMETRY_BUNDLE")]
EPSILONS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


class _Captured(Exception):
    pass


def _capture(tmp_path):
    from drbx import deck

    path = tmp_path / "hsx.toml"
    path.write_text(f'[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\ngeometry = "{BUNDLE}"\noutput = "{tmp_path / "o" / "hsx"}"\n')
    name, cfg = deck.load_deck(path)
    cfg["fci_braginskii"] = dataclasses.replace(cfg["fci_braginskii"], final_time=0.0075, num_steps=10)
    real = jax.shard_map

    def stop(fun, **kwargs):
        raise _Captured(sys._getframe(1).f_locals)

    jax.shard_map = stop
    try:
        deck.BACKENDS[name].run(cfg, lambda message: None)
    except _Captured as caught:
        return caught.args[0]
    finally:
        jax.shard_map = real
    raise RuntimeError("the run never reached its first shard_map")


@pytest.fixture(scope="module")
def operators(tmp_path_factory):
    cap = _capture(tmp_path_factory.mktemp("hsx"))
    state = cap["state"]
    shape = tuple(int(n) for n in state.density.shape)

    def build(model, st):
        face_bc, ctx = model._face_bcs(st), model._stencil_builder_context()
        gs, ds, core = model._fci_support_core_pair(face_bc=face_bc, context=ctx)
        gl, dl, length = model._fci_leg_pair(face_bc=face_bc, context=ctx)
        zero = jnp.zeros(shape)
        gp, dp, _ = model._fci_current_phi_boundary_pair(face_bc=face_bc, context=ctx, wall_endpoint_current_values=(zero, zero))
        n = int(np.prod(shape))
        eye = jnp.eye(n).reshape((n,) + shape)
        dense = lambda op: jax.lax.map(lambda e: op(e).reshape(-1), eye, batch_size=64).T
        return {"Gs": dense(gs), "Ds": dense(ds), "Gl": dense(gl), "Dl": dense(dl), "Gp": dense(gp), "Dp": dense(dp),
                "length": length, "volume": model._fci_pair_cell_mass(), "active": model.geometry.active_cell_mask_owned}

    spatial, geo = cap["spatial_spec"], cap["geometry_spec"]

    def compiled(st):
        body = lambda s, cf, mf, cv: build(cap["build_local_model"](cf, mf, cv), s)
        mapped = jax.shard_map(body, mesh=cap["mesh"], in_specs=(cap["state_spec"], geo, geo, geo), out_specs=P(), check_vma=False)
        return mapped(st, cap["cell_fields"], cap["map_fields"], cap["control_volume_fields"])

    mats = jax.tree.map(np.asarray, jax.jit(compiled)(state))
    act = np.flatnonzero(mats["active"].ravel())
    sub = lambda m: m[np.ix_(act, act)]
    out = {k: sub(mats[k]) for k in ("Gs", "Ds", "Gl", "Dl", "Gp", "Dp")}
    out.update(vol=mats["volume"].ravel()[act], length=mats["length"].ravel()[act], act=act, cap=cap, shape=shape)
    return out


def test_every_pair_is_an_exact_adjoint(operators):
    v = operators["vol"]
    f = np.random.default_rng(0).standard_normal((len(v), 8))
    for g, d in (("Gs", "Ds"), ("Gl", "Dl"), ("Gp", "Dp")):
        G, D = operators[g], operators[d]
        assert np.linalg.norm(D + (G.T * v[None]) / v[:, None]) <= 1e-13 * np.linalg.norm(D), g
        if g != "Gp":  # the current/phi pair carries the wall current: not sum-free by design
            assert np.max(np.abs(v @ (D @ f)) / (np.abs(v)[:, None] * np.abs(D @ f)).sum(0)) <= 1e-13, g
    assert np.max(np.abs(operators["Gs"].sum(1))) <= 1e-13 * np.max(np.abs(operators["Gs"]).sum(1))
    for g, d in (("Gs", "Ds"), ("Gl", "Dl")):
        vl = v[:, None] * (operators[d] @ operators[g])  # conduction operator
        assert np.linalg.norm(vl - vl.T) <= 1e-13 * np.linalg.norm(vl)
        assert np.max(sla.eigvalsh(0.5 * (vl + vl.T))) <= 1e-12 * np.linalg.norm(vl)


def _blob(operators):
    from drbx.fci_braginskii import run as hsx_run

    geometry = operators["cap"]["global_geometry"]
    traced, valid = hsx_run._trace_logical_labels_to_eta_plane(geometry, reference_eta=np.pi, tracing_substeps_per_plane=4)
    u, _, eta = np.meshgrid(*(np.asarray(g.centers) for g in (geometry.grid.x, geometry.grid.y, geometry.grid.z)), indexing="ij")
    dist2 = traced[..., 0] ** 2 + 0.65**2 - 2 * traced[..., 0] * 0.65 * np.cos(traced[..., 1] - 0.5)
    profile = np.exp(-dist2 / (2 * 0.1**2)) * valid
    act = operators["act"]
    return dict(profile=profile, dist2=dist2, u=u, eta=eta, valid=valid, act=act)


def test_step_advection_along_b_is_bounded_with_face_flux_dissipation(operators, capsys):
    """Wiesenberger: step-modulated aligned blob advected 8 legs; report -D alone and with eps."""
    b, v, act = _blob(operators), operators["vol"], operators["act"]
    dx = 2 * np.pi / operators["shape"][2]
    f0 = (b["profile"] * (np.abs(b["eta"] - np.pi) < np.pi / 2)).ravel()[act]
    rows = {}
    for eps in EPSILONS:
        A = -operators["Ds"] + eps * (operators["Dl"] * operators["length"][None, :]) @ operators["Gl"]
        fT = ssl.expm_multiply(8 * dx * A, f0)
        rows[eps] = (fT.min(), fT.max(), np.sqrt(np.sum(v * fT**2) / np.sum(v * f0**2)), np.sum(v * fT) / np.sum(v * f0) - 1.0)
        with capsys.disabled():
            print(f"[step advection] eps={eps}: min/max {rows[eps][0]:+.3f}/{rows[eps][1]:+.3f}, L2 growth {rows[eps][2]:.3f}, mass {rows[eps][3]:+.1e}")
    assert all(abs(row[3]) < 1e-12 for row in rows.values())  # the dissipation is conservative
    assert rows[2.0][2] < rows[0.0][2]


def test_anisotropic_heat_blob_reports_perpendicular_leakage(operators, capsys):
    """Stegmeir heat blob (eps_perp = 0): L1 error vs the aligned steady state, central and compact."""
    b, v, act = _blob(operators), operators["vol"], operators["act"]
    dx = 2 * np.pi / operators["shape"][2]
    t0 = (b["profile"] * np.cos(0.5 * (b["eta"] - np.pi)) ** 2).ravel()[act]
    pf = b["profile"].ravel()[act]
    ideal = pf * np.sum(v * t0) / np.sum(v * pf)
    inside = (b["u"] < 0.8).ravel()[act] & b["valid"].ravel()[act]
    for label, a in (("D G", operators["Ds"] @ operators["Gs"]), ("D+ G+", operators["Dl"] @ operators["Gl"])):
        values, vectors = sla.eig(a)
        coefficients = np.linalg.solve(vectors, t0.astype(complex))
        errors = []
        for t in (1.0, 1e2, 1e4):
            tt = (vectors @ (np.exp(t * values / dx**2) * coefficients)).real
            errors.append(np.sum((v * np.abs(tt - ideal))[inside]) / np.sum((v * ideal)[inside]))
        with capsys.disabled():
            print(f"[heat blob {label}] L1 error at chi t/dx^2 = 1, 1e2, 1e4: " + ", ".join(f"{e:.3f}" for e in errors))
        assert np.all(np.isfinite(errors)) and np.max(values.real) < 1e-9  # no growth; the leakage is reported
