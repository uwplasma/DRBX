"""Checked field-period factor reuse: N16 synthetic periodic P, Dirichlet, all core/ring nodes."""
from __future__ import annotations

import jax
import numpy as np
import pytest

from drbx.geometry.sbp_laplacian_assembly import iter_inplane_blocks
from drbx.native.fci_perpendicular_plane_preconditioner import (
    apply_core_schur_preconditioner, build_core_schur_preconditioner)
from drbx.native.fci_perpendicular_sbp_laplacian import laplacian_form
from drbx.native.fci_perpendicular_sbp_laplacian_solve import build_dirichlet_preconditioner, solve_dirichlet
from tests.sbp_sharding_case import periodic_laplacian_case
from tests.test_fci_perpendicular_sbp_sharding import _subprocess


@pytest.fixture(scope="module")
def periodic():
    return periodic_laplacian_case()


def leaves(prec):
    return [np.asarray(a) for a in jax.tree_util.tree_leaves(prec)]


@pytest.mark.parametrize("group", [1, 3, 16])
def test_period_leaves_storage_and_logical_metadata(periodic, group):
    lay, lp = periodic
    full = build_dirichlet_preconditioner(lp, factor_dtype="float32", group_planes=group)
    reused = build_dirichlet_preconditioner(lp, factor_dtype="float32", group_planes=group, period_nfp=4)
    assert full.meta.period_nfp == 1 and reused.meta.period_nfp == 4
    assert reused.n_owners == full.n_owners == lay.n_eta * lay.P
    assert reused.nbytes * 4 == full.nbytes
    assert reused.info["period_reuse"]["active"] and reused.info["period_reuse"]["stored_planes"] == 4
    for a, b, axis in zip(leaves(full), leaves(reused), [1, 1, 0, 0]):
        assert np.array_equal(a, np.concatenate([b] * 4, axis=axis))


def test_same_solution_and_iteration_count(periodic):
    lay, lp = periodic
    rng = np.random.default_rng(55)
    star = rng.standard_normal((lay.n_eta, lay.P))
    rhs = -np.asarray(laplacian_form(lp, star, None, "dirichlet", None, 1.0)) / (lp.Hp * lp.structure.deta)
    solve = jax.jit(lambda p: solve_dirichlet(lp, rhs, None, p, rtol=1e-11, maxit=200))
    full = build_dirichlet_preconditioner(lp, factor_dtype="float32")
    period = build_dirichlet_preconditioner(lp, factor_dtype="float32", period_nfp=4)
    xf, fi = solve(full)
    xp, pi = solve(period)
    assert bool(fi["converged"]) and bool(pi["converged"])
    assert int(fi["iterations"]) == int(pi["iterations"])
    np.testing.assert_allclose(xp, xf, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("offset,planes", [(0, 16), (1, 7), (5, 3), (8, 8)])
def test_jitted_apply_handles_partial_and_offset_periods(periodic, offset, planes):
    lay, lp = periodic
    # f64 isolates the unchanged algebra from f32 batching roundoff.
    full = build_dirichlet_preconditioner(lp, factor_dtype="float64")
    period = build_dirichlet_preconditioner(lp, factor_dtype="float64", period_nfp=4)
    r = np.random.default_rng(8).standard_normal(lay.n_eta * lay.P)
    ref = np.asarray(apply_core_schur_preconditioner(full, r)).reshape(lay.n_eta, lay.P)[offset:offset + planes]
    got = jax.jit(lambda p, b, k: apply_core_schur_preconditioner(p, b, plane_offset=k))(
        period, r.reshape(lay.n_eta, lay.P)[offset:offset + planes].reshape(-1), offset)
    np.testing.assert_allclose(np.asarray(got).reshape(ref.shape), ref, rtol=1e-10, atol=1e-12)


def test_nonperiodic_state_coefficient_falls_back_exactly(periodic):
    lay, lp = periodic
    coeff = np.ones((lay.n_eta, lay.P)); coeff[-1] *= 1.05
    full = build_dirichlet_preconditioner(lp, coeff, factor_dtype="float32")
    checked = build_dirichlet_preconditioner(lp, coeff, factor_dtype="float32", period_nfp=4)
    assert checked.meta.period_nfp == 1 and checked.nbytes == full.nbytes
    info = checked.info["period_reuse"]
    assert not info["active"] and info["reason"] == "matrix_mismatch"
    assert info["max_relative_difference"] > info["matrix_rtol"]
    assert all(np.array_equal(a, b) for a, b in zip(leaves(full), leaves(checked)))


def test_nondivisible_planes_and_explicit_disable(periodic):
    _, lp = periodic
    full = build_dirichlet_preconditioner(lp)
    checked = build_dirichlet_preconditioner(lp, period_nfp=3)
    assert checked.info["period_reuse"]["reason"] == "plane_count_not_divisible"
    assert full.info["period_reuse"]["reason"] == "disabled"
    assert all(np.array_equal(a, b) for a, b in zip(leaves(full), leaves(checked)))


@pytest.mark.parametrize("nfp", [0, -1, 1.5, True])
def test_invalid_period_option(periodic, nfp):
    with pytest.raises(ValueError, match="period_nfp"):
        build_dirichlet_preconditioner(periodic[1], period_nfp=nfp)


def test_one_shot_iterator_cannot_silently_lose_fallback_matrices(periodic):
    _, lp = periodic; st = lp.structure
    with pytest.raises(ValueError, match="replayable"):
        build_core_schur_preconditioner(iter_inplane_blocks(lp), n_planes=st.n_eta, n_core=st.Nc,
                                        n_rings=st.m, ring_size=st.N, period_nfp=4)


def test_checked_group_validation(periodic):
    _, lp = periodic; st = lp.structure
    groups = list(iter_inplane_blocks(lp, group_planes=4))
    kw = dict(n_planes=st.n_eta, n_core=st.Nc, n_rings=st.m, ring_size=st.N, period_nfp=4)
    with pytest.raises(ValueError, match="cover"):
        build_core_schur_preconditioner(groups[:-1], **kw)
    bad = [(k0, k1, a.copy()) for k0, k1, a in groups]
    bad[-1][2].data[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_core_schur_preconditioner(bad, **kw)


def test_periodic_sharded_solve_on_four_host_devices():
    result = _subprocess("period_cg")
    assert result["devices"] == 4
    for row in result["cases"]:
        assert row["active"] and row["logical_bytes"] * row["nfp"] == row["full_bytes"]
        assert row["iterations"] == row["reference_iterations"]
        assert row["relative_difference"] < 1e-10
        assert row["converged"]
