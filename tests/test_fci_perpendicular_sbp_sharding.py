"""Eta-sharded nodal SBP bracket, curvature, Laplacian and CG: multi-device agreement (subprocess with forced host devices) and plan checks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax.numpy as jnp

from drbx.native.fci_perpendicular_plane_preconditioner import apply_core_schur_preconditioner
from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket
from drbx.native.fci_perpendicular_sbp_laplacian import (
    F_HALO,
    PLAN_HALO,
    LaplacianBoundaryData,
    extend_laplacian_plan,
    laplacian_action,
    laplacian_action_ext,
    laplacian_form,
    laplacian_form_ext,
)
from drbx.native.fci_perpendicular_sbp_laplacian_solve import build_dirichlet_preconditioner
from drbx.native.fci_perpendicular_sbp_sharding import (
    CURVATURE_HALO,
    NODAL_HALO,
    laplacian_plan_specs,
    make_plane_mesh,
    nodal_plan_specs,
    shard_core_schur_preconditioner,
    shard_laplacian_plan,
    shard_nodal_plan,
    sharded_laplacian_action,
    sharded_laplacian_form,
    sharded_sbp_bracket,
    sharded_solve_dirichlet,
)
from tests import sbp_sharding_case as case

CASE = Path(case.__file__).resolve()


def _subprocess(*args, devices: int = 4, timeout: int = 900) -> dict:
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={devices}".strip()
    env["JAX_PLATFORMS"] = "cpu"
    done = subprocess.run([sys.executable, str(CASE), *args], capture_output=True, text=True, env=env,
                          timeout=timeout, check=False)
    assert done.returncode == 0, f"subprocess failed ({done.returncode}):\n{done.stderr[-4000:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_halo_constant():
    assert NODAL_HALO == 3


def test_sharded_matches_single_device_in_subprocess():
    res = _subprocess("agree")
    assert res["devices"] == 4
    for s in (1, 2, 4):
        assert res[f"S{s}"] <= 1e-14, res
    assert res["ref_max"] > 0


def test_family_a_core_sharded_matches_single_device_in_subprocess():
    res = _subprocess("family_a")
    assert res["P"] == 254 and res["ref_max"] > 0
    for s in (1, 2, 4):
        assert res[f"S{s}"] <= 1e-14, res


def test_too_few_planes_per_shard_raises():
    res = _subprocess("short")
    assert res["S4"] is not None and "halo" in res["S4"]
    assert res["S8"] is not None


def test_single_shard_in_process_and_validation():
    lay, plan = case.build(8)
    rng = np.random.default_rng(0)
    phi, g = rng.standard_normal((8, lay.P)), rng.standard_normal((8, lay.P))   # scalar field, no wall data
    mesh = make_plane_mesh(1)
    sharded = shard_nodal_plan(plan, 1, mesh)
    got = np.asarray(jax.jit(lambda sp, ph, gg: sharded_sbp_bracket(sp, ph, gg, None, case.RHO, mesh))(sharded, phi, g))
    ref = np.asarray(jax.jit(lambda p, ph, gg: sbp_bracket(p, ph, gg, None, case.RHO))(plan, phi, g))
    assert np.abs(got - ref).max() <= 1e-14 * np.abs(ref).max()
    with pytest.raises(ValueError, match="divisible"):
        shard_nodal_plan(plan, 3)
    with pytest.raises(ValueError, match="halo"):
        shard_nodal_plan(plan, 4)
    with pytest.raises(ValueError, match="mesh"):
        shard_nodal_plan(plan, 2, mesh)
    with pytest.raises(ValueError, match="mesh"):
        sharded_sbp_bracket(sharded._replace(n_shards=2), phi, g, None, case.RHO, mesh)
    specs = nodal_plan_specs(plan)
    assert specs.jac == jax.sharding.PartitionSpec("z") and specs.wxy == jax.sharding.PartitionSpec()
    assert specs.blocks[0].Du == jax.sharding.PartitionSpec() and specs.structure == plan.structure


# ---------------------------------------------------------------------------
# Curvature, Laplacian, CG
# ---------------------------------------------------------------------------
def test_halo_constants_of_the_laplacian_and_curvature():
    # features read f at +-2 (D_eta) and -1..+2 (staggered); the transpose collects the cotangents of planes k +- 2 -> f at +-4
    assert (F_HALO, PLAN_HALO, CURVATURE_HALO) == (4, 2, 3)


def test_curvature_sharded_matches_single_device_in_subprocess():
    res = _subprocess("curvature")
    assert res["P"] == 254
    for name in ("default", "pi", "jump_core"):
        assert res[f"{name}_ref_max"] > 0
        for s in (1, 2, 4):
            assert res[f"{name}_S{s}"] <= 1e-15, (name, s, res)     # jitted: XLA fusion may differ by one ulp
    assert res["eager_S4_bitwise"], res                            # op by op the sharded curvature is bitwise the single-device one


def test_laplacian_form_and_action_sharded_match_single_device_in_subprocess():
    res = _subprocess("laplacian")
    assert res["P"] == 254
    for name in ("dirichlet", "conormal", "physical", "nodata"):
        assert res[f"{name}_ref_max"] > 100
        for s in (1, 2, 4):
            assert res[f"{name}_form_S{s}"] <= 1e-14, (name, s, res)
            assert res[f"{name}_action_S{s}"] <= 1e-14, (name, s, res)


def test_sharded_cg_matches_solve_dirichlet_in_subprocess():
    res = _subprocess("cg")
    for tag in ("cold", "warm"):
        assert res[f"{tag}_ref_converged"] and 5 <= res[f"{tag}_ref_iterations"] < 100
        for s in (1, 2, 4):
            assert res[f"{tag}_S{s}_converged"], res
            assert res[f"{tag}_S{s}_iterations"] == res[f"{tag}_ref_iterations"], res
            assert res[f"{tag}_S{s}"] <= 1e-12, res


def test_laplacian_needs_four_planes_per_shard():
    res = _subprocess("laplacian_short")
    for key in ("E8_S4", "E12_S4", "E16_S8"):
        assert res[key] is not None and "halo 4" in res[key], res        # p = 2, 3 and 2: below F_HALO (3 would pass the bracket)


@pytest.fixture(scope="module")
def lap8():
    return case.laplacian_case(8)


def test_laplacian_ext_single_shard_in_process_and_validation(lap8):
    lay, lp = lap8
    rng = np.random.default_rng(3)
    f = rng.standard_normal((8, lay.P, 2))
    g = rng.standard_normal((8, lp.structure.N, 2))
    bcd = LaplacianBoundaryData(value=(g,))
    ref = np.asarray(laplacian_form(lp, f, bcd, "dirichlet"))
    mesh = make_plane_mesh(1)
    sharded = shard_laplacian_plan(lp, 1, mesh)
    got = np.asarray(sharded_laplacian_form(sharded, f, bcd, mesh))
    assert np.abs(got - ref).max() <= 1e-14 * np.abs(ref).max()
    ref_a = np.asarray(laplacian_action(lp, f, bcd, "dirichlet"))
    got_a = np.asarray(sharded_laplacian_action(sharded, f, bcd, mesh))
    assert np.abs(got_a - ref_a).max() <= 1e-14 * np.abs(ref_a).max()
    # the ext form on a periodically extended block, with no mesh
    lp_ext = extend_laplacian_plan(lp)
    f_ext = jnp.concatenate([f[-F_HALO:], f, f[:F_HALO]])
    g_ext = jnp.concatenate([g[-PLAN_HALO:], g, g[:PLAN_HALO]])
    ext = np.asarray(laplacian_action_ext(lp_ext, f_ext, LaplacianBoundaryData(value=(g_ext,))))
    assert np.abs(ext - ref_a).max() <= 1e-14 * np.abs(ref_a).max()
    with pytest.raises(ValueError, match="planes"):
        laplacian_form_ext(lp_ext, f, None)                 # not extended
    with pytest.raises(ValueError, match="coeff=None"):
        sharded_laplacian_form(sharded, f, bcd, mesh, coeff=np.ones((8, lay.P)))
    with pytest.raises(ValueError, match="coeff=None"):
        sharded_laplacian_action(sharded, f, bcd, mesh, coeff=np.ones((8, lay.P)))
    with pytest.raises(ValueError, match="coeff=None"):
        sharded_solve_dirichlet(sharded, f[..., 0], bcd, None, mesh, coeff=np.ones((8, lay.P)))
    with pytest.raises(ValueError, match="divisible"):
        shard_laplacian_plan(lp, 3)
    with pytest.raises(ValueError, match="halo 4"):
        shard_laplacian_plan(lp, 4)                          # p = 2
    with pytest.raises(ValueError, match="mesh"):
        shard_laplacian_plan(lp, 2, mesh)
    with pytest.raises(ValueError, match="mesh"):
        sharded_laplacian_form(sharded._replace(n_shards=2), f, bcd, mesh)
    specs = laplacian_plan_specs(sharded.plan)
    P = jax.sharding.PartitionSpec
    assert specs.A == P("z") and specs.kappa == P("z") and specs.Hp == P("z") and specs.wall_alpha == P("z")
    assert specs.Du == P() and specs.tau == P() and specs.wxy == P() and specs.structure == lp.structure
    assert sharded.plan.A.shape[0] == 8 + 2 * PLAN_HALO


def test_preconditioner_applies_to_a_plane_shard_of_its_factors(lap8):
    lay, lp = lap8
    prec = build_dirichlet_preconditioner(lp, rings_per_block=2, group_planes=4)
    r = np.random.default_rng(5).standard_normal(8 * lay.P)
    full = np.asarray(apply_core_schur_preconditioner(prec, r))
    k0, k1 = 2, 6
    local = dataclasses_replace_leaves(prec, k0, k1)
    got = np.asarray(apply_core_schur_preconditioner(local, r[k0 * lay.P:k1 * lay.P]))
    assert got.shape == ((k1 - k0) * lay.P,)
    assert np.abs(got - full[k0 * lay.P:k1 * lay.P]).max() <= 1e-14 * np.abs(full).max()
    with pytest.raises(TypeError):
        shard_core_schur_preconditioner(object(), 1)
    assert shard_core_schur_preconditioner(prec, 1) is prec


def dataclasses_replace_leaves(prec, k0, k1):
    """The factors of planes ``k0 .. k1`` (``meta`` still describes all planes)."""
    return type(prec)(prec.lo[:, k0:k1], prec.dinv[:, k0:k1], prec.z[k0:k1], prec.sinv[k0:k1], prec.meta)
