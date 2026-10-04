"""Eta-sharded nodal SBP bracket: multi-device agreement (subprocess with forced host devices) and plan checks."""
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

from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket
from drbx.native.fci_perpendicular_sbp_sharding import (
    NODAL_HALO,
    make_plane_mesh,
    nodal_plan_specs,
    shard_nodal_plan,
    sharded_sbp_bracket,
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
