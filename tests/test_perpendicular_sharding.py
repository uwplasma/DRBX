"""eta-sharded perpendicular layer (P08 step 6, stage A): relabelling, halo exchange, plan sharding (fast).

* the plane-major relabelling and its error cases;
* ``exchange_plane_halo`` against a NumPy periodic gather for 1, 2 and 4 shards and halos 1, 2 (multi-device cases run
  in a subprocess with forced host devices: the device count must be set before JAX is imported);
* source-row / tensor-row selection and localization of one shard against the global apply (toy tensor grid);
* a synthetic plane-structured plan (random rows of every kind, donors within two planes of their owners): the combined
  RHS sharded over 1, 2 and 4 shards equals the single-device RHS, and the plan coverage is the expected one.
The real N32 closure check is in ``tests/test_perpendicular_sharding_real.py`` (slow).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.native.fci_perpendicular_sharding import (
    DEFAULT_HALO, _Ctx, _Shard, _select_sources, exchange_plane_halo, from_plane_major, plane_major_permutation,
    PlaneLayout, to_plane_major)
from drbx.native.fci_perpendicular_source_rows import apply_source_rows
from drbx.stencils.loader import LoaderGrid, lower_point_chunks
from tests import perpendicular_sharding_case as case
from tests.test_stencils_tensor_loader import _boundary, _factored
from tests.test_stencils_tensor_rows import IDENTITY_PLAN, _context, _sources

CASE = Path(case.__file__).resolve()


def _subprocess(*args, devices: int = 4, timeout: int = 600) -> dict:
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={devices}".strip()
    env["JAX_PLATFORMS"] = "cpu"
    done = subprocess.run([sys.executable, str(CASE), *args], capture_output=True, text=True, env=env,
                          timeout=timeout, check=False)
    assert done.returncode == 0, f"subprocess failed ({done.returncode}):\n{done.stderr[-4000:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# Relabelling
# --------------------------------------------------------------------------

def test_plane_major_permutation_round_trip():
    n = 8
    raw_to_owner = case.plane_structured_owners(n)
    perm, inverse, m = plane_major_permutation(raw_to_owner, n)
    n_owners = int(raw_to_owner.max()) + 1
    assert m == n_owners // n == 4 * n
    assert np.array_equal(inverse[perm], np.arange(n_owners)) and np.array_equal(perm[inverse], np.arange(n_owners))
    plane = np.zeros(n_owners, dtype=int)
    plane[raw_to_owner] = np.arange(n ** 3) % n
    assert np.array_equal(perm // m, plane)                                  # new id = plane * m + rank
    for k in range(n):                                                       # ranks follow the old id order
        old = np.flatnonzero(plane == k)
        assert np.array_equal(perm[old], k * m + np.arange(m))
    assert not np.array_equal(inverse, np.arange(n_owners))                  # the old numbering is not plane-contiguous
    values = np.random.default_rng(0).normal(size=(n_owners, 3))
    pm = to_plane_major(values, inverse)
    assert np.array_equal(pm.reshape(n, m, 3)[:, 2], values[inverse.reshape(n, m)[:, 2]])
    assert np.array_equal(from_plane_major(pm, perm), values)
    assert np.array_equal(to_plane_major(jnp.asarray(values), inverse), pm)


def test_plane_major_permutation_rejects_unequal_planes_and_multi_plane_owners():
    n = 8
    raw_to_owner = case.plane_structured_owners(n).copy()
    raw = np.arange(n ** 3)
    merge = (raw // n ** 2 == 3) & ((raw // n) % n < 4) & (raw % n == 0)             # four owners of one plane become one
    raw_to_owner[merge] = raw_to_owner[merge][0]
    _, relabelled = np.unique(raw_to_owner, return_inverse=True)
    with pytest.raises(ValueError, match="equally many owners"):
        plane_major_permutation(relabelled, n)
    with pytest.raises(ValueError, match="eta plane"):                                # owners spanning two planes
        plane_major_permutation(np.arange(8 ** 3) // 2, 8)


# --------------------------------------------------------------------------
# Halo exchange
# --------------------------------------------------------------------------

@pytest.mark.parametrize("halo", (1, 2, 3))
def test_exchange_plane_halo_single_shard_is_a_local_periodic_wrap(halo):
    owned = np.random.default_rng(1).normal(size=(5, 3, 2))
    got = np.asarray(exchange_plane_halo(jnp.asarray(owned), halo, "z", 1))
    assert got.shape == (5 + 2 * halo, 3, 2)
    assert np.array_equal(got, owned[(np.arange(-halo, 5 + halo)) % 5])


def test_exchange_plane_halo_matches_numpy_periodic_gather_in_subprocess():
    result = _subprocess("halo")
    assert result["devices"] == 4
    expected = {f"Sz{sz}_h{h}" for sz in (1, 2, 4) for h in (1, 2)}
    assert expected <= set(result)
    assert all(result[key] == 0.0 for key in expected), result


# --------------------------------------------------------------------------
# Source rows (CSR and tensor-encoded) of one shard against the global apply
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy():
    context = _context()
    _, _, factors, chunk = _sources(context, IDENTITY_PLAN)
    item, _dense = _factored(chunk, factors)
    plan = lower_point_chunks([item], grid=LoaderGrid.from_context(context))
    n = context.n
    perm, inverse, m = plane_major_permutation(context.ro, n)
    return dict(context=context, plan=plan, layout=PlaneLayout(perm, inverse, m), n=n,
                fields=np.random.default_rng(3).normal(size=(len(context.vol), 3)))


@pytest.mark.parametrize("n_shards", (2,))
def test_source_row_selection_reproduces_the_global_apply_on_the_needed_slots(toy, n_shards):
    plan, payload, n = toy["plan"], toy["plan"].payload, toy["n"]
    assert payload.tensor_batches and payload.batches                       # both encodings are exercised
    ctx = _Ctx(toy["layout"], n, n_shards, DEFAULT_HALO)
    boundary = _boundary(plan.boundary_points, 3)
    ref_v, ref_g = (np.asarray(x) for x in apply_source_rows(payload, toy["fields"], boundary))
    rng = np.random.default_rng(5)
    need_v = rng.random(payload.n_targets) < 0.4
    need_g = rng.random(payload.n_gradient_targets) < 0.4
    for s in range(n_shards):
        sh = _Shard(ctx, s)
        local, vmap, gmap = _select_sources(payload, need_v, need_g, sh, "toy")
        assert local.n_targets <= payload.n_targets and local.n_targets >= need_v.sum()
        fields = np.concatenate([toy["fields"][sh.ext_old], np.zeros((1, 3))])           # local order plus the trash row
        v, g = (np.asarray(x) for x in apply_source_rows(local, fields, boundary))
        slots = np.flatnonzero(need_v)
        assert (vmap[slots] >= 0).all()
        scale = max(np.abs(ref_v).max(), 1.0)
        assert np.abs(v[vmap[slots]] - ref_v[slots]).max() <= 1e-13 * scale
        gslots = np.flatnonzero(need_g)
        assert (gmap[gslots] >= 0).all()
        assert np.abs(g[gmap[gslots]] - ref_g[gslots]).max() <= 1e-13 * max(np.abs(ref_g).max(), 1.0)


def test_source_row_selection_reports_donors_outside_the_window(toy):
    payload, n = toy["plan"].payload, toy["n"]
    ctx = _Ctx(toy["layout"], n, 4, 1)                                       # 2 planes owned, 1 halo plane each side
    sh = _Shard(ctx, 0)
    everything = (np.ones(payload.n_targets, dtype=bool), np.ones(payload.n_gradient_targets, dtype=bool))
    with pytest.raises(ValueError, match="outside the extended window|outside the extended"):
        _select_sources(payload, *everything, sh, "toy")


# --------------------------------------------------------------------------
# Synthetic plane-structured plan, sharded RHS against single device
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic():
    return _subprocess("synthetic")


def test_synthetic_sharded_plan_covers_every_entity(synthetic):
    assert synthetic["devices"] == 4
    for sz, cov in synthetic["coverage"].items():
        assert cov["cells_kept"] == cov["cells"]                             # every cell on exactly one shard
        assert cov["faces_min_count"] == 1 and cov["faces_max_count"] == (1 if sz == "1" else 2)
        assert cov["faces_on_two_shards"] == cov["faces_across_blocks"]      # exactly the block-boundary faces
        assert cov["p07_unique"] == cov["p07"] and cov["p07_kept"] >= cov["p07"]
    assert synthetic["coverage"]["4"]["faces_across_blocks"] == 2 * synthetic["coverage"]["2"]["faces_across_blocks"]


def test_synthetic_plan_needs_a_halo_of_three_planes(synthetic):
    message = synthetic["halo2_error"]
    assert message is not None and "weighted donors outside the extended window of halo 2" in message
    assert "[-3, 3]" in message                                              # the offending eta offsets


@pytest.mark.parametrize("kinds", ("dirichlet", "mixed"))
def test_synthetic_sharded_rhs_equals_single_device(synthetic, kinds):
    entry = synthetic[kinds]
    # random wall states make P06 non-finite on the wall ring (``nan_owners``); the NaN pattern is part of the comparison
    for sz in (1, 2, 4):
        assert entry[f"Sz{sz}"]["max_rel"] <= 1e-12, (kinds, sz, entry[f"Sz{sz}"])
    # the per-shard counters of the duplicated block-boundary faces bound the single-device ones from above
    single = entry["single_diagnostics"]
    for name in ("floor_hits", "spectral_fallback"):
        assert sum(entry["Sz1"]["diagnostics"][name]) == single[name]
        assert sum(entry["Sz4"]["diagnostics"][name]) >= single[name]
