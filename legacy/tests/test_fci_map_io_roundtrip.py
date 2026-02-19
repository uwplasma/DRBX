from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC1D
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.fci.io import load_fci_maps_npz, save_fci_maps_npz
from jaxdrb.fci.parallel import parallel_derivative_target_aware_3d


def test_fci_map_npz_roundtrip(tmp_path) -> None:
    grid = FCISlabGrid.make(
        nx=24,
        ny=20,
        nz=10,
        Lx=2 * math.pi,
        Ly=2 * math.pi,
        Lz=3.0,
        Bx=0.1,
        By=-0.05,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    out = tmp_path / "fci_maps.npz"
    save_fci_maps_npz(
        out,
        map_fwd=grid.map_fwd,
        map_bwd=grid.map_bwd,
        meta={"note": "roundtrip test"},
    )
    map_fwd, map_bwd, meta = load_fci_maps_npz(out)
    assert meta["note"] == "roundtrip test"

    key = jax.random.key(0)
    f = jax.random.normal(key, (grid.nz, grid.nx, grid.ny))

    bc = BC1D.dirichlet(left=0.0, right=0.0, nu=0.0)
    dpar0 = parallel_derivative_target_aware_3d(
        f,
        map_fwd=grid.map_fwd,
        map_bwd=grid.map_bwd,
        open_field_line=True,
        bc=bc,
    )
    dpar1 = parallel_derivative_target_aware_3d(
        f,
        map_fwd=map_fwd,
        map_bwd=map_bwd,
        open_field_line=True,
        bc=bc,
    )
    assert float(jnp.max(jnp.abs(dpar0 - dpar1))) < 1e-12
