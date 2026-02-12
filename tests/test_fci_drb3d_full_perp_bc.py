from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxdrb.bc import BC2D
from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def _random_state(
    key: jax.Array, shape: tuple[int, int, int], amp: float = 1e-3
) -> FCIDRB3DFullState:
    k = jax.random.split(key, 5)
    return FCIDRB3DFullState(
        n=amp * jax.random.normal(k[0], shape),
        omega=amp * jax.random.normal(k[1], shape),
        vpar_e=amp * jax.random.normal(k[2], shape),
        vpar_i=amp * jax.random.normal(k[3], shape),
        Te=amp * jax.random.normal(k[4], shape),
    )


def _boundary_rms(f: jnp.ndarray) -> float:
    bd = jnp.concatenate(
        [
            f[:, 0, :].reshape(-1),
            f[:, -1, :].reshape(-1),
            f[:, :, 0].reshape(-1),
            f[:, :, -1].reshape(-1),
        ]
    )
    return float(jnp.sqrt(jnp.mean(bd**2)))


@pytest.mark.parametrize("perp_operator", ["fd", "fv"])
def test_fci_drb3d_full_dirichlet_wall_relaxation_damps_boundaries(perp_operator: str) -> None:
    grid = FCISlabGrid.make(
        nx=16,
        ny=16,
        nz=6,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.2,
        Bz=1.0,
        open_field_line=False,
    )
    y0 = _random_state(jax.random.key(11), (grid.nz, grid.nx, grid.ny), amp=1e-2)
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.05,
        Dn=3e-3,
        DOmega=3e-3,
        Dvpar=3e-3,
        DTe=3e-3,
        chi_par=3e-3,
        sheath_on=False,
        perp_operator=perp_operator,  # type: ignore[arg-type]
        perp_bc=BC2D.dirichlet(x=0.0, y=0.0),
        perp_bc_nu=4.0,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    ys, y_end = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.02,
        nsteps=200,
        save_every=20,
        solver="dopri5",
    )
    assert bool(jnp.isfinite(ys.n).all())
    assert _boundary_rms(y_end.n) < 0.08 * _boundary_rms(y0.n)


@pytest.mark.parametrize("perp_operator", ["fd", "fv"])
def test_fci_drb3d_full_neumann_wall_rhs_is_finite(perp_operator: str) -> None:
    grid = FCISlabGrid.make(
        nx=14,
        ny=12,
        nz=6,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.2,
        Bz=1.0,
        open_field_line=False,
    )
    y0 = _random_state(jax.random.key(22), (grid.nz, grid.nx, grid.ny))
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.05,
        Dn=2e-3,
        DOmega=2e-3,
        Dvpar=2e-3,
        DTe=2e-3,
        chi_par=2e-3,
        sheath_on=False,
        perp_operator=perp_operator,  # type: ignore[arg-type]
        perp_bc=BC2D.neumann(x=0.0, y=0.0),
        perp_bc_nu=2.0,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    dy = model.rhs(0.0, y0)
    assert bool(jnp.isfinite(dy.n).all())
    assert bool(jnp.isfinite(dy.omega).all())
    assert bool(jnp.isfinite(dy.vpar_e).all())
    assert bool(jnp.isfinite(dy.vpar_i).all())
    assert bool(jnp.isfinite(dy.Te).all())
