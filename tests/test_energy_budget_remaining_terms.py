from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.bcs import LineBCs
from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter, FieldAlignedGrid
from jaxdrb.core.geometry_line import LineGeometryAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.region_bcs import RegionBC, RegionBCField
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.system import DRBSystem
from jaxdrb.geometry.base import Geometry
from jaxdrb.geometry.plane import Grid2D


def _make_state(shape: tuple[int, ...], *, hot_ion: bool = False) -> DRBSystemState:
    rng = np.random.default_rng(4)
    n = jnp.asarray(rng.normal(size=shape))
    omega = jnp.asarray(rng.normal(size=shape))
    Te = jnp.asarray(rng.normal(size=shape))
    Ti = jnp.asarray(rng.normal(size=shape)) if hot_ion else None
    z = jnp.zeros_like(n)
    return DRBSystemState(
        n=n,
        omega=omega,
        vpar_e=z,
        vpar_i=z,
        Te=Te,
        Ti=Ti,
        psi=None,
        N=None,
    )


def test_energy_budget_diffusion_extra_and_bc_terms() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"kpar": 1.0},
            "closure": {"sol": {"sol_on": False}, "sheath": {"sheath_on": False}},
            "transport": {
                "Dn": 0.1,
                "DTe": 0.2,
                "DOmega": 0.05,
                "phi_dissipation_on": True,
                "phi_par_dissipation": 0.05,
            },
            "numerics": {
                "poisson": "spectral",
                "term_schedule": [
                    "diffusion",
                    "extra_dissipation",
                    "field_bc_relax",
                    "perp_bc_relax",
                ],
            },
            "bcs": {"bc_enforce_nu": 1.0, "perp_bc_nu": 1.0},
        },
    )
    grid = Grid2D.make(
        nx=8,
        ny=8,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        dealias=False,
        bc_x="dirichlet",
        bc_y="dirichlet",
    )
    geom = Geometry2DAdapter(grid=grid, params=params)
    system = DRBSystem(params=params, geom=geom)

    y = _make_state((grid.nx, grid.ny))
    budget = system.energy_budget(y)
    for key in ("diffusion", "extra_dissipation", "field_bc_relax", "perp_bc_relax"):
        assert f"E_dot_{key}" in budget


def test_energy_budget_sol_terms() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "closure": {
                "sol": {
                    "sol_on": True,
                    "sol_xs": 0.3,
                    "sol_width": 0.1,
                    "sol_parallel_loss_on": True,
                    "sol_parallel_loss_coeff": 0.5,
                    "sol_parallel_loss_q": 2.0,
                    "sol_sheath_phi_on": True,
                    "sol_sheath_phi_dissipation_on": True,
                    "sol_sheath_omega_on": True,
                    "sol_omega_bc_dirichlet_on": True,
                    "sol_vpar_bc_dirichlet_on": True,
                    "sol_edge_relax_on": True,
                }
            },
            "numerics": {
                "poisson": "spectral",
                "term_schedule": [
                    "sol_sources",
                    "sol_sinks",
                    "sol_parallel_loss",
                    "sol_sheath_phi",
                    "sol_sheath_omega",
                    "sol_omega_bc",
                    "sol_vpar_bc",
                    "sol_edge_relax",
                ],
            },
        },
    )
    grid = Grid2D.make(
        nx=8,
        ny=8,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        dealias=False,
        bc_x="periodic",
        bc_y="periodic",
    )
    geom = Geometry2DAdapter(grid=grid, params=params)
    system = DRBSystem(params=params, geom=geom)

    y = _make_state((grid.nx, grid.ny))
    budget = system.energy_budget(y)
    for key in (
        "sol_sources",
        "sol_sinks",
        "sol_parallel_loss",
        "sol_sheath_phi",
        "sol_sheath_omega",
        "sol_omega_bc",
        "sol_vpar_bc",
        "sol_edge_relax",
    ):
        assert f"E_dot_{key}" in budget


def test_energy_budget_region_bc_relax() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"kpar": 0.0},
            "numerics": {
                "term_schedule": ["region_bc_relax"],
                "poisson": "spectral",
            },
            "region_bc_on": True,
        },
    )
    mask = np.ones((4, 4), dtype=np.float64)
    region_masks = {"core": mask}
    region_bcs = (
        RegionBC(
            name="core",
            mask=jnp.asarray(mask),
            n=RegionBCField(kind="dirichlet", value=1.0, nu=1.0),
            Te=RegionBCField(kind="dirichlet", value=1.0, nu=1.0),
            omega=RegionBCField(kind="dirichlet", value=0.0, nu=1.0),
            vpar_e=RegionBCField(kind="dirichlet", value=0.0, nu=1.0),
            vpar_i=RegionBCField(kind="dirichlet", value=0.0, nu=1.0),
            Ti=RegionBCField(kind="dirichlet", value=1.0, nu=1.0),
            psi=RegionBCField(kind="dirichlet", value=0.0, nu=1.0),
        ),
    )
    grid = FieldAlignedGrid.make(
        nx=4,
        ny=4,
        nz=4,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        bc_x="periodic",
        bc_y="periodic",
        dealias=False,
        open_field_line=True,
        region_masks=region_masks,
        region_bcs=region_bcs,
    )
    geom = FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=jnp.zeros((4, 4)),
        curv_y=jnp.zeros((4, 4)),
        dpar_factor=jnp.ones((4,)),
        B=jnp.ones((4, 4)),
    )
    system = DRBSystem(params=params, geom=geom)
    y = _make_state((4, 4, 4), hot_ion=True)
    budget = system.energy_budget(y)
    assert "E_dot_region_bc_relax" in budget


def test_energy_budget_sheath_term() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "closure": {"sheath": {"sheath_on": True}},
            "numerics": {"term_schedule": ["sheath"], "poisson": "spectral"},
        },
    )
    grid = FieldAlignedGrid.make(
        nx=4,
        ny=4,
        nz=6,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        bc_x="periodic",
        bc_y="periodic",
        dealias=False,
        open_field_line=True,
    )
    geom = FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=jnp.zeros((4, 4)),
        curv_y=jnp.zeros((4, 4)),
        dpar_factor=jnp.ones((6,)),
        B=jnp.ones((4, 4)),
    )
    system = DRBSystem(params=params, geom=geom)
    y = _make_state((6, 4, 4), hot_ion=True)
    budget = system.energy_budget(y)
    assert "E_dot_sheath" in budget


class _LineGeom(Geometry):
    def __init__(self, nl: int, dl: float) -> None:
        self.l = jnp.linspace(0.0, dl * (nl - 1), nl)
        self.dl = float(dl)

    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:
        return jnp.full_like(self.l, kx**2 + ky**2)

    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        dl = float(self.dl)
        df = jnp.zeros_like(f)
        df = df.at[1:-1].set((f[2:] - f[:-2]) / (2.0 * dl))
        df = df.at[0].set((f[1] - f[0]) / dl)
        df = df.at[-1].set((f[-1] - f[-2]) / dl)
        return df

    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(f)


def test_energy_budget_line_bcs_term() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "numerics": {"term_schedule": ["line_bcs"]},
            "line_bcs": LineBCs.all_dirichlet(value=0.0, nu=1.0),
        },
    )
    geom = LineGeometryAdapter(geom=_LineGeom(nl=8, dl=0.1), params=params, kx=0.0, ky=0.0)
    system = DRBSystem(params=params, geom=geom)
    y = _make_state((8,), hot_ion=True)
    budget = system.energy_budget(y)
    assert "E_dot_line_bcs" in budget
