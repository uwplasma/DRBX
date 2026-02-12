from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jaxdrb.fci.builder import EssosToroidalFCIConfig, build_fci_maps_essos_toroidal_planes
from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
from jaxdrb.nonlinear.neutrals import NeutralParams


def _essos_root() -> Path:
    essos = pytest.importorskip("essos")
    roots = list(getattr(essos, "__path__", []))
    if not roots:
        pytest.skip("Could not locate ESSOS package path.")
    p = Path(roots[0]).resolve()
    if (p / "examples" / "input_files").exists():
        return p
    if (p.parent / "examples" / "input_files").exists():
        return p.parent
    if (Path("/Users/rogerio/local/ESSOS/examples/input_files")).exists():
        return Path("/Users/rogerio/local/ESSOS")
    pytest.skip("Could not locate ESSOS repo root containing examples/input_files.")


def test_fci_drb3d_full_essos_biotsavart_multiphysics_smoke() -> None:
    pytest.importorskip("essos")
    from essos.coils import Coils_from_json
    from essos.fields import BiotSavart

    root = _essos_root()
    coils_file = root / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
    if not coils_file.exists():
        pytest.skip("ESSOS Biot-Savart coils file not found.")

    field = BiotSavart(Coils_from_json(str(coils_file)))
    cfg = EssosToroidalFCIConfig(
        R0=1.18,
        Z0=-0.08,
        dR=0.03,
        dZ=0.04,
        nR=6,
        nZ=6,
        phi0=0.0,
        dphi=0.1,
        nphi=7,
        open_field_line=True,
        cell_centered=True,
        R_min=1.18,
        R_max=1.33,
        Z_min=-0.12,
        Z_max=0.12,
    )
    map_fwd, map_bwd, _ = build_fci_maps_essos_toroidal_planes(
        cfg, field=field, nsub=6, dl_min=1e-2
    )
    l = cfg.phi0 + cfg.dphi * jnp.arange(cfg.nphi)
    grid = FCISlabGrid.from_maps(
        x0=cfg.R0,
        y0=cfg.Z0,
        dx=cfg.dR,
        dy=cfg.dZ,
        nx=cfg.nR,
        ny=cfg.nZ,
        l=l,
        map_fwd=map_fwd,
        map_bwd=map_bwd,
        open_field_line=True,
        cell_centered=True,
    )
    assert int(jnp.count_nonzero(grid.sheath_mask)) > 0

    shape = (grid.nz, grid.nx, grid.ny)
    key = jax.random.key(404)
    k = jax.random.split(key, 8)
    amp = 3e-4
    y0 = FCIDRB3DFullState(
        n=amp * jax.random.normal(k[0], shape),
        omega=amp * jax.random.normal(k[1], shape),
        vpar_e=amp * jax.random.normal(k[2], shape),
        vpar_i=amp * jax.random.normal(k[3], shape),
        Te=0.16 + amp * jax.random.normal(k[4], shape),
        Ti=0.15 + amp * jax.random.normal(k[5], shape),
        psi=amp * jax.random.normal(k[6], shape),
        N=0.2 + amp * jax.random.normal(k[7], shape),
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.02,
        Dn=5e-4,
        DOmega=6e-4,
        Dvpar=5e-4,
        DTe=5e-4,
        DTi=5e-4,
        Dpsi=4e-4,
        chi_par=7e-4,
        hot_ion_on=True,
        tau_i=0.6,
        em_on=True,
        beta=0.04,
        neutrals_on=True,
        neutrals=NeutralParams(
            enabled=True,
            Dn0=3e-4,
            nu_ion=6e-3,
            nu_rec=4e-3,
            n_background=1.0,
        ),
        sheath_on=True,
        sheath_bc_model="simple",
        sheath_nu_mom=0.4,
        sheath_nu_particle=0.12,
        sheath_nu_energy=0.08,
        sheath_gamma_e=3.2,
        sheath_gamma_i=3.0,
        perp_operator="fd",
        bracket="arakawa",
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    dy0 = model.rhs(0.0, y0)
    assert bool(jnp.isfinite(dy0.n).all())
    assert bool(jnp.isfinite(dy0.Te).all())
    assert bool(jnp.isfinite(dy0.psi).all())
    assert bool(jnp.isfinite(dy0.N).all())

    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.002,
        nsteps=20,
        save_every=5,
        solver="dopri5",
    )
    assert bool(jnp.isfinite(ys.n).all())
    assert bool(jnp.isfinite(ys.omega).all())
    assert bool(jnp.isfinite(ys.Te).all())

    y_last = FCIDRB3DFullState(
        n=ys.n[-1],
        omega=ys.omega[-1],
        vpar_e=ys.vpar_e[-1],
        vpar_i=ys.vpar_i[-1],
        Te=ys.Te[-1],
        Ti=None if ys.Ti is None else ys.Ti[-1],
        psi=None if ys.psi is None else ys.psi[-1],
        N=None if ys.N is None else ys.N[-1],
    )
    pb = model.particle_budget_terms(y_last)
    eb = model.energy_budget_terms(y_last)

    pb_vals = jnp.stack(
        [pb[k] for k in ("total", "advective", "parallel", "neutral", "sheath", "other")]
    )
    eb_vals = jnp.stack(
        [
            eb[k]
            for k in (
                "total",
                "conservative",
                "source",
                "dissipative_other",
                "sheath",
                "residual",
            )
        ]
    )
    assert bool(jnp.isfinite(pb_vals).all())
    assert bool(jnp.isfinite(eb_vals).all())

    # Budget decomposition must close to near-roundoff.
    assert float(jnp.abs(pb["other"])) < 1e-10
    assert float(jnp.abs(eb["residual"])) < 1e-7

    # Open-field-line target channel must be active for this setup.
    assert float(jnp.abs(pb["parallel"])) > 1e-6
