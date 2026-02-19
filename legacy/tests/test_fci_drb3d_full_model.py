from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
from jaxdrb.nonlinear.neutrals import NeutralParams


def _rand_state(key, shape) -> FCIDRB3DFullState:
    k = jax.random.split(key, 5)
    return FCIDRB3DFullState(
        n=jax.random.normal(k[0], shape),
        omega=jax.random.normal(k[1], shape),
        vpar_e=jax.random.normal(k[2], shape),
        vpar_i=jax.random.normal(k[3], shape),
        Te=jax.random.normal(k[4], shape),
    )


def _state_l2_diff(a: FCIDRB3DFullState, b: FCIDRB3DFullState) -> float:
    err = (
        (a.n - b.n) ** 2
        + (a.omega - b.omega) ** 2
        + (a.vpar_e - b.vpar_e) ** 2
        + (a.vpar_i - b.vpar_i) ** 2
        + (a.Te - b.Te) ** 2
    )
    if a.Ti is not None and b.Ti is not None:
        err = err + (a.Ti - b.Ti) ** 2
    if a.psi is not None and b.psi is not None:
        err = err + (a.psi - b.psi) ** 2
    if a.N is not None and b.N is not None:
        err = err + (a.N - b.N) ** 2
    return float(jnp.sqrt(jnp.mean(err)))


def test_fci_drb3d_full_conservative_gate_periodic() -> None:
    grid = FCISlabGrid.make(
        nx=18,
        ny=16,
        nz=8,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=False,
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.0,
        Dn=0.0,
        DOmega=0.0,
        Dvpar=0.0,
        DTe=0.0,
        chi_par=0.0,
        sheath_on=False,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    key = jax.random.key(7)
    base = _rand_state(key, (1, grid.nx, grid.ny))
    y0 = FCIDRB3DFullState(
        n=jnp.broadcast_to(base.n, (grid.nz, grid.nx, grid.ny)),
        omega=jnp.broadcast_to(base.omega, (grid.nz, grid.nx, grid.ny)),
        vpar_e=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        vpar_i=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        Te=jnp.zeros((grid.nz, grid.nx, grid.ny)),
    )

    dy = model.rhs(0.0, y0)
    erate = float(model.energy_rate(y0, dy))
    prate = float(model.particle_rate(dy))
    crate = float(jnp.mean(dy.vpar_i - dy.vpar_e))

    assert abs(prate) < 1e-11
    assert abs(crate) < 1e-11
    assert abs(erate) < 5e-9


def test_fci_drb3d_full_sheath_budget_consistency() -> None:
    grid = FCISlabGrid.make(
        nx=14,
        ny=12,
        nz=16,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=6.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.0,
        Dn=0.0,
        DOmega=0.0,
        Dvpar=0.0,
        DTe=0.0,
        chi_par=0.0,
        sheath_on=True,
        sheath_nu_mom=1.4,
        sheath_nu_particle=0.8,
        sheath_nu_energy=0.6,
        sheath_gamma_e=3.0,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    key = jax.random.key(13)
    base = _rand_state(key, (1, grid.nx, grid.ny))
    y0 = FCIDRB3DFullState(
        n=jnp.broadcast_to(base.n, (grid.nz, grid.nx, grid.ny)),
        omega=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        vpar_e=jnp.broadcast_to(base.vpar_e, (grid.nz, grid.nx, grid.ny)),
        vpar_i=jnp.broadcast_to(base.vpar_i, (grid.nz, grid.nx, grid.ny)),
        Te=jnp.abs(jnp.broadcast_to(base.Te, (grid.nz, grid.nx, grid.ny))) + 0.2,
    )

    dy = model.rhs(0.0, y0)
    p_rate = float(model.particle_rate(dy))
    e_rate = float(model.energy_rate(y0, dy))
    p_sh, e_sh = model.sheath_budget_rates(y0)

    assert p_rate < 0.0
    assert e_rate < 0.0
    assert abs(p_rate - float(p_sh)) < 1e-10
    assert abs(e_rate - float(e_sh)) < 5e-9


def test_fci_drb3d_full_operator_split_reconstruction() -> None:
    grid = FCISlabGrid.make(
        nx=10,
        ny=10,
        nz=8,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    key = jax.random.key(101)
    y0 = _rand_state(key, (grid.nz, grid.nx, grid.ny))
    params_full = FCIDRB3DFullParams(
        omega_n=0.5,
        omega_Te=0.4,
        kappa=0.6,
        alpha=0.2,
        eta_par=0.1,
        Dn=1e-3,
        DOmega=1e-3,
        Dvpar=1e-3,
        DTe=1e-3,
        chi_par=2e-2,
        sheath_on=True,
        sheath_nu_mom=0.7,
        sheath_nu_particle=0.5,
        sheath_nu_energy=0.4,
        operator_split_on=False,
    )
    model_full = FCIDRB3DFullModel(params=params_full, grid=grid)
    dy_full = model_full.rhs(0.0, y0)

    params_split = eqx.tree_at(lambda p: p.operator_split_on, params_full, True)
    model_split = FCIDRB3DFullModel(params=params_split, grid=grid)
    dy_split = model_split.rhs(0.0, y0)

    assert _state_l2_diff(dy_full, dy_split) < 1e-11


def test_fci_drb3d_full_sheath_budget_consistency_loizu_linear() -> None:
    grid = FCISlabGrid.make(
        nx=14,
        ny=10,
        nz=18,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=6.0,
        Bx=0.0,
        By=0.1,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.0,
        Dn=0.0,
        DOmega=0.0,
        Dvpar=0.0,
        DTe=0.0,
        chi_par=0.0,
        hot_ion_on=True,
        tau_i=0.5,
        DTi=0.0,
        sheath_on=True,
        sheath_bc_model="loizu_linear",
        sheath_nu_mom=1.0,
        sheath_nu_particle=0.7,
        sheath_nu_energy=0.5,
        sheath_gamma_e=3.0,
        sheath_gamma_i=3.5,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    key = jax.random.key(991)
    base_shape = (1, grid.nx, grid.ny)
    k = jax.random.split(key, 6)
    y0 = FCIDRB3DFullState(
        n=jnp.broadcast_to(0.1 * jax.random.normal(k[0], base_shape), (grid.nz, grid.nx, grid.ny)),
        omega=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        vpar_e=jnp.broadcast_to(
            0.05 * jax.random.normal(k[2], base_shape), (grid.nz, grid.nx, grid.ny)
        ),
        vpar_i=jnp.broadcast_to(
            0.05 * jax.random.normal(k[3], base_shape), (grid.nz, grid.nx, grid.ny)
        ),
        Te=jnp.broadcast_to(
            0.2 + 0.05 * jax.random.normal(k[4], base_shape), (grid.nz, grid.nx, grid.ny)
        ),
        Ti=jnp.broadcast_to(
            0.2 + 0.05 * jax.random.normal(k[5], base_shape), (grid.nz, grid.nx, grid.ny)
        ),
    )
    dy = model.rhs(0.0, y0)
    p_rate = float(model.particle_rate(dy))
    e_rate = float(model.energy_rate(y0, dy))
    p_sh, e_sh = model.sheath_budget_rates(y0)
    assert bool(jnp.isfinite(p_sh))
    assert bool(jnp.isfinite(e_sh))
    assert p_rate < 0.0
    assert e_rate < 0.0
    assert abs(p_rate - float(p_sh)) < 5e-9
    assert abs(e_rate - float(e_sh)) < 1e-4


def test_fci_drb3d_full_hot_em_neutrals_split_reconstruction() -> None:
    grid = FCISlabGrid.make(
        nx=10,
        ny=10,
        nz=8,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.15,
        Bz=1.0,
        open_field_line=False,
    )
    shape = (grid.nz, grid.nx, grid.ny)
    key = jax.random.key(717)
    k = jax.random.split(key, 8)
    y0 = FCIDRB3DFullState(
        n=1e-3 * jax.random.normal(k[0], shape),
        omega=1e-3 * jax.random.normal(k[1], shape),
        vpar_e=1e-3 * jax.random.normal(k[2], shape),
        vpar_i=1e-3 * jax.random.normal(k[3], shape),
        Te=1e-3 * jax.random.normal(k[4], shape),
        Ti=1e-3 * jax.random.normal(k[5], shape),
        psi=1e-3 * jax.random.normal(k[6], shape),
        N=1e-3 * jax.random.normal(k[7], shape),
    )
    params = FCIDRB3DFullParams(
        omega_n=0.8,
        omega_Te=0.5,
        omega_Ti=0.4,
        kappa=0.6,
        alpha=0.2,
        eta_par=0.05,
        Dn=1e-3,
        DOmega=1e-3,
        Dvpar=1e-3,
        DTe=1e-3,
        DTi=1e-3,
        Dpsi=1e-3,
        chi_par=2e-2,
        hot_ion_on=True,
        tau_i=0.7,
        em_on=True,
        beta=0.1,
        neutrals_on=True,
        neutrals=NeutralParams(enabled=True, Dn0=1e-3, nu_ion=0.03, nu_rec=0.01),
        sheath_on=False,
        operator_split_on=False,
    )
    dy_full = FCIDRB3DFullModel(params=params, grid=grid).rhs(0.0, y0)
    dy_split = FCIDRB3DFullModel(
        params=eqx.tree_at(lambda p: p.operator_split_on, params, True), grid=grid
    ).rhs(0.0, y0)
    assert _state_l2_diff(dy_full, dy_split) < 1e-11


def test_fci_drb3d_full_neutral_exchange_total_particles_conserved() -> None:
    grid = FCISlabGrid.make(
        nx=12,
        ny=10,
        nz=6,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=False,
    )
    shape = (grid.nz, grid.nx, grid.ny)
    key = jax.random.key(331)
    k = jax.random.split(key, 7)
    y0 = FCIDRB3DFullState(
        n=0.02 * jax.random.normal(k[0], shape),
        omega=jnp.zeros(shape),
        vpar_e=jnp.zeros(shape),
        vpar_i=jnp.zeros(shape),
        Te=jnp.zeros(shape),
        N=0.2 + 0.02 * jax.random.normal(k[1], shape),
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.0,
        Dn=0.0,
        DOmega=0.0,
        Dvpar=0.0,
        DTe=0.0,
        chi_par=0.0,
        neutrals_on=True,
        neutrals=NeutralParams(
            enabled=True,
            Dn0=0.0,
            S0=0.0,
            nu_sink=0.0,
            nu_ion=0.1,
            nu_rec=0.03,
            n_background=1.0,
        ),
        sheath_on=False,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    dy = model.rhs(0.0, y0)
    assert y0.N is not None and dy.N is not None
    total_rate = float(model.total_particle_rate(dy))
    assert abs(total_rate) < 2e-10


def test_fci_drb3d_full_hot_em_conservative_rate_gate_periodic() -> None:
    grid = FCISlabGrid.make(
        nx=12,
        ny=12,
        nz=8,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=False,
    )
    shape2 = (1, grid.nx, grid.ny)
    key = jax.random.key(811)
    k = jax.random.split(key, 7)
    y0 = FCIDRB3DFullState(
        n=jnp.broadcast_to(0.03 * jax.random.normal(k[0], shape2), (grid.nz, grid.nx, grid.ny)),
        omega=jnp.broadcast_to(0.03 * jax.random.normal(k[1], shape2), (grid.nz, grid.nx, grid.ny)),
        vpar_e=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        vpar_i=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        Te=jnp.broadcast_to(0.02 * jax.random.normal(k[2], shape2), (grid.nz, grid.nx, grid.ny)),
        Ti=jnp.broadcast_to(0.02 * jax.random.normal(k[3], shape2), (grid.nz, grid.nx, grid.ny)),
        psi=jnp.broadcast_to(0.02 * jax.random.normal(k[4], shape2), (grid.nz, grid.nx, grid.ny)),
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.0,
        Dn=0.0,
        DOmega=0.0,
        Dvpar=0.0,
        DTe=0.0,
        DTi=0.0,
        Dpsi=0.0,
        chi_par=0.0,
        hot_ion_on=True,
        em_on=True,
        beta=0.2,
        sheath_on=False,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    dy = model.rhs(0.0, y0)
    assert abs(float(model.particle_rate(dy))) < 2e-11
    assert abs(float(model.current_content(dy))) < 2e-10
    assert abs(float(model.energy_rate(y0, dy))) < 5e-7


def test_fci_drb3d_full_em_sheath_current_closure_response() -> None:
    grid = FCISlabGrid.make(
        nx=12,
        ny=10,
        nz=16,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=6.0,
        Bx=0.0,
        By=0.1,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    shape = (grid.nz, grid.nx, grid.ny)
    key = jax.random.key(66)
    k = jax.random.split(key, 6)
    y0 = FCIDRB3DFullState(
        n=0.04 * jax.random.normal(k[0], shape),
        omega=jnp.zeros(shape),
        vpar_e=0.04 * jax.random.normal(k[1], shape),
        vpar_i=0.04 * jax.random.normal(k[2], shape),
        Te=0.2 + 0.04 * jax.random.normal(k[3], shape),
        Ti=0.2 + 0.04 * jax.random.normal(k[4], shape),
        psi=0.04 * jax.random.normal(k[5], shape),
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.0,
        Dn=0.0,
        DOmega=0.0,
        Dvpar=0.0,
        DTe=0.0,
        DTi=0.0,
        Dpsi=0.0,
        chi_par=0.0,
        hot_ion_on=True,
        em_on=True,
        beta=0.1,
        sheath_on=True,
        sheath_bc_model="loizu_linear",
        sheath_nu_mom=0.8,
        sheath_nu_particle=0.0,
        sheath_nu_energy=0.0,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    split = model.rhs_decomposed(0.0, y0)
    assert split.dissipative.psi is not None
    # Sheath-induced parallel-current mismatch must feed back into psi in EM mode.
    assert float(jnp.max(jnp.abs(split.dissipative.psi))) > 1e-12


def test_fci_drb3d_full_plate_sheath_budget_gate_multiphysics() -> None:
    """Finite-time plate/sheath budget gate on the full 3D multiphysics branch."""

    grid = FCISlabGrid.make(
        nx=10,
        ny=10,
        nz=12,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=6.0,
        Bx=0.0,
        By=0.15,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    shape = (grid.nz, grid.nx, grid.ny)
    key = jax.random.key(1203)
    k = jax.random.split(key, 8)
    amp = 5e-4
    y0 = FCIDRB3DFullState(
        n=amp * jax.random.normal(k[0], shape),
        omega=amp * jax.random.normal(k[1], shape),
        vpar_e=amp * jax.random.normal(k[2], shape),
        vpar_i=amp * jax.random.normal(k[3], shape),
        Te=0.18 + amp * jax.random.normal(k[4], shape),
        Ti=0.17 + amp * jax.random.normal(k[5], shape),
        psi=amp * jax.random.normal(k[6], shape),
        N=0.22 + amp * jax.random.normal(k[7], shape),
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.03,
        me_hat=0.3,
        Dn=3e-4,
        DOmega=3e-4,
        Dvpar=3e-4,
        DTe=3e-4,
        DTi=3e-4,
        Dpsi=3e-4,
        chi_par=4e-4,
        hot_ion_on=True,
        tau_i=0.7,
        em_on=True,
        beta=0.06,
        neutrals_on=True,
        neutrals=NeutralParams(
            enabled=True,
            Dn0=2e-4,
            S0=0.0,
            nu_sink=0.0,
            nu_ion=5e-3,
            nu_rec=3e-3,
            n_background=1.0,
            nu_cx_omega=0.0,
        ),
        sheath_on=True,
        sheath_bc_model="loizu_linear",
        sheath_nu_mom=0.5,
        sheath_nu_particle=0.18,
        sheath_nu_energy=0.1,
        sheath_gamma_e=3.2,
        sheath_gamma_i=3.0,
        perp_operator="spectral",
        bracket="arakawa",
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.002,
        nsteps=140,
        save_every=5,
        solver="dopri5",
    )
    ts = 0.002 * jnp.arange(5, 141, 5)

    energy = []
    total_particles = []
    energy_rates = []
    particle_rates = []
    sheath_prates = []
    sheath_erates = []
    adv_rates = []
    parallel_rates = []
    other_prates = []
    split_residuals = []
    for i, ti in enumerate(ts):
        yi = FCIDRB3DFullState(
            n=ys.n[i],
            omega=ys.omega[i],
            vpar_e=ys.vpar_e[i],
            vpar_i=ys.vpar_i[i],
            Te=ys.Te[i],
            Ti=None if ys.Ti is None else ys.Ti[i],
            psi=None if ys.psi is None else ys.psi[i],
            N=None if ys.N is None else ys.N[i],
        )
        dyi = model.rhs(float(ti), yi)
        pb = model.particle_budget_terms(yi)
        eb = model.energy_budget_terms(yi)

        energy.append(float(model.energy(yi)))
        total_particles.append(float(model.total_particle_content(yi)))
        energy_rates.append(float(model.energy_rate(yi, dyi)))
        particle_rates.append(float(model.total_particle_rate(dyi)))
        sheath_prates.append(float(pb["sheath"]))
        sheath_erates.append(float(model.sheath_budget_rates(yi)[1]))
        adv_rates.append(float(pb["advective"]))
        parallel_rates.append(float(pb["parallel"]))
        other_prates.append(float(pb["other"]))
        split_residuals.append(float(eb["residual"]))

    energy = jnp.asarray(energy)
    total_particles = jnp.asarray(total_particles)
    energy_rates = jnp.asarray(energy_rates)
    particle_rates = jnp.asarray(particle_rates)
    sheath_prates = jnp.asarray(sheath_prates)
    sheath_erates = jnp.asarray(sheath_erates)
    adv_rates = jnp.asarray(adv_rates)
    parallel_rates = jnp.asarray(parallel_rates)
    other_prates = jnp.asarray(other_prates)
    split_residuals = jnp.asarray(split_residuals)

    assert bool(jnp.isfinite(ys.n).all())
    assert bool(jnp.isfinite(ys.Te).all())
    assert bool(jnp.isfinite(ys.psi).all())
    assert bool(jnp.isfinite(ys.N).all())

    dt_save = float(ts[1] - ts[0])
    dE_dt_fd = jnp.gradient(energy, dt_save)
    dP_dt_fd = jnp.gradient(total_particles, dt_save)
    rel_e = jnp.sqrt(jnp.mean((dE_dt_fd - energy_rates) ** 2)) / jnp.maximum(
        jnp.sqrt(jnp.mean(energy_rates**2)), 1e-12
    )
    rel_p = jnp.sqrt(jnp.mean((dP_dt_fd - particle_rates) ** 2)) / jnp.maximum(
        jnp.sqrt(jnp.mean(particle_rates**2)), 1e-12
    )

    # Finite-time budget closure from RHS rates.
    assert float(rel_e) < 3.5e-2
    assert float(rel_p) < 2.5e-2
    # Sheath closures should provide net losses in this setup.
    assert float(jnp.median(sheath_prates)) < 0.0
    assert float(jnp.median(sheath_erates)) < 0.0
    # Arakawa advection should keep the mean near roundoff.
    assert float(jnp.max(jnp.abs(adv_rates))) < 1e-12
    # Open-field-line target transport must be active (nonzero parallel net rate).
    assert float(jnp.max(jnp.abs(parallel_rates))) > 1e-5
    # With periodic perpendicular operators, mean-rate drift from other terms stays tiny.
    assert float(jnp.max(jnp.abs(other_prates))) < 1e-9
    # Split energy decomposition should close to roundoff.
    assert float(jnp.max(jnp.abs(split_residuals))) < 5e-12
