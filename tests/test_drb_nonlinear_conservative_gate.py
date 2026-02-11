from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.nonlinear.integrate import diffeqsolve, diffeqsolve_fixed_steps
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.invariants import cold_ion_invariant_rates_from_rhs, cold_ion_invariants
from jaxdrb.models.params import DRBParams


def test_cold_ion_drb_conservative_gate_energy_mass_charge_current_momentum() -> None:
    """Hard gate for conservative invariants in the actual field-line DRB branch.

    This checks the implemented cold-ion DRB equations (not HW2D) in a periodic conservative limit.
    """
    nl = 64
    kx = 0.0
    ky = 0.35
    dt = 1.0e-3
    nsteps = 2000

    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        alpha_Te_ohm=1.71,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        boussinesq=True,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )

    y0 = State.random(jax.random.key(123), nl, amplitude=1.0e-2)

    def rhs_local(t: jnp.ndarray, y: State) -> State:
        return rhs_nonlinear(t, y, params, geom, kx=kx, ky=ky, eq=eq)

    inv0 = cold_ion_invariants(y0, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
    v0 = jnp.array(
        [inv0["energy"], inv0["mass"], inv0["charge"], inv0["current"], inv0["momentum"]]
    )

    ys, _ = diffeqsolve_fixed_steps(
        rhs_local,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
        save_every=1,
    )

    def inv_vec(y: State) -> jnp.ndarray:
        inv = cold_ion_invariants(y, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
        return jnp.array(
            [inv["energy"], inv["mass"], inv["charge"], inv["current"], inv["momentum"]]
        )

    vals = jax.vmap(inv_vec)(ys)
    V = np.asarray(jnp.vstack([v0[None, :], vals]))

    # 0: energy (relative span + end drift), 1..4: conservative means (absolute spans).
    E = V[:, 0]
    E0 = E[0]
    rel_span_E = (np.max(E) - np.min(E)) / max(abs(E0), 1e-30)
    rel_end_E = abs(E[-1] - E0) / max(abs(E0), 1e-30)

    mass_span = float(np.max(np.abs(V[:, 1] - V[0, 1])))
    charge_span = float(np.max(np.abs(V[:, 2] - V[0, 2])))
    current_span = float(np.max(np.abs(V[:, 3] - V[0, 3])))
    mom_span = float(np.max(np.abs(V[:, 4] - V[0, 4])))

    assert rel_span_E < 2e-4
    assert rel_end_E < 2e-4
    assert mass_span < 2e-10
    assert charge_span < 2e-10
    assert current_span < 2e-10
    assert mom_span < 2e-10


def test_cold_ion_drb_energy_decreases_with_resistivity() -> None:
    """In the no-drive periodic limit, finite resistivity should dissipate the quadratic energy."""
    nl = 64
    kx = 0.0
    ky = 0.35
    dt = 1.0e-3
    nsteps = 1000

    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.2,
        alpha_Te_ohm=1.71,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        boussinesq=True,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    y0 = State.random(jax.random.key(7), nl, amplitude=1.0e-2)

    def rhs_local(t: jnp.ndarray, y: State) -> State:
        return rhs_nonlinear(t, y, params, geom, kx=kx, ky=ky, eq=eq)

    E0 = float(cold_ion_invariants(y0, params=params, geom=geom, kx=kx, ky=ky, eq=eq)["energy"])
    sol = diffeqsolve(
        rhs_local,
        y0=y0,
        t0=0.0,
        t1=float(dt * nsteps),
        dt0=float(dt),
        solver="dopri5",
        adaptive=False,
        save_ts=None,
        max_steps=nsteps * 2 + 200,
        progress=False,
    )
    E1 = float(cold_ion_invariants(sol.ys, params=params, geom=geom, kx=kx, ky=ky, eq=eq)["energy"])
    assert E1 < E0


def test_cold_ion_drb_instantaneous_invariant_rates_vanish_in_conservative_subset() -> None:
    """RHS-level check: d/dt of conservative diagnostics is ~0 at random state."""
    nl = 64
    kx = 0.0
    ky = 0.35
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        alpha_Te_ohm=1.71,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        boussinesq=True,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    y = State.random(jax.random.key(99), nl, amplitude=1.0e-2)
    dy = rhs_nonlinear(0.0, y, params, geom, kx=kx, ky=ky, eq=eq)

    rates = cold_ion_invariant_rates_from_rhs(y, dy, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
    dmass = float(rates["dmass_dt"])
    dcharge = float(rates["dcharge_dt"])
    dcurrent = float(rates["dcurrent_dt"])
    dmom = float(rates["dmomentum_dt"])
    dE_fd = float(rates["denergy_dt"])

    assert abs(dmass) < 5e-12
    assert abs(dcharge) < 5e-12
    assert abs(dcurrent) < 5e-12
    assert abs(dmom) < 5e-12
    assert abs(dE_fd) < 2e-8


def test_cold_ion_operator_gate_multi_seed_multi_ky() -> None:
    """Strict operator gate on the periodic conservative branch across seeds and k_y."""
    nl = 64
    kx = 0.0
    kys = jnp.asarray([0.12, 0.28, 0.45, 0.72], dtype=jnp.float64)
    seeds = [5, 17, 29, 41]
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        alpha_Te_ohm=1.71,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        boussinesq=True,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )

    def max_abs_rates(y: State, ky: float) -> jnp.ndarray:
        dy = rhs_nonlinear(0.0, y, params, geom, kx=kx, ky=ky, eq=eq)
        rates = cold_ion_invariant_rates_from_rhs(
            y, dy, params=params, geom=geom, kx=kx, ky=ky, eq=eq
        )
        return jnp.asarray(
            [
                jnp.abs(rates["denergy_dt"]),
                jnp.abs(rates["dmass_dt"]),
                jnp.abs(rates["dcharge_dt"]),
                jnp.abs(rates["dcurrent_dt"]),
                jnp.abs(rates["dmomentum_dt"]),
            ],
            dtype=jnp.float64,
        )

    maxima = np.zeros(5, dtype=float)
    for seed in seeds:
        y = State.random(jax.random.key(seed), nl, amplitude=1.0e-2)
        for ky in np.asarray(kys):
            vals = np.asarray(max_abs_rates(y, ky))
            maxima = np.maximum(maxima, vals)

    # [energy, mass, charge, current, momentum]
    assert maxima[0] < 5e-8
    assert maxima[1] < 1e-11
    assert maxima[2] < 1e-11
    assert maxima[3] < 1e-11
    assert maxima[4] < 1e-11
