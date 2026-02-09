from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.invariants import cold_ion_invariants
from jaxdrb.models.params import DRBParams


def _rk4_step(y: State, t: jnp.ndarray, dt: float, rhs_fn) -> State:
    def add(a: State, b: State, scale: float) -> State:
        return jax.tree_util.tree_map(lambda x, y_: x + scale * y_, a, b)

    k1 = rhs_fn(t, y)
    k2 = rhs_fn(t + 0.5 * dt, add(y, k1, 0.5 * dt))
    k3 = rhs_fn(t + 0.5 * dt, add(y, k2, 0.5 * dt))
    k4 = rhs_fn(t + dt, add(y, k3, dt))
    return jax.tree_util.tree_map(
        lambda a, b, c, d, e: a + (dt / 6.0) * (b + 2.0 * c + 2.0 * d + e),
        y,
        k1,
        k2,
        k3,
        k4,
    )


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

    @jax.jit
    def evolve_and_measure(y_init: State) -> jnp.ndarray:
        inv0 = cold_ion_invariants(y_init, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
        v0 = jnp.array(
            [inv0["energy"], inv0["mass"], inv0["charge"], inv0["current"], inv0["momentum"]]
        )

        def scan_step(carry, _):
            t, y = carry
            y_next = _rk4_step(y, t, dt, rhs_local)
            inv = cold_ion_invariants(y_next, params=params, geom=geom, kx=kx, ky=ky, eq=eq)
            vec = jnp.array(
                [inv["energy"], inv["mass"], inv["charge"], inv["current"], inv["momentum"]]
            )
            return (t + dt, y_next), vec

        (_, _), vals = jax.lax.scan(
            scan_step, (jnp.asarray(0.0, dtype=jnp.float64), y_init), xs=None, length=nsteps
        )
        return jnp.vstack([v0[None, :], vals])

    V = np.asarray(evolve_and_measure(y0))

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

    @jax.jit
    def evolve(y_init: State) -> tuple[jnp.ndarray, jnp.ndarray]:
        E0 = cold_ion_invariants(y_init, params=params, geom=geom, kx=kx, ky=ky, eq=eq)["energy"]

        def body(i, carry):
            t, y = carry
            return (t + dt, _rk4_step(y, t, dt, rhs_local))

        _, y1 = jax.lax.fori_loop(0, nsteps, body, (jnp.asarray(0.0, dtype=jnp.float64), y_init))
        E1 = cold_ion_invariants(y1, params=params, geom=geom, kx=kx, ky=ky, eq=eq)["energy"]
        return E0, E1

    E0, E1 = [float(x) for x in evolve(y0)]
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

    # Directional derivative for the simple mean invariants.
    dmass = float(jnp.mean(jnp.real(dy.n)))
    dcharge = float(jnp.mean(jnp.real(dy.omega)))
    dcurrent = float(jnp.mean(jnp.real(dy.vpar_i - dy.vpar_e)))
    dmom = float(jnp.mean(jnp.real(dy.vpar_i + params.me_hat * dy.vpar_e)))

    # Energy rate by centered finite-difference directional derivative of E(y).
    # A centered estimate is much less sensitive to cancellation error than a one-sided
    # difference in this near-conservative limit.
    eps = 1.0e-7
    E0 = float(cold_ion_invariants(y, params=params, geom=geom, kx=kx, ky=ky, eq=eq)["energy"])
    y_plus = jax.tree_util.tree_map(lambda a, b: a + eps * b, y, dy)
    y_minus = jax.tree_util.tree_map(lambda a, b: a - eps * b, y, dy)
    E_plus = float(
        cold_ion_invariants(y_plus, params=params, geom=geom, kx=kx, ky=ky, eq=eq)["energy"]
    )
    E_minus = float(
        cold_ion_invariants(y_minus, params=params, geom=geom, kx=kx, ky=ky, eq=eq)["energy"]
    )
    dE_fd = (E_plus - E_minus) / (2.0 * eps)

    assert abs(dmass) < 5e-12
    assert abs(dcharge) < 5e-12
    assert abs(dcurrent) < 5e-12
    assert abs(dmom) < 5e-12
    assert abs(dE_fd) < 2e-8
