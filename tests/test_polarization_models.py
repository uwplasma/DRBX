from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import Equilibrium, State, equilibrium, phi_from_omega
from jaxdrb.models.params import DRBParams


def test_phi_from_omega_non_boussinesq_scales_with_n0() -> None:
    nl = 5
    omega = (1.0 + 0.0j) * jnp.ones((nl,), dtype=jnp.complex128)
    k2 = 2.0 * jnp.ones((nl,), dtype=jnp.float64)

    phi_b = phi_from_omega(omega, k2, kperp2_min=1e-12, boussinesq=True)
    phi_nb = phi_from_omega(
        omega,
        k2,
        kperp2_min=1e-12,
        boussinesq=False,
        n0=2.0 * jnp.ones((nl,), dtype=jnp.float64),
        n0_min=1e-12,
    )
    assert np.allclose(np.asarray(phi_nb), np.asarray(phi_b) / 2.0)


def test_non_boussinesq_matches_boussinesq_when_n0_is_one() -> None:
    nl = 10
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    y0 = equilibrium(nl)
    v = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)

    params_b = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        boussinesq=True,
    )
    params_nb = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        boussinesq=False,
    )

    eq = Equilibrium.constant(nl, n0=1.0)
    mv_b = linear_matvec(y0, params_b, geom, kx=0.0, ky=0.3, eq=eq)
    mv_nb = linear_matvec(y0, params_nb, geom, kx=0.0, ky=0.3, eq=eq)

    out_b = mv_b(v)
    out_nb = mv_nb(v)

    assert np.allclose(np.asarray(out_nb.n), np.asarray(out_b.n))
    assert np.allclose(np.asarray(out_nb.omega), np.asarray(out_b.omega))
    assert np.allclose(np.asarray(out_nb.vpar_e), np.asarray(out_b.vpar_e))
    assert np.allclose(np.asarray(out_nb.vpar_i), np.asarray(out_b.vpar_i))
    assert np.allclose(np.asarray(out_nb.Te), np.asarray(out_b.Te))


def test_non_boussinesq_changes_dynamics_when_n0_not_one() -> None:
    nl = 10
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    y0 = equilibrium(nl)
    v = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)

    params_b = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        boussinesq=True,
    )
    params_nb = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        boussinesq=False,
    )

    eq = Equilibrium.constant(nl, n0=2.0)
    mv_b = linear_matvec(y0, params_b, geom, kx=0.0, ky=0.3, eq=eq)
    mv_nb = linear_matvec(y0, params_nb, geom, kx=0.0, ky=0.3, eq=eq)

    out_b = mv_b(v)
    out_nb = mv_nb(v)

    diff = np.linalg.norm(np.asarray(out_nb.n - out_b.n))
    ref = np.linalg.norm(np.asarray(out_b.n)) + 1e-30
    assert diff / ref > 1e-6
