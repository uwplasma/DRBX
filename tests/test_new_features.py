from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.linear.arnoldi import arnoldi_leading_ritz_vector
from jaxdrb.linear.growthrate import estimate_growth_rate_jax
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams
from tests.helpers import state_to_vec


def test_arnoldi_leading_ritz_vector_has_small_residual() -> None:
    nl = 12
    geom = SlabGeometry.make(nl=nl, shat=0.4, curvature0=0.2)
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    y_eq = equilibrium(nl)
    matvec = linear_matvec(y_eq, params, geom, kx=0.0, ky=0.3)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)

    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=60, seed=0)

    v = np.asarray(state_to_vec(ritz.vector))
    Av = np.asarray(state_to_vec(matvec(ritz.vector)))
    resid = Av - np.asarray(ritz.eigenvalue) * v

    rel = np.linalg.norm(resid) / (np.linalg.norm(v) * (abs(ritz.eigenvalue) + 1.0))
    assert np.isfinite(rel)
    assert rel < 1e-8


def test_estimate_growth_rate_jax_is_differentiable() -> None:
    nl = 8
    geom = SlabGeometry.make(nl=nl, shat=0.2, curvature0=0.1)
    y_eq = equilibrium(nl)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)

    def gamma_of_omega_n(omega_n: jax.Array) -> jax.Array:
        params = DRBParams(
            omega_n=omega_n,
            omega_Te=0.0,
            eta=1.0,
            me_hat=0.05,
            curvature_on=True,
            Dn=0.01,
            DOmega=0.01,
            DTe=0.01,
        )
        matvec = linear_matvec(y_eq, params, geom, kx=0.0, ky=0.3)
        gr = estimate_growth_rate_jax(matvec, v0, tmax=6.0, dt0=0.02, nsave=60, fit_window=0.5)
        return gr.gamma

    g = gamma_of_omega_n(jnp.array(0.8))
    dg = jax.grad(gamma_of_omega_n)(jnp.array(0.8))

    assert jnp.isfinite(g)
    assert jnp.isfinite(dg)


def test_tabulated_geometry_reads_B(tmp_path) -> None:
    nl = 16
    l = np.linspace(0.0, 2.0 * np.pi, nl, endpoint=False)
    gxx = np.ones_like(l)
    gxy = np.zeros_like(l)
    gyy = np.ones_like(l)
    B = 1.0 + 0.1 * np.cos(l)

    path = tmp_path / "geom.npz"
    np.savez(path, l=l, gxx=gxx, gxy=gxy, gyy=gyy, B=B)

    geom = TabulatedGeometry.from_npz(path)
    assert np.allclose(np.asarray(geom.B()), B)
