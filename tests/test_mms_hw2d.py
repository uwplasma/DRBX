from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams
from jaxdrb.nonlinear.mms_hw2d import MMSHW2D


def _mms_error(*, nx: int, bracket: str) -> float:
    Lx = 2 * jnp.pi
    Ly = 2 * jnp.pi
    grid = Grid2D.make(nx=nx, ny=nx, Lx=Lx, Ly=Ly, dealias=False)
    params = HW2DParams(
        kappa=0.7,
        alpha=0.4,
        Dn=1e-3,
        DOmega=2e-3,
        bracket=bracket,  # type: ignore[arg-type]
        poisson="spectral",
        dealias_on=False,
    )
    model = HW2DModel(params=params, grid=grid)

    mms = MMSHW2D(kx=3, ky=2, sigma=0.2, Aphi=0.25, An=0.2, phase=0.3)
    t = 0.4

    x1 = jnp.linspace(0.0, Lx, nx, endpoint=False)
    y1 = jnp.linspace(0.0, Ly, nx, endpoint=False)
    X, Y = jnp.meshgrid(x1, y1, indexing="ij")

    y = mms.state(X, Y, t, Lx=Lx, Ly=Ly)
    dy = model.rhs(t, y)

    f = mms.forcing(
        X,
        Y,
        t,
        Lx=Lx,
        Ly=Ly,
        kappa=params.kappa,
        alpha=params.alpha,
        Dn=params.Dn,
        DOmega=params.DOmega,
    )

    # Exact time derivatives.
    dt_n = mms.sigma * y.n
    dt_w = mms.sigma * y.omega

    rn = dt_n - dy.n - f.n
    rw = dt_w - dy.omega - f.omega
    err = jnp.sqrt(jnp.mean(rn**2 + rw**2))
    return float(err)


def test_mms_hw2d_spectral_is_near_machine_precision():
    err = _mms_error(nx=64, bracket="spectral")
    assert err < 1e-10


def test_mms_hw2d_arakawa_converges_second_order():
    e1 = _mms_error(nx=32, bracket="arakawa")
    e2 = _mms_error(nx=64, bracket="arakawa")
    assert e2 < e1
    assert e1 / e2 > 3.0


def test_mms_hw2d_centered_converges_second_order():
    e1 = _mms_error(nx=32, bracket="centered")
    e2 = _mms_error(nx=64, bracket="centered")
    assert e2 < e1
    assert e1 / e2 > 3.0
