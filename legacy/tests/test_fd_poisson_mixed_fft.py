from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.nonlinear.fd import inv_laplacian_mixed_fft, laplacian


def test_inv_laplacian_mixed_fft_recovers_phi() -> None:
    jax.config.update("jax_enable_x64", True)

    nx, ny = 48, 64
    Lx, Ly = 1.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / ny
    bc = BC2D(kind_x=2, kind_y=0, x_grad=0.0, y_grad=0.0)

    key = jax.random.key(0)
    phi = jax.random.normal(key, (nx, ny))
    phi = phi - jnp.mean(phi)
    rhs = laplacian(phi, dx, dy, bc)

    phi_rec = inv_laplacian_mixed_fft(rhs, dx=dx, dy=dy, bc=bc)
    phi_rec = phi_rec - jnp.mean(phi_rec)

    err = jnp.max(jnp.abs(phi_rec - phi))
    assert float(err) < 1e-8
