from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.driver import build_system_from_config
from jaxdrb.parity_fv import laplacian_xy_spectral, solve_poisson_xy_spectral


def _periodic_mode(nz: int, nx: int, ny: int) -> jnp.ndarray:
    z = jnp.arange(nz)[:, None, None]
    x = jnp.arange(nx)[None, :, None]
    y = jnp.arange(ny)[None, None, :]
    return jnp.sin(2.0 * jnp.pi * x / nx) * jnp.cos(2.0 * jnp.pi * y / ny) * (1.0 + 0.1 * z)


def test_solve_poisson_xy_spectral_inverts_laplacian_mode() -> None:
    nz, nx, ny = 4, 32, 24
    dx = 1.0 / nx
    dy = 1.0 / ny
    phi = _periodic_mode(nz, nx, ny)
    omega = laplacian_xy_spectral(phi, dx=dx, dy=dy)
    phi_rec = solve_poisson_xy_spectral(omega, dx=dx, dy=dy, gauge_fix=True)

    # Mean-free mode should invert to numerical precision.
    err = np.asarray(phi_rec - phi)
    assert float(np.sqrt(np.mean(err * err))) < 1e-10


def test_parity_fv_system_poisson_roundtrip_spectral() -> None:
    cfg = {
        "engine": "parity_fv",
        "geometry": {"kind": "slab", "nx": 28, "ny": 20, "nz": 3, "Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
        "numerics": {"poisson_scale": 1.0, "parity_poisson_solver": "spectral_xy"},
        "initial": {"n0": 1.0, "Te0": 1.0},
    }
    built = build_system_from_config(cfg)
    phi = _periodic_mode(3, 28, 20)
    omega = built.system._omega_from_phi(phi)
    phi_rec = built.system._phi_from_omega(omega)
    err = np.asarray(phi_rec - phi)
    assert float(np.sqrt(np.mean(err * err))) < 1e-10


def test_parity_fv_system_poisson_identity_scaling() -> None:
    cfg = {
        "engine": "parity_fv",
        "geometry": {"kind": "slab", "nx": 12, "ny": 10, "nz": 5, "Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
        "numerics": {"poisson_scale": 2.5, "parity_poisson_solver": "identity"},
        "initial": {"n0": 1.0, "Te0": 1.0},
    }
    built = build_system_from_config(cfg)
    omega = jnp.linspace(0.0, 1.0, np.prod(built.state.n.shape)).reshape(built.state.n.shape)
    phi = built.system._phi_from_omega(omega)
    omega_back = built.system._omega_from_phi(phi)
    assert np.allclose(np.asarray(omega_back), np.asarray(omega), atol=1e-12, rtol=1e-12)
