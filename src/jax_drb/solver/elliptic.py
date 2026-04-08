from __future__ import annotations

from dataclasses import dataclass

from jax import lax

import jax.numpy as jnp


@dataclass(frozen=True)
class FourierHelmholtzOperator:
    lower_diagonals: jnp.ndarray
    diagonals: jnp.ndarray
    upper_diagonals: jnp.ndarray
    rhs_scale: jnp.ndarray
    nz: int
    zlength: float


def build_fourier_helmholtz_operator(
    *,
    dx: jnp.ndarray,
    dz: jnp.ndarray,
    g11: jnp.ndarray,
    g33: jnp.ndarray,
    rhs_scale: jnp.ndarray,
    nz: int,
) -> FourierHelmholtzOperator:
    dx = jnp.asarray(dx, dtype=jnp.float64)
    dz = jnp.asarray(dz, dtype=jnp.float64)
    g11 = jnp.asarray(g11, dtype=jnp.float64)
    g33 = jnp.asarray(g33, dtype=jnp.float64)
    rhs_scale = jnp.asarray(rhs_scale, dtype=jnp.float64)

    zlength = float(dz[0]) * float(nz)
    x_coef = g11 / (dx * dx)
    modes = nz // 2 + 1
    wave_numbers = (2.0 * jnp.pi * jnp.arange(modes, dtype=jnp.float64)) / zlength
    diagonals = -2.0 * x_coef[None, :] - jnp.square(wave_numbers)[:, None] * g33[None, :]
    diagonals = diagonals.at[:, 0].add(-x_coef[0])
    diagonals = diagonals.at[:, -1].add(-x_coef[-1])

    lower_diagonals = jnp.zeros_like(diagonals, dtype=jnp.complex128)
    upper_diagonals = jnp.zeros_like(diagonals, dtype=jnp.complex128)
    lower_diagonals = lower_diagonals.at[:, 1:].set(x_coef[1:][None, :].astype(jnp.complex128))
    upper_diagonals = upper_diagonals.at[:, :-1].set(x_coef[:-1][None, :].astype(jnp.complex128))

    return FourierHelmholtzOperator(
        lower_diagonals=lower_diagonals,
        diagonals=diagonals.astype(jnp.complex128),
        upper_diagonals=upper_diagonals,
        rhs_scale=rhs_scale,
        nz=int(nz),
        zlength=zlength,
    )


def solve_fourier_helmholtz(
    rhs: jnp.ndarray,
    *,
    operator: FourierHelmholtzOperator,
) -> jnp.ndarray:
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    rhs_hat = jnp.fft.rfft(rhs * operator.rhs_scale[:, None], axis=-1)
    mode_rhs = jnp.swapaxes(rhs_hat, 0, 1)
    mode_solution = lax.linalg.tridiagonal_solve(
        operator.lower_diagonals,
        operator.diagonals,
        operator.upper_diagonals,
        mode_rhs[..., None],
    ).squeeze(-1)
    interior_hat = jnp.swapaxes(mode_solution, 0, 1)
    return jnp.fft.irfft(interior_hat, n=operator.nz, axis=-1)
