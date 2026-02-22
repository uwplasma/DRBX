from __future__ import annotations

from typing import Callable, Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.nonlinear import fd as fd_ops
from jaxdrb.nonlinear import fv as fv_ops
from jaxdrb.nonlinear import spectral as spec_ops
from jaxdrb.operators.brackets import (
    poisson_bracket_arakawa,
    poisson_bracket_arakawa_fd,
    poisson_bracket_centered,
)

PerpScheme = Literal["spectral", "fd", "fv"]
BracketScheme = Literal["spectral", "arakawa", "centered"]


class PerpOperatorBundle(eqx.Module):
    """Container for perpendicular operators on a single (x, y) plane."""

    scheme: PerpScheme = eqx.field(static=True)
    bracket: BracketScheme = eqx.field(static=True)

    # Spatial steps (used by FD/FV and Arakawa).
    dx: float
    dy: float
    dealias_on: bool = eqx.field(static=True, default=True)
    bracket_zero_mean: bool = eqx.field(static=True, default=False)
    bc: BC2D | None = None

    # Spectral caches (optional).
    kx: jnp.ndarray | None = None
    ky: jnp.ndarray | None = None
    k2: jnp.ndarray | None = None

    def ddx(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.scheme == "spectral":
            return spec_ops.ddx(f, self.kx)
        if self.scheme == "fd":
            if self.bc is None:
                raise ValueError("FD ddx requires BC2D.")
            return fd_ops.ddx(f, self.dx, self.bc)
        if self.scheme == "fv":
            if self.bc is None:
                raise ValueError("FV ddx requires BC2D.")
            return fv_ops.ddx(f, self.dx, self.bc)
        raise ValueError(f"Unsupported perp scheme: {self.scheme}")

    def ddy(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.scheme == "spectral":
            return spec_ops.ddy(f, self.ky)
        if self.scheme == "fd":
            if self.bc is None:
                raise ValueError("FD ddy requires BC2D.")
            return fd_ops.ddy(f, self.dy, self.bc)
        if self.scheme == "fv":
            if self.bc is None:
                raise ValueError("FV ddy requires BC2D.")
            return fv_ops.ddy(f, self.dy, self.bc)
        raise ValueError(f"Unsupported perp scheme: {self.scheme}")

    def laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.scheme == "spectral":
            return spec_ops.laplacian(f, self.k2)
        if self.scheme == "fd":
            if self.bc is None:
                raise ValueError("FD laplacian requires BC2D.")
            return fd_ops.laplacian(f, self.dx, self.dy, self.bc)
        if self.scheme == "fv":
            if self.bc is None:
                raise ValueError("FV laplacian requires BC2D.")
            return fv_ops.laplacian(f, self.dx, self.dy, self.bc)
        raise ValueError(f"Unsupported perp scheme: {self.scheme}")

    def biharmonic(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.scheme == "spectral":
            return spec_ops.biharmonic(f, self.k2)
        if self.scheme == "fd":
            if self.bc is None:
                raise ValueError("FD biharmonic requires BC2D.")
            return fd_ops.biharmonic(f, self.dx, self.dy, self.bc)
        raise ValueError("Biharmonic is not implemented for FV operators.")

    def inv_laplacian(self, f: jnp.ndarray, *, k2_min: float) -> jnp.ndarray:
        if self.scheme != "spectral":
            raise ValueError("inv_laplacian is only available in spectral mode.")
        return spec_ops.inv_laplacian(f, self.k2, k2_min=k2_min)

    def bracket_op(
        self,
        phi: jnp.ndarray,
        f: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_f: BC2D | None = None,
    ) -> jnp.ndarray:
        if self.bracket == "spectral":
            if self.scheme != "spectral":
                raise ValueError("Spectral bracket requires spectral perp operators.")
            return spec_ops.poisson_bracket_spectral(
                phi, f, kx=self.kx, ky=self.ky, dealias_on=self.dealias_on
            )

        if self.bracket == "arakawa":
            if bc_phi is None:
                return poisson_bracket_arakawa(phi, f, self.dx, self.dy)
            j = poisson_bracket_arakawa_fd(phi, f, self.dx, self.dy, bc_phi, bc_f)
        else:
            j = poisson_bracket_centered(phi, f, self.dx, self.dy)

        if self.bracket_zero_mean:
            j = j - j.mean()
        return j


def _spec_kgrid(
    nx: int, ny: int, dx: float, dy: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    kx_1d = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx))
    ky_1d = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy))
    kx, ky = jnp.meshgrid(kx_1d, ky_1d, indexing="ij")
    return kx, ky, kx**2 + ky**2


def build_perp_operator_bundle(
    *,
    scheme: PerpScheme,
    bracket: BracketScheme,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dealias_on: bool = True,
    bracket_zero_mean: bool = False,
    bc: BC2D | None = None,
) -> PerpOperatorBundle:
    """Build a PerpOperatorBundle with precomputed spectral caches."""

    if scheme == "spectral":
        kx, ky, k2 = _spec_kgrid(nx, ny, dx, dy)
    else:
        kx = ky = k2 = None
    return PerpOperatorBundle(
        scheme=scheme,
        bracket=bracket,
        dealias_on=dealias_on,
        bracket_zero_mean=bracket_zero_mean,
        dx=float(dx),
        dy=float(dy),
        bc=bc,
        kx=kx,
        ky=ky,
        k2=k2,
    )
