"""Hermes species state-preparation mirror.

Planned in Phase 4 of `/Users/rogerio/local/jax_drb/plan.md`.
"""

from __future__ import annotations

import jax.numpy as jnp

from .boundary import apply_neumann_field3d
from .derivs import ddx_centered_guarded, ddy_centered_guarded_local
from .transform import (
    build_shifted_metric_fft_phases,
    build_shifted_metric_weights,
    to_field_aligned_all,
    to_field_aligned_all_fft,
)
from .types import FieldAlignedLocalLayout


def _as_field_aligned_metric(
    arr: jnp.ndarray | float,
    *,
    npar: int,
    nx: int,
    nbinorm: int,
    name: str,
) -> jnp.ndarray:
    out = jnp.asarray(arr, dtype=jnp.float64)
    if out.ndim == 0:
        return jnp.full((npar, nx), out, dtype=jnp.float64)
    if out.ndim == 1:
        if out.shape[0] == npar:
            return jnp.broadcast_to(out[:, None], (npar, nx))
        if out.shape[0] == nx:
            return jnp.broadcast_to(out[None, :], (npar, nx))
    if out.ndim == 2 and out.shape == (npar, nx):
        return out
    if out.ndim == 2 and out.shape == (npar, 1):
        return jnp.broadcast_to(out, (npar, nx))
    if out.ndim == 2 and out.shape == (1, nx):
        return jnp.broadcast_to(out, (npar, nx))
    if out.ndim == 3 and out.shape == (npar, nx, nbinorm):
        return out[..., 0]
    raise ValueError(
        f"{name} must be scalar, shape ({npar}, {nx}), or shape ({npar}, {nx}, {nbinorm}); "
        f"got {out.shape}."
    )


def prepare_poloidal_y_dfdx_local_ref(
    field: jnp.ndarray,
    *,
    dx: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    layout: FieldAlignedLocalLayout,
    interp: str = "spectral",
) -> jnp.ndarray:
    """Mirror the local `DDX -> communicate -> applyBoundary -> toFieldAligned` chain.

    Source of truth:
    - `/Users/rogerio/local/hermes-3/src/div_ops.cxx`
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/include/bout/index_derivs_interface.hxx`
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`

    This helper intentionally targets the local guard-inclusive field-aligned
    layout `(npar, nx, nbinorm)` used by the shifted-transform mirror tests,
    not the active solver storage contract.
    """

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    if field_arr.ndim != 3:
        raise ValueError(
            f"field must have shape `(npar, nx, nbinorm)` for local Y-flux prep, got {field_arr.shape}."
        )
    layout.validate(tuple(int(v) for v in field_arr.shape))
    npar, nx, nbinorm = (int(v) for v in field_arr.shape)
    dx2d = _as_field_aligned_metric(dx, npar=npar, nx=nx, nbinorm=nbinorm, name="dx")
    dx3d = jnp.broadcast_to(dx2d[:, :, None], field_arr.shape)
    dfdx = ddx_centered_guarded(field_arr, dx3d, layout=None)
    dfdx = apply_neumann_field3d(
        dfdx,
        axis=1,
        interior_start=layout.xstart,
        interior_end=layout.xend,
        spacing=dx3d,
        lower_gradient=0.0,
        upper_gradient=0.0,
        guard_width=layout.x_guards,
    )

    interp_name = str(interp).lower()
    if interp_name == "spectral":
        phases = build_shifted_metric_fft_phases(
            z_shift,
            nx=nx,
            npar=npar,
            nbinorm=nbinorm,
            zlength=float(zlength),
            open_field_line=layout.open_field_line,
        )
        return to_field_aligned_all_fft(dfdx, phases)

    if interp_name == "linear":
        dz = float(zlength) / max(nbinorm, 1)
        if dz <= 0.0:
            raise ValueError(f"zlength={zlength} gives invalid binormal spacing {dz}.")
        shift_idx = jnp.asarray(z_shift, dtype=jnp.float64) / dz
        weights = build_shifted_metric_weights(
            shift_idx,
            nx=nx,
            npar=npar,
            nbinorm=nbinorm,
            open_field_line=layout.open_field_line,
        )
        return to_field_aligned_all(dfdx, weights)

    raise ValueError(f"Unsupported interp={interp!r}; expected 'spectral' or 'linear'.")


def prepare_poloidal_y_dfdx_local(
    field: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    """Fused mirror entrypoint for the local Y-flux preparation chain."""

    return prepare_poloidal_y_dfdx_local_ref(field, **kwargs)


def prepare_poloidal_x_dfdy_local_ref(
    field: jnp.ndarray,
    *,
    dy: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    layout: FieldAlignedLocalLayout,
) -> jnp.ndarray:
    """Mirror the local `DDY -> communicate -> applyBoundary` chain for X-flux."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    if field_arr.ndim != 3:
        raise ValueError(
            f"field must have shape `(npar, nx, nbinorm)` for local X-flux prep, got {field_arr.shape}."
        )
    layout.validate(tuple(int(v) for v in field_arr.shape))
    npar, nx, nbinorm = (int(v) for v in field_arr.shape)
    dy3d = jnp.broadcast_to(
        _as_field_aligned_metric(dy, npar=npar, nx=nx, nbinorm=nbinorm, name="dy")[:, :, None],
        field_arr.shape,
    )
    dx3d = jnp.broadcast_to(
        _as_field_aligned_metric(dx, npar=npar, nx=nx, nbinorm=nbinorm, name="dx")[:, :, None],
        field_arr.shape,
    )
    dfdy = ddy_centered_guarded_local(field_arr, dy3d, layout=layout)
    return apply_neumann_field3d(
        dfdy,
        axis=1,
        interior_start=layout.xstart,
        interior_end=layout.xend,
        spacing=dx3d,
        lower_gradient=0.0,
        upper_gradient=0.0,
        guard_width=layout.x_guards,
    )


def prepare_poloidal_x_dfdy_local(
    field: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    """Fused mirror entrypoint for the local X-flux preparation chain."""

    return prepare_poloidal_x_dfdy_local_ref(field, **kwargs)


def density_transform_impl(*args, **kwargs):
    raise NotImplementedError("Phase 4: mirror density state preparation is not landed yet.")


def pressure_transform_impl(*args, **kwargs):
    raise NotImplementedError("Phase 4: mirror pressure state preparation is not landed yet.")
