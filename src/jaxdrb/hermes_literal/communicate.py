"""Literal local communication helpers for parallel subdomains.

These helpers mirror the numerical effect of BOUT/Hermes processor-local
parallel guard exchange for the Stage 1 baseline without reproducing the full
MPI layer. They construct local guard-inclusive slabs from global physical
arrays so local literal operators can run against the same storage contract
used by the Hermes sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .boundary import apply_neumann_field3d
from .types import FieldAlignedLocalLayout


@dataclass(frozen=True)
class ParallelSubdomain:
    """Metadata for a local parallel slab embedded in guard cells."""

    start: int
    stop: int
    guard_width: int = 2
    open_field_line: bool = True

    @property
    def block(self) -> int:
        return int(self.stop - self.start)

    def layout(self, *, nx: int) -> FieldAlignedLocalLayout:
        gw = int(self.guard_width)
        return FieldAlignedLocalLayout(
            pstart=gw,
            pend=gw + self.block - 1,
            xstart=gw,
            xend=gw + int(nx) - 1,
            p_guards=gw,
            x_guards=gw,
            open_field_line=bool(self.open_field_line),
        )


def build_parallel_subdomain(
    *,
    start: int,
    stop: int,
    guard_width: int = 2,
    open_field_line: bool = True,
) -> ParallelSubdomain:
    if int(stop) <= int(start):
        raise ValueError(f"Invalid subdomain interval [{start}, {stop}).")
    return ParallelSubdomain(
        start=int(start),
        stop=int(stop),
        guard_width=int(guard_width),
        open_field_line=bool(open_field_line),
    )


def _pad_missing_guards(
    out: jnp.ndarray,
    *,
    fill_low: jnp.ndarray,
    fill_high: jnp.ndarray,
    filled_low: int,
    filled_high: int,
    guard_width: int,
    block: int,
) -> jnp.ndarray:
    result = out
    if filled_low < guard_width:
        n_missing = guard_width - filled_low
        result = result.at[:n_missing].set(jnp.broadcast_to(fill_low, result[:n_missing].shape))
    if filled_high < guard_width:
        start = guard_width + block + filled_high
        n_missing = guard_width - filled_high
        result = result.at[start : start + n_missing].set(
            jnp.broadcast_to(fill_high, result[start : start + n_missing].shape)
        )
    return result


def slice_parallel_subdomain_3d(
    field: jnp.ndarray,
    *,
    subdomain: ParallelSubdomain,
    periodic_parallel: bool = False,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = True,
    neighbor_planes: bool = False,
) -> jnp.ndarray:
    """Construct a guard-inclusive local `(npar, nx, ny)` slab from a global field."""

    arr = jnp.asarray(field, dtype=jnp.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D field, got shape {arr.shape}.")

    npar = int(arr.shape[0])
    gw = int(subdomain.guard_width)
    start = int(subdomain.start)
    stop = int(subdomain.stop)
    block = int(subdomain.block)

    if start < 0 or stop > npar:
        raise ValueError(f"Subdomain [{start}, {stop}) is outside field shape {arr.shape}.")

    if periodic_parallel:
        idx = jnp.mod(jnp.arange(start - gw, stop + gw), npar)
        return arr[idx]

    out = jnp.zeros((block + 2 * gw, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
    core = arr[start:stop]
    out = out.at[gw : gw + block].set(core)

    if neighbor_planes:
        low_src = arr[max(0, start - gw) : start]
        high_src = arr[stop : min(npar, stop + gw)]
    else:
        low_src = arr[0:0]
        high_src = arr[0:0]
    n_low = int(low_src.shape[0])
    n_high = int(high_src.shape[0])
    if n_low > 0:
        out = out.at[gw - n_low : gw].set(low_src)
    if n_high > 0:
        out = out.at[gw + block : gw + block + n_high].set(high_src)

    fill_low = core[:1]
    fill_high = core[-1:]
    if n_low > 0:
        fill_low = low_src[:1]
    if n_high > 0:
        fill_high = high_src[-1:]
    out = _pad_missing_guards(
        out,
        fill_low=fill_low,
        fill_high=fill_high,
        filled_low=n_low,
        filled_high=n_high,
        guard_width=gw,
        block=block,
    )

    if start == 0 and lower_boundary_open:
        out = apply_neumann_field3d(
            out,
            axis=0,
            interior_start=gw,
            interior_end=gw + block - 1,
            spacing=1.0,
            lower_gradient=0.0,
            upper_gradient=0.0,
            guard_width=gw,
            apply_lower=True,
            apply_upper=False,
        )
    if stop == npar and upper_boundary_open:
        out = apply_neumann_field3d(
            out,
            axis=0,
            interior_start=gw,
            interior_end=gw + block - 1,
            spacing=1.0,
            lower_gradient=0.0,
            upper_gradient=0.0,
            guard_width=gw,
            apply_lower=False,
            apply_upper=True,
        )
    return out


def slice_parallel_subdomain_2d(
    field: jnp.ndarray,
    *,
    subdomain: ParallelSubdomain,
    periodic_parallel: bool = False,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = True,
    neighbor_planes: bool = False,
) -> jnp.ndarray:
    """2D metric wrapper around :func:`slice_parallel_subdomain_3d`."""

    arr = jnp.asarray(field, dtype=jnp.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field, got shape {arr.shape}.")
    out = slice_parallel_subdomain_3d(
        arr[:, :, None],
        subdomain=subdomain,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
        neighbor_planes=neighbor_planes,
    )
    return out[:, :, 0]
