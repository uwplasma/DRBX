"""Analytic island-divertor stellarator field (B8).

The rotating-ellipse embedding carrying an *island-divertor* magnetic field: the
rotational transform is sheared, ``iota(x)`` rising from axis to edge, and
resonant radial field perturbations ``B^x ~ eps sin(m theta - n zeta)`` open
magnetic island chains at the rational surfaces ``iota = n/m``. With two
overlapping chains the edge becomes stochastic: edge field lines wander
radially and strike the wall — genuinely **open** field lines whose endpoint
masks come from the field-line tracer, not from a hand-placed limiter — while
the core below the resonances stays closed. This is the analytic island-divertor
configuration of the B8 benchmark ladder rung.

The field components (contravariant, with ``J`` the metric Jacobian):

    B^zeta  = c / J
    B^theta = iota(x) * c / J
    B^x     = (c / J) * sum_k eps_k * sin(m_k theta - n_k zeta)

so field lines obey ``dx/dzeta = sum_k eps_k sin(m_k theta - n_k zeta)`` and
``dtheta/dzeta = iota(x)`` — the standard perturbed-twist-map form whose island
half-width at a resonance is ``sqrt(eps m / iota')``. The helper
:func:`island_divertor_field_line_rhs` exposes exactly that reduced system for
Poincare / connection-length studies, and the full :class:`FciGeometry3D` uses
the same components so the FCI tracer sees the same field.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .embedding import metric_from_position_fn
from .fci_geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    MetricGeometry,
    Spacing3D,
    _bmag_from_contravariant_components,
    build_fci_maps_from_b_contravariant,
    logical_grid_from_axis_vectors,
)
from .rotating_ellipse import rotating_ellipse_position

__all__ = [
    "IslandDivertorField",
    "build_island_divertor_geometry",
    "island_divertor_connection_length",
    "island_divertor_field_line_rhs",
]


@dataclass(frozen=True)
class IslandDivertorField:
    """The analytic island-divertor field parameters.

    ``resonances`` is a tuple of ``(m, n, epsilon)`` triplets; each opens an
    island chain at the surface where ``iota(x) = n / m``. ``iota`` varies
    linearly from ``iota_axis`` at ``x_min`` to ``iota_edge`` at ``x_max``.
    """

    x_min: float = 0.2
    x_max: float = 1.0
    iota_axis: float = 0.55
    iota_edge: float = 0.82
    resonances: tuple[tuple[int, int, float], ...] = ((3, 2, 0.005), (4, 3, 0.005), (5, 4, 0.005))

    def iota(self, x):
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        return self.iota_axis + (self.iota_edge - self.iota_axis) * x_norm

    def radial_perturbation(self, theta, zeta):
        perturbation = jnp.zeros_like(jnp.asarray(theta, dtype=jnp.float64) + jnp.asarray(zeta))
        for m, n, epsilon in self.resonances:
            perturbation = perturbation + float(epsilon) * jnp.sin(m * theta - n * zeta)
        return perturbation


def island_divertor_field_line_rhs(field: IslandDivertorField, x, theta, zeta):
    """Field-line equations ``(dx/dzeta, dtheta/dzeta)`` of the reduced system."""

    return field.radial_perturbation(theta, zeta), field.iota(x)


def island_divertor_connection_length(
    field: IslandDivertorField,
    x: jnp.ndarray,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    *,
    direction: int = 1,
    max_transits: int = 40,
    steps_per_transit: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trace field lines and return ``(connection_length_transits, is_open)``.

    Integrates the reduced field-line system with RK4 from every starting point
    (arrays broadcast together) for up to ``max_transits`` toroidal transits in
    the given ``direction``. A line is **open** if it leaves the radial domain;
    its connection length is the number of transits to the exit (open lines) or
    ``max_transits`` (closed lines). Pure JAX (``vmap``/``scan``), so a full 3D
    grid of starting points traces in one shot.
    """

    x0, theta0, zeta0 = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(theta, dtype=jnp.float64),
        jnp.asarray(zeta, dtype=jnp.float64),
    )
    shape = x0.shape
    dz = float(direction) * 2.0 * jnp.pi / float(steps_per_transit)

    def rhs(x_value, theta_value, zeta_value):
        return field.radial_perturbation(theta_value, zeta_value), field.iota(x_value)

    def step(carry, _):
        x_value, theta_value, zeta_value, alive, steps_alive = carry
        k1x, k1t = rhs(x_value, theta_value, zeta_value)
        k2x, k2t = rhs(x_value + 0.5 * dz * k1x, theta_value + 0.5 * dz * k1t, zeta_value + 0.5 * dz)
        k3x, k3t = rhs(x_value + 0.5 * dz * k2x, theta_value + 0.5 * dz * k2t, zeta_value + 0.5 * dz)
        k4x, k4t = rhs(x_value + dz * k3x, theta_value + dz * k3t, zeta_value + dz)
        x_next = x_value + dz / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        theta_next = theta_value + dz / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        # Only the outer boundary is material (the divertor wall); the inner
        # boundary is the artificial core cutoff, so reflect there.
        x_next = jnp.maximum(x_next, field.x_min)
        alive_next = alive & (x_next < field.x_max)
        x_next = jnp.where(alive, x_next, x_value)
        theta_next = jnp.where(alive, theta_next, theta_value)
        return (x_next, theta_next, zeta_value + dz, alive_next, steps_alive + alive_next), None

    initial = (
        x0.ravel(),
        theta0.ravel(),
        zeta0.ravel(),
        jnp.ones(x0.size, dtype=bool),
        jnp.zeros(x0.size, dtype=jnp.int32),
    )
    (final_x, _, _, alive, steps_alive), _ = jax.lax.scan(
        step, initial, None, length=int(max_transits) * int(steps_per_transit)
    )
    del final_x
    connection_transits = steps_alive.astype(jnp.float64) / float(steps_per_transit)
    return connection_transits.reshape(shape), (~alive).reshape(shape)


def build_island_divertor_geometry(
    shape: tuple[int, int, int],
    *,
    field: IslandDivertorField = IslandDivertorField(),
    r0: float = 3.0,
    elongation: float = 0.35,
    n_field_periods: int = 1,
    c_phi: float = 3.0,
    construct_fci_maps: bool = False,
    map_substeps: int = 8,
    open_field_line_masks: bool = False,
    mask_max_transits: int = 30,
) -> FciGeometry3D:
    """Build the island-divertor geometry on the rotating-ellipse embedding.

    With ``open_field_line_masks=True`` the endpoint masks come from
    multi-transit field-line tracing (:func:`island_divertor_connection_length`):
    a cell is an open endpoint if its field line leaves the radial domain within
    ``mask_max_transits`` toroidal transits — the island-divertor scrape-off
    layer *emerges from the field itself* (stochastic-edge lines are open with
    finite connection length, core lines stay closed). ``construct_fci_maps``
    additionally traces the one-cell FCI parallel maps for the operators.
    """

    nx, ny, nz = shape
    x_faces = jnp.linspace(field.x_min, field.x_max, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    zeta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    grid = CellCenteredGrid3D(
        x=Grid1D(centers=0.5 * (x_faces[:-1] + x_faces[1:]), faces=x_faces),
        y=Grid1D(centers=0.5 * (theta_faces[:-1] + theta_faces[1:]), faces=theta_faces),
        z=Grid1D(centers=0.5 * (zeta_faces[:-1] + zeta_faces[1:]), faces=zeta_faces),
    )
    target_shape = grid.shape

    def _position(u: jax.Array) -> jax.Array:
        return rotating_ellipse_position(
            u[0], u[1], u[2], r0=r0, elongation=elongation, n_field_periods=n_field_periods
        )

    def _metric(logical_grid: jax.Array) -> MetricGeometry:
        return metric_from_position_fn(_position, logical_grid)

    def _bfield(logical_grid: jax.Array, metric: MetricGeometry) -> BFieldGeometry:
        x = logical_grid[..., 0]
        theta = logical_grid[..., 1]
        zeta = logical_grid[..., 2]
        toroidal = float(c_phi) / metric.J
        B_contra = jnp.stack(
            (
                field.radial_perturbation(theta, zeta) * toroidal,
                field.iota(x) * toroidal,
                toroidal,
            ),
            axis=-1,
        )
        return BFieldGeometry(
            B_contra=B_contra,
            Bmag=_bmag_from_contravariant_components(B_contra, metric.g_cov),
        )

    def _logical(x_axis, y_axis, z_axis):
        return logical_grid_from_axis_vectors(x_axis, y_axis, z_axis)

    cell_logical = _logical(grid.x.centers, grid.y.centers, grid.z.centers)
    cell_metric = _metric(cell_logical)
    cell_bfield = _bfield(cell_logical, cell_metric)
    face_metric = FaceMetricGeometry(
        x=_metric(_logical(grid.x.faces, grid.y.centers, grid.z.centers)),
        y=_metric(_logical(grid.x.centers, grid.y.faces, grid.z.centers)),
        z=_metric(_logical(grid.x.centers, grid.y.centers, grid.z.faces)),
    )
    face_bfield = FaceBFieldGeometry(
        x=_bfield(_logical(grid.x.faces, grid.y.centers, grid.z.centers), face_metric.x),
        y=_bfield(_logical(grid.x.centers, grid.y.faces, grid.z.centers), face_metric.y),
        z=_bfield(_logical(grid.x.centers, grid.y.centers, grid.z.faces), face_metric.z),
    )

    if construct_fci_maps:
        map_fields = build_fci_maps_from_b_contravariant(
            grid,
            cell_bfield.B_contra,
            cell_bfield.Bmag,
            periodic_axes=(False, True, True),
            substeps=int(map_substeps),
        )
    else:
        ones = jnp.ones(target_shape, dtype=jnp.float64)
        zeros = jnp.zeros(target_shape, dtype=jnp.float64)
        map_fields = {
            "forward_x": zeros,
            "forward_y": zeros,
            "backward_x": zeros,
            "backward_y": zeros,
            "forward_endpoint_x": zeros,
            "forward_endpoint_y": zeros,
            "forward_endpoint_z": zeros,
            "backward_endpoint_x": zeros,
            "backward_endpoint_y": zeros,
            "backward_endpoint_z": zeros,
            "forward_length": ones,
            "backward_length": ones,
            "forward_boundary": zeros.astype(bool),
            "backward_boundary": zeros.astype(bool),
        }

    if open_field_line_masks:
        forward_transits, forward_open = island_divertor_connection_length(
            field, cell_logical[..., 0], cell_logical[..., 1], cell_logical[..., 2],
            direction=+1, max_transits=mask_max_transits,
        )
        backward_transits, backward_open = island_divertor_connection_length(
            field, cell_logical[..., 0], cell_logical[..., 1], cell_logical[..., 2],
            direction=-1, max_transits=mask_max_transits,
        )
        del forward_transits, backward_transits
        map_fields["forward_boundary"] = jnp.asarray(map_fields["forward_boundary"], dtype=bool) | forward_open
        map_fields["backward_boundary"] = jnp.asarray(map_fields["backward_boundary"], dtype=bool) | backward_open

    maps = FciMaps3D(**{name: map_fields[name] for name in (
        "forward_x", "forward_y", "backward_x", "backward_y",
        "forward_endpoint_x", "forward_endpoint_y", "forward_endpoint_z",
        "backward_endpoint_x", "backward_endpoint_y", "backward_endpoint_z",
        "forward_length", "backward_length", "forward_boundary", "backward_boundary",
    )})
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], target_shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], target_shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], target_shape),
    )
    return FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
    )
