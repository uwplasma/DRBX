"""Composite structured three-dimensional MMPDE solver.

This module contains a small, conservative solver for a globally coupled
structured mesh with topology ``D^2 x S^1``.  The first two logical axes are
non-periodic and the last axis is periodic.  It is deliberately independent
of magnetic geometry: callers may supply a projector for eta or boundary
constraints, and a periodic-image callback for the physical field-period
identification.

The objective is a dimensionless composite of a normalized frozen-monitor
Dirichlet edge regularizer and four cell-quality terms: metric alignment,
metric-volume equidistribution, a positive-volume barrier, and neighboring
log-volume smoothness.  The reference metric volume ``vbar`` and nodal
monitor are frozen for the solve/backtracking objective.  Candidate meshes
must also pass the hard positive-Jacobian check, and the initial descent trial
is limited by a scale-aware physical cell-edge cap before any projector is
called.  Component histories are returned for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


Array = np.ndarray
Monitor = Array | Callable[[Array], Array]
Projector = Callable[[Array], Array]
PeriodicImage = Callable[[Array, int], Array]
CandidateValidator = Callable[[Array], bool]


@dataclass(frozen=True)
class _AffinePeriodicTransform:
    """Validated affine image map, using row-vector point storage."""

    rotation: Array
    translation: Array

    def __call__(self, points: Array, turns: int) -> Array:
        if turns == 0:
            return np.asarray(points, dtype=float)
        if turns == 1:
            return np.asarray(points, dtype=float) @ self.rotation.T + self.translation
        rotation = np.linalg.matrix_power(self.rotation, turns)
        translation = np.zeros(3)
        for _ in range(abs(turns)):
            if turns > 0:
                translation = self.rotation @ translation + self.translation
            else:
                translation = self.rotation.T @ (translation - self.translation)
        return np.asarray(points, dtype=float) @ rotation.T + translation


@dataclass(frozen=True)
class MMPDEOptions:
    """Numerical and objective controls for :func:`solve_mmpde`.

    ``dirichlet_weight`` scales the normalized edge regularizer.  The other
    weights select the alignment, equidistribution, positive-volume barrier,
    and neighboring log-volume smoothness terms, respectively.  The cell
    monitor and initial mean metric volume used by these terms are frozen
    during each solve.  ``maximum_step_fraction`` caps the first trial's
    maximum free-node displacement relative to the initial median physical
    cell-edge length; ordinary backtracking remains active after that cap.
    """

    max_iterations: int = 500
    progress_interval: int = 0
    tolerance: float = 1.0e-8
    initial_step: float = 1.0
    backtracking_factor: float = 0.5
    minimum_step: float = 1.0e-12
    minimum_jacobian: float = 1.0e-12
    maximum_step_fraction: float = 0.25
    # The edge term is retained as a weak, normalized regularizer.  The
    # remaining terms are cell-quality objectives and are dimensionless.
    dirichlet_weight: float = 0.05
    alignment_weight: float = 0.2
    equidistribution_weight: float = 0.2
    jacobian_barrier_weight: float = 1.0e-3
    volume_smoothness_weight: float = 0.1
    barrier_power: float = 2.0


@dataclass
class MMPDEResult:
    """Output and diagnostics from a structured MMPDE solve.

    ``component_energy_history`` stores raw, unweighted component values for
    ``dirichlet``, ``alignment``, ``equidistribution``,
    ``jacobian_barrier``, ``volume_smoothness``, and ``total``.  It defaults
    to an empty mapping for compatibility with older result objects.
    """

    positions: Array
    converged: bool
    iterations: int
    energy_history: Array
    residual_history: Array
    minimum_jacobian_history: Array
    component_energy_history: dict[str, Array] = field(default_factory=dict)

    @property
    def max_free_node_update(self) -> float:
        """Final maximum free-node update, or zero for an empty history."""

        return float(self.residual_history[-1]) if self.residual_history.size else 0.0


@dataclass(frozen=True)
class _EdgeFamily:
    """Vectorized data for one structured edge family.

    Endpoint indices are flattened indices into the first three dimensions of
    the position array.  ``wraps`` is true only for the periodic eta seam;
    keeping the seam separate lets the regular families use ordinary indexed
    gathers and avoids per-edge Python objects.
    """

    p: Array
    q: Array
    monitor: Array
    coefficient: Array
    wraps: bool


@dataclass(frozen=True)
class _EdgeFamilies:
    """The four structured edge families used by the Dirichlet term."""

    u: _EdgeFamily
    v: _EdgeFamily
    eta: _EdgeFamily
    seam: _EdgeFamily

    def __iter__(self):
        return iter((self.u, self.v, self.eta, self.seam))


def _fail(message: str) -> None:
    raise ValueError(message)


def _validate_options(options: MMPDEOptions) -> None:
    if options.max_iterations < 0:
        _fail("max_iterations must be nonnegative")
    if (
        int(options.progress_interval) != options.progress_interval
        or options.progress_interval < 0
    ):
        _fail("progress_interval must be a nonnegative integer")
    if options.tolerance < 0 or options.initial_step <= 0:
        _fail("tolerance must be nonnegative and initial_step positive")
    if not 0 < options.backtracking_factor < 1:
        _fail("backtracking_factor must lie strictly between zero and one")
    if options.minimum_step <= 0 or options.minimum_jacobian <= 0:
        _fail("minimum_step and minimum_jacobian must be positive")
    if not np.isfinite(options.maximum_step_fraction) or options.maximum_step_fraction <= 0:
        _fail("maximum_step_fraction must be finite and positive")
    weights = (
        options.dirichlet_weight,
        options.alignment_weight,
        options.equidistribution_weight,
        options.jacobian_barrier_weight,
        options.volume_smoothness_weight,
    )
    if not all(np.isfinite(w) and w >= 0 for w in weights):
        _fail("MMPDE energy weights must be finite and nonnegative")
    if not np.isfinite(options.barrier_power) or options.barrier_power <= 0:
        _fail("barrier_power must be finite and positive")


def _validate_positions(positions: Array) -> Array:
    x = np.asarray(positions, dtype=float)
    if x.ndim != 4 or x.shape[-1] != 3:
        _fail("positions must have shape (nu, nv, neta, 3)")
    if min(x.shape[:3]) < 2:
        _fail("each logical axis needs at least two nodes")
    if not np.all(np.isfinite(x)):
        _fail("positions must be finite")
    return x.copy()


def _validate_axes(axes: tuple[Array, Array, Array] | None, shape: tuple[int, int, int]) -> tuple[Array, Array, Array]:
    if axes is None:
        u, v = (np.linspace(0.0, 1.0, n) for n in shape[:2])
        eta = np.linspace(0.0, 1.0, shape[2], endpoint=False)
        return u, v, eta
    if len(axes) != 3:
        _fail("logical_axes must contain three one-dimensional axes")
    out = tuple(np.asarray(a, dtype=float) for a in axes)
    for axis, n in zip(out, shape):
        if axis.ndim != 1 or axis.size != n or not np.all(np.isfinite(axis)):
            _fail("each logical axis must be finite and match its node count")
        if np.any(np.diff(axis) <= 0):
            _fail("logical axes must be strictly increasing")
    eta_spacing = np.diff(out[2])
    if not np.allclose(eta_spacing, eta_spacing[0], rtol=1e-10, atol=1e-12):
        _fail("the endpoint-exclusive periodic eta axis must be uniformly spaced")
    return out  # type: ignore[return-value]


def _default_periodic_image(points: Array, turns: int) -> Array:
    return np.asarray(points)


def _infer_periodic_transform(periodic_image: PeriodicImage) -> _AffinePeriodicTransform:
    """Infer and validate ``q_image = R @ q + t`` from the callback.

    Points are stored as rows, so the returned map evaluates as
    ``points @ R.T + t``.  The callback is checked at basis and off-axis probe
    points so nonlinear or non-rigid callbacks fail before mesh calculations.
    """

    origin = np.zeros(3)
    basis = np.eye(3)
    samples = np.vstack((origin, basis))
    values = np.asarray(periodic_image(samples, 1), dtype=float)
    if values.shape != samples.shape or not np.all(np.isfinite(values)):
        _fail("periodic_image must return finite points with the input shape")
    translation = values[0]
    rotation = (values[1:] - translation).T
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-8, atol=1e-10):
        _fail("periodic_image must be affine with an orthogonal linear part")
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0 or not np.isclose(determinant, 1.0, rtol=1e-8, atol=1e-10):
        _fail("periodic_image must be orientation-preserving (det(R)=+1)")

    probes = np.array(
        [[0.37, -0.21, 0.53], [-0.8, 0.41, 0.19], [1.2, 0.6, -0.7]],
        dtype=float,
    )
    probe_values = np.asarray(periodic_image(probes, 1), dtype=float)
    expected = probes @ rotation.T + translation
    if probe_values.shape != probes.shape or not np.all(np.isfinite(probe_values)):
        _fail("periodic_image must return finite points with the input shape")
    if not np.allclose(probe_values, expected, rtol=1e-8, atol=1e-10):
        _fail("periodic_image must be affine with a rigid, orientation-preserving linear part")
    return _AffinePeriodicTransform(rotation, translation)


def _validate_mask(fixed_mask: Array | None, shape: tuple[int, int, int]) -> Array:
    if fixed_mask is None:
        mask = np.zeros(shape, dtype=bool)
        mask[0, :, :] = mask[-1, :, :] = True
        mask[:, 0, :] = mask[:, -1, :] = True
        return mask
    mask = np.asarray(fixed_mask, dtype=bool)
    if mask.shape != shape:
        _fail("fixed_mask must have shape (nu, nv, neta)")
    return mask.copy()


def _monitor_nodes(monitor: Monitor, positions: Array) -> Array | None:
    if callable(monitor):
        values = np.asarray(monitor(positions), dtype=float)
        if values.shape != positions.shape[:-1] + (3, 3):
            _fail("callable monitor must return (..., 3, 3)")
    else:
        values = np.asarray(monitor, dtype=float)
        if values.shape == (3, 3):
            values = np.broadcast_to(values, positions.shape[:-1] + (3, 3)).copy()
        elif values.shape != positions.shape[:-1] + (3, 3):
            _fail("monitor must have shape (3, 3) or (nu, nv, neta, 3, 3)")
    if not np.all(np.isfinite(values)):
        _fail("monitor must be finite")
    if not np.allclose(values, np.swapaxes(values, -1, -2), rtol=1e-10, atol=1e-12):
        _fail("monitor must be symmetric")
    eigenvalues = np.linalg.eigvalsh(values.reshape(-1, 3, 3))
    if np.any(eigenvalues <= 0):
        _fail("monitor must be positive definite")
    return values


def _edge_data(
    positions: Array,
    axes: tuple[Array, Array, Array],
    monitor_nodes: Array | None,
    monitor: Monitor,
    periodic_image: PeriodicImage,
    periodic_rotation: Array,
) -> _EdgeFamilies:
    """Build vectorized edge-family data for the Dirichlet term.

    Nodal monitors have already been checked by :func:`_monitor_nodes`, so
    their arithmetic edge averages are SPD without another eigendecomposition.
    A callable monitor is evaluated once per family and validated as one
    batch.  The monitor and coefficients are frozen by the caller for the
    duration of an objective/backtracking evaluation.
    """

    shape = positions.shape[:3]
    nu, nv, neta = shape
    du, dv, deta = (np.diff(axis) for axis in axes)
    # The two non-periodic edge families have one edge per eta node.  At the
    # last eta node their transverse logical measure uses the final periodic
    # spacing, matching the scalar implementation's ``min(k, neta-2)`` rule.
    u_node_spacing = np.concatenate((du, du[-1:]))
    v_node_spacing = np.concatenate((dv, dv[-1:]))
    eta_node_spacing = np.concatenate((deta, deta[-1:]))
    flat_indices = np.arange(nu * nv * neta, dtype=np.intp).reshape(shape)

    def monitor_at(midpoints: Array) -> Array:
        values = np.asarray(monitor(midpoints), dtype=float)
        expected = (midpoints.shape[0], 3, 3)
        if values.shape != expected or not np.all(np.isfinite(values)):
            _fail("monitor callback returned an invalid matrix")
        if not np.allclose(values, np.swapaxes(values, -1, -2), rtol=1e-10, atol=1e-12):
            _fail("monitor callback must return symmetric matrices")
        if np.any(np.linalg.eigvalsh(values) <= 0.0):
            _fail("monitor callback must return SPD matrices")
        return values

    def family(
        p: Array,
        q: Array,
        coefficients: Array,
        wraps: bool = False,
    ) -> _EdgeFamily:
        p = np.asarray(p, dtype=np.intp).reshape(-1)
        q = np.asarray(q, dtype=np.intp).reshape(-1)
        coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        if monitor_nodes is None:
            p_points = positions.reshape(-1, 3)[p]
            q_points = positions.reshape(-1, 3)[q]
            if wraps:
                q_points = np.asarray(periodic_image(q_points, 1), dtype=float)
            monitors = monitor_at(0.5 * (p_points + q_points))
        else:
            nodal = monitor_nodes.reshape(-1, 3, 3)
            p_monitor = nodal[p]
            q_monitor = nodal[q]
            if wraps:
                q_monitor = np.einsum(
                    "ab,nbc,cd->nad", periodic_rotation, q_monitor, periodic_rotation.T
                )
            monitors = 0.5 * (p_monitor + q_monitor)
        return _EdgeFamily(p, q, monitors, coefficients, wraps)

    # Each coefficient is the transverse logical measure divided by the
    # edge's logical length.  The eta axis is periodic and validated to be
    # uniform, but retain its final spacing for the seam exactly as before.
    u = family(
        flat_indices[:-1, :, :], flat_indices[1:, :, :],
        (v_node_spacing[None, :, None] * eta_node_spacing[None, None, :]) / du[:, None, None],
    )
    v = family(
        flat_indices[:, :-1, :], flat_indices[:, 1:, :],
        (u_node_spacing[:, None, None] * eta_node_spacing[None, None, :]) / dv[None, :, None],
    )
    eta = family(
        flat_indices[:, :, :-1], flat_indices[:, :, 1:],
        (u_node_spacing[:, None, None] * v_node_spacing[None, :, None])
        / deta[None, None, :],
    )
    seam = family(
        flat_indices[:, :, -1], flat_indices[:, :, 0],
        (u_node_spacing[:, None] * v_node_spacing[None, :]) / deta[-1], wraps=True,
    )
    return _EdgeFamilies(u=u, v=v, eta=eta, seam=seam)


def _energy_gradient(
    positions: Array,
    edges: _EdgeFamilies,
    fixed: Array,
    periodic_image: PeriodicImage,
    periodic_rotation: Array,
) -> tuple[float, Array]:
    flat_positions = positions.reshape(-1, 3)
    gradient = np.zeros_like(flat_positions)
    energy = 0.0
    for family in edges:
        p = flat_positions[family.p]
        q = flat_positions[family.q]
        if family.wraps:
            q = np.asarray(periodic_image(q, 1), dtype=float)
        difference = q - p
        edge_gradient = family.coefficient[:, None] * np.einsum(
            "nij,nj->ni", family.monitor, difference
        )
        energy += 0.5 * np.sum(
            family.coefficient * np.einsum("ni,nij,nj->n", difference, family.monitor, difference)
        )
        np.add.at(gradient, family.p, -edge_gradient)
        if family.wraps:
            edge_gradient = edge_gradient @ periodic_rotation
        np.add.at(gradient, family.q, edge_gradient)
    gradient[fixed.reshape(-1)] = 0.0
    return float(energy), gradient.reshape(positions.shape)


def _cell_data(
    positions: Array,
    axes: tuple[Array, Array, Array],
    monitor_nodes: Array,
    periodic_image: PeriodicImage,
    periodic_rotation: Array,
) -> tuple[list[tuple[int, int, int]], Array, Array]:
    """Return cell indices, physical edge matrices, and frozen cell monitors.

    The columns of each ``F`` are physical edges, rather than derivatives
    divided by logical cell widths.  At the eta seam the third edge ends at
    the rotated image of the first layer.  The cell monitor is the arithmetic
    mean of the eight nodal monitors, with the seam layer pulled into the
    current image before averaging.
    """

    nu, nv, neta = positions.shape[:3]
    cells = list(np.ndindex(nu - 1, nv - 1, neta))
    p = positions[:-1, :-1]
    q0 = positions[1:, :-1]
    q1 = positions[:-1, 1:]
    seam_points = np.asarray(
        periodic_image(positions[:-1, :-1, 0].reshape(-1, 3), 1), dtype=float
    ).reshape(nu - 1, nv - 1, 3)
    q2 = np.concatenate((positions[:-1, :-1, 1:], seam_points[..., None, :]), axis=2)
    matrices = np.stack((q0 - p, q1 - p, q2 - p), axis=-1).reshape(-1, 3, 3)

    rotated_first = np.einsum(
        "ab,ijbc,cd->ijad", periodic_rotation, monitor_nodes[:, :, 0], periodic_rotation.T
    )
    next_monitor = np.concatenate(
        (monitor_nodes[:, :, 1:], rotated_first[..., None, :, :]), axis=2
    )
    monitors_grid = (
        monitor_nodes[:-1, :-1]
        + monitor_nodes[1:, :-1]
        + monitor_nodes[:-1, 1:]
        + monitor_nodes[1:, 1:]
        + next_monitor[:-1, :-1]
        + next_monitor[1:, :-1]
        + next_monitor[:-1, 1:]
        + next_monitor[1:, 1:]
    ) / 8.0
    return cells, matrices, monitors_grid.reshape(-1, 3, 3)


def _cell_quality_energy_gradient(
    positions: Array,
    axes: tuple[Array, Array, Array],
    fixed: Array,
    monitor_nodes: Array,
    periodic_image: PeriodicImage,
    periodic_rotation: Array,
    options: MMPDEOptions,
    initial_vbar: float,
    initial_dirichlet_energy: float,
    frozen_edges: _EdgeFamilies,
) -> tuple[float, Array, dict[str, float]]:
    """Evaluate the frozen-monitor composite cell objective and its gradient."""

    cells, matrices, monitors = _cell_data(
        positions, axes, monitor_nodes, periodic_image, periodic_rotation
    )
    ncell = len(cells)
    gradient = np.zeros_like(positions)
    edge_energy, edge_gradient = _energy_gradient(
        positions, frozen_edges, fixed, periodic_image, periodic_rotation
    )
    dirichlet_scale = max(float(initial_dirichlet_energy), np.finfo(float).tiny)
    components = {
        "dirichlet": edge_energy / dirichlet_scale,
        "alignment": 0.0,
        "equidistribution": 0.0,
        "jacobian_barrier": 0.0,
        "volume_smoothness": 0.0,
    }
    gradient += options.dirichlet_weight * edge_gradient / dirichlet_scale
    if ncell == 0:
        return 0.0, gradient, components

    detF = np.linalg.det(matrices)
    detM = np.linalg.det(monitors)
    if np.any(detF <= 0.0) or np.any(detM <= 0.0) or not np.all(np.isfinite(detF * detM)):
        return np.inf, np.zeros_like(positions), {**components, "total": np.inf}
    C = np.einsum("nai,nab,nbj->nij", matrices, monitors, matrices)
    detC = np.linalg.det(C)
    if np.any(detC <= 0.0) or not np.all(np.isfinite(detC)):
        return np.inf, np.zeros_like(positions), {**components, "total": np.inf}
    inverse_transposes = np.transpose(np.linalg.inv(matrices), (0, 2, 1))
    volumes = detF * np.sqrt(detM)
    detC13 = detC ** (1.0 / 3.0)
    ratios = np.trace(C, axis1=1, axis2=2) / (3.0 * detC13)
    residuals = ratios - 1.0
    components["alignment"] = float(np.mean(residuals * residuals))
    d_ratio_dC = np.eye(3)[None, :, :] / (3.0 * detC13[:, None, None]) - (
        np.trace(C, axis1=1, axis2=2)[:, None, None]
        * np.transpose(np.linalg.inv(C), (0, 2, 1))
        / (9.0 * detC13[:, None, None])
    )
    align_gradients = (4.0 / ncell) * residuals[:, None, None] * np.einsum(
        "nab,nbj,njk->nak", monitors, matrices, d_ratio_dC
    )

    log_ratio = np.log(volumes / initial_vbar)
    components["equidistribution"] = float(np.mean(log_ratio * log_ratio))
    barrier_values = (initial_vbar / volumes) ** options.barrier_power
    components["jacobian_barrier"] = float(np.mean(barrier_values))

    # Build the neighbor derivative in log-volume space first.  This includes
    # the eta seam by rolling each (u,v) cell plane.
    cell_shape = (positions.shape[0] - 1, positions.shape[1] - 1, positions.shape[2])
    log_volume = log_ratio.reshape(cell_shape)
    log_coefficients = np.zeros(cell_shape, dtype=float)
    u_difference = log_volume[:-1] - log_volume[1:]
    v_difference = log_volume[:, :-1] - log_volume[:, 1:]
    eta_difference = log_volume - np.roll(log_volume, -1, axis=2)
    log_coefficients[:-1] += 2.0 * u_difference
    log_coefficients[1:] -= 2.0 * u_difference
    log_coefficients[:, :-1] += 2.0 * v_difference
    log_coefficients[:, 1:] -= 2.0 * v_difference
    log_coefficients += 2.0 * eta_difference
    log_coefficients -= 2.0 * np.roll(eta_difference, 1, axis=2)
    pair_count = u_difference.size + v_difference.size + eta_difference.size
    if pair_count:
        components["volume_smoothness"] = float(
            (
                np.sum(u_difference * u_difference)
                + np.sum(v_difference * v_difference)
                + np.sum(eta_difference * eta_difference)
            )
            / pair_count
        )
        log_coefficients /= pair_count

    cell_gradients = options.alignment_weight * align_gradients
    cell_gradients += (
        options.equidistribution_weight
        * (2.0 / ncell)
        * log_ratio[:, None, None]
        * inverse_transposes
    )
    cell_gradients += (
        options.jacobian_barrier_weight
        * (-options.barrier_power / ncell * barrier_values[:, None, None])
        * inverse_transposes
    )
    cell_gradients += (
        options.volume_smoothness_weight
        * log_coefficients.reshape(-1, 1, 1)
        * inverse_transposes
    )
    cell_gradients = cell_gradients.reshape(*cell_shape, 3, 3)
    gradient[:-1, :-1] -= np.sum(cell_gradients, axis=-1)
    gradient[1:, :-1] += cell_gradients[..., :, 0]
    gradient[:-1, 1:] += cell_gradients[..., :, 1]
    gradient[:-1, :-1, 1:] += cell_gradients[:, :, :-1, :, 2]
    gradient[:-1, :-1, 0] += np.einsum(
        "ab,ijb->ija", periodic_rotation.T, cell_gradients[:, :, -1, :, 2]
    )

    gradient[fixed] = 0.0
    components = {name: float(value) for name, value in components.items()}
    total = sum(
        getattr(options, f"{name}_weight") * value
        for name, value in components.items()
        if name != "dirichlet" and name != "total"
    ) + options.dirichlet_weight * components["dirichlet"]
    components["total"] = float(total)
    return float(total), gradient, components


def _cell_jacobians(positions: Array, axes: tuple[Array, Array, Array], periodic_image: PeriodicImage) -> Array:
    nu, nv, neta = positions.shape[:3]
    du, dv = np.diff(axes[0]), np.diff(axes[1])
    p = positions[:-1, :-1]
    q0 = positions[1:, :-1]
    q1 = positions[:-1, 1:]
    seam_points = np.asarray(
        periodic_image(positions[:-1, :-1, 0].reshape(-1, 3), 1), dtype=float
    ).reshape(nu - 1, nv - 1, 3)
    q2 = np.concatenate((positions[:-1, :-1, 1:], seam_points[..., None, :]), axis=2)
    edges = np.stack((q0 - p, q1 - p, q2 - p), axis=-1)
    edges[..., 0] /= du[:, None, None, None]
    edges[..., 1] /= dv[None, :, None, None]
    edges[..., 2] /= axes[2][-1] - axes[2][-2]
    return np.linalg.det(edges)


def solve_mmpde(
    positions: Array,
    *,
    logical_axes: tuple[Array, Array, Array] | None = None,
    monitor: Monitor | None = None,
    fixed_mask: Array | None = None,
    projector: Projector | None = None,
    periodic_image: PeriodicImage | None = None,
    options: MMPDEOptions | None = None,
    candidate_validator: CandidateValidator | None = None,
) -> MMPDEResult:
    """Relax a globally coupled structured ``D^2 x S^1`` mesh.

    ``positions`` has shape ``(nu, nv, neta, 3)``.  Axes 0 and 1 are
    non-periodic; axis 2 has one wrap edge.  ``periodic_image(points, 1)``
    maps the first eta layer into the physical image after one field period.
    A projector is called on every candidate iterate; fixed nodes are then
    restored exactly.  If ``candidate_validator`` is provided, it is called
    on the initial mesh after its cell-Jacobian check and on every candidate
    after projection, fixed-node restoration, and its cell-Jacobian check,
    immediately before energy evaluation.  It must return a scalar Python
    ``bool`` or NumPy ``bool_``.  ``False`` rejects a candidate and causes
    backtracking; a false initial result raises ``ValueError``.  Exceptions
    raised by the callback propagate unchanged.  Any other return value
    raises ``TypeError``.  The normalized Dirichlet edge term is retained as
    a weak regularizer alongside cell alignment, equidistribution,
    positive-volume barrier, and neighboring log-volume smoothness
    objectives.  The initial mean metric volume and each iteration's nodal
    monitor are frozen during that iteration's line search.  A hard
    positive-Jacobian check is always applied.  The initial trial is capped
    by ``maximum_step_fraction`` times the initial median physical cell-edge
    length before the projector is called.  The monitor is evaluated once per
    iteration and held fixed while backtracking evaluates that iteration's
    candidates.
    """

    opts = options or MMPDEOptions()
    _validate_options(opts)
    x = _validate_positions(positions)
    axes = _validate_axes(logical_axes, x.shape[:3])
    fixed = _validate_mask(fixed_mask, x.shape[:3])
    image = periodic_image or _default_periodic_image
    affine_image = _infer_periodic_transform(image)
    if monitor is None:
        monitor = np.eye(3)
    nodes_monitor = _monitor_nodes(monitor, x)
    initial_jac = _cell_jacobians(x, axes, affine_image)
    if np.any(initial_jac <= opts.minimum_jacobian):
        _fail("initial mesh has a cell Jacobian below minimum_jacobian")
    if candidate_validator is not None:
        initial_valid = candidate_validator(x.copy())
        if not isinstance(initial_valid, (bool, np.bool_)):
            raise TypeError("candidate_validator must return a bool")
        if not bool(initial_valid):
            _fail("initial mesh rejected by candidate_validator")
    initial_edges = _edge_data(x, axes, nodes_monitor, monitor, affine_image, affine_image.rotation)
    initial_dirichlet_energy, _ = _energy_gradient(
        x, initial_edges, fixed, affine_image, affine_image.rotation
    )
    _, initial_matrices, initial_monitors = _cell_data(
        x, axes, nodes_monitor, affine_image, affine_image.rotation
    )
    if initial_matrices.size == 0:
        _fail("mesh must contain at least one structured cell")
    initial_vbar = float(
        np.mean(np.linalg.det(initial_matrices) * np.sqrt(np.linalg.det(initial_monitors)))
    )
    characteristic_cell_edge_length = float(
        np.median(np.linalg.norm(initial_matrices, axis=2))
    )
    if not np.isfinite(characteristic_cell_edge_length) or characteristic_cell_edge_length <= 0:
        _fail("initial mesh has no finite positive characteristic cell-edge length")
    energy, _, initial_components = _cell_quality_energy_gradient(
        x,
        axes,
        fixed,
        nodes_monitor,
        affine_image,
        affine_image.rotation,
        opts,
        initial_vbar,
        initial_dirichlet_energy,
        initial_edges,
    )
    energies = [energy]
    component_histories = {
        name: [value] for name, value in initial_components.items() if name != "total"
    }
    residuals: list[float] = []
    min_jacobians = [float(np.min(initial_jac))]
    converged = False
    iterations = 0
    progress_interval = int(opts.progress_interval)
    if progress_interval > 0:
        print(
            "[MMPDE] starting: "
            f"nodes={x.shape[:3]}, max_iterations={opts.max_iterations}, "
            f"energy={energy:.6e}, min_jacobian={min_jacobians[-1]:.6e}",
            flush=True,
        )
    for iteration in range(opts.max_iterations):
        # A callable monitor is sampled at the start of the iteration and is
        # then frozen for both the gradient and all backtracking candidates.
        iteration_monitor = _monitor_nodes(monitor, x) if callable(monitor) else nodes_monitor
        frozen_edges = (
            _edge_data(x, axes, iteration_monitor, monitor, affine_image, affine_image.rotation)
            if callable(monitor)
            else initial_edges
        )
        current_energy, gradient, current_components = _cell_quality_energy_gradient(
            x,
            axes,
            fixed,
            iteration_monitor,
            affine_image,
            affine_image.rotation,
            opts,
            initial_vbar,
            initial_dirichlet_energy,
            frozen_edges,
        )
        free_gradient = gradient[~fixed]
        norm = float(np.max(np.linalg.norm(free_gradient, axis=-1))) if free_gradient.size else 0.0
        if norm <= opts.tolerance:
            converged = True
            if progress_interval > 0:
                print(
                    f"[MMPDE] converged before iteration {iteration + 1}: "
                    f"gradient_norm={norm:.6e}",
                    flush=True,
                )
            break
        # The composite gradient can have a large magnitude when a cell is
        # close to degeneracy.  Limit the first trial before invoking the
        # projector, whose valid coordinate domain may be bounded.  Subsequent
        # trials still use ordinary backtracking from this scale-aware step.
        gradient_displacement = (
            float(np.max(np.linalg.norm(free_gradient, axis=-1))) if free_gradient.size else 0.0
        )
        if gradient_displacement > 0.0:
            maximum_step = (
                opts.maximum_step_fraction * characteristic_cell_edge_length
                / gradient_displacement
            )
        else:
            maximum_step = opts.initial_step
        step = min(opts.initial_step, maximum_step)
        accepted = False
        while step >= opts.minimum_step:
            candidate = x - step * gradient
            if projector is not None:
                candidate = np.asarray(projector(candidate.copy()), dtype=float)
                if candidate.shape != x.shape or not np.all(np.isfinite(candidate)):
                    _fail("projector must return finite positions with the input shape")
            candidate[fixed] = x[fixed]
            candidate_jac = _cell_jacobians(candidate, axes, affine_image)
            if np.all(candidate_jac > opts.minimum_jacobian):
                if candidate_validator is not None:
                    candidate_valid = candidate_validator(candidate.copy())
                    if not isinstance(candidate_valid, (bool, np.bool_)):
                        raise TypeError("candidate_validator must return a bool")
                    if not bool(candidate_valid):
                        step *= opts.backtracking_factor
                        continue
                candidate_energy, _, candidate_components = _cell_quality_energy_gradient(
                    candidate,
                    axes,
                    fixed,
                    iteration_monitor,
                    affine_image,
                    affine_image.rotation,
                    opts,
                    initial_vbar,
                    initial_dirichlet_energy,
                    frozen_edges,
                )
                if np.isfinite(candidate_energy) and candidate_energy < current_energy:
                    accepted = True
                    break
            step *= opts.backtracking_factor
        if not accepted:
            if progress_interval > 0:
                print(
                    f"[MMPDE] stopped at iteration {iteration + 1}: "
                    "line search found no acceptable update",
                    flush=True,
                )
            break
        update = float(np.max(np.linalg.norm(candidate[~fixed] - x[~fixed], axis=-1))) if np.any(~fixed) else 0.0
        x = candidate
        iterations = iteration + 1
        energies.append(candidate_energy)
        for name in component_histories:
            component_histories[name].append(candidate_components[name])
        residuals.append(update)
        min_jacobians.append(float(np.min(candidate_jac)))
        if (
            progress_interval > 0
            and (
                iterations % progress_interval == 0
                or iterations == opts.max_iterations
            )
        ):
            print(
                f"[MMPDE] iteration {iterations}/{opts.max_iterations}: "
                f"energy={candidate_energy:.6e}, "
                f"max_update={update:.6e}, "
                f"min_jacobian={min_jacobians[-1]:.6e}",
                flush=True,
            )
        if update <= opts.tolerance:
            converged = True
            break
    if progress_interval > 0:
        print(
            f"[MMPDE] finished: iterations={iterations}, "
            f"converged={converged}, energy={energies[-1]:.6e}, "
            f"min_jacobian={min_jacobians[-1]:.6e}",
            flush=True,
        )
    return MMPDEResult(
        positions=x,
        converged=converged,
        iterations=iterations,
        energy_history=np.asarray(energies),
        residual_history=np.asarray(residuals),
        minimum_jacobian_history=np.asarray(min_jacobians),
        component_energy_history={
            **{name: np.asarray(values) for name, values in component_histories.items()},
            "total": np.asarray(energies),
        },
    )


__all__ = ["CandidateValidator", "MMPDEOptions", "MMPDEResult", "solve_mmpde"]
