"""Generic structured three-dimensional MMPDE solver.

This module contains a small, conservative baseline for a globally coupled
structured mesh with topology ``D^2 x S^1``.  The first two logical axes are
non-periodic and the last axis is periodic.  It is deliberately independent
of magnetic geometry: callers may supply a projector for eta or boundary
constraints, and a periodic-image callback for the physical field-period
identification.

The baseline minimizes a frozen-monitor weighted Dirichlet (Winslow-type)
energy on all structured edges.  It is intended as reliable infrastructure,
not as a replacement for a higher-order production MMPDE functional.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Numerical controls for :func:`solve_mmpde`."""

    max_iterations: int = 500
    tolerance: float = 1.0e-8
    initial_step: float = 1.0
    backtracking_factor: float = 0.5
    minimum_step: float = 1.0e-12
    minimum_jacobian: float = 1.0e-12


@dataclass
class MMPDEResult:
    """Output and diagnostics from a structured MMPDE solve."""

    positions: Array
    converged: bool
    iterations: int
    energy_history: Array
    residual_history: Array
    minimum_jacobian_history: Array

    @property
    def max_free_node_update(self) -> float:
        """Final maximum free-node update, or zero for an empty history."""

        return float(self.residual_history[-1]) if self.residual_history.size else 0.0


def _fail(message: str) -> None:
    raise ValueError(message)


def _validate_options(options: MMPDEOptions) -> None:
    if options.max_iterations < 0:
        _fail("max_iterations must be nonnegative")
    if options.tolerance < 0 or options.initial_step <= 0:
        _fail("tolerance must be nonnegative and initial_step positive")
    if not 0 < options.backtracking_factor < 1:
        _fail("backtracking_factor must lie strictly between zero and one")
    if options.minimum_step <= 0 or options.minimum_jacobian <= 0:
        _fail("minimum_step and minimum_jacobian must be positive")


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
) -> list[tuple[tuple[tuple[int, int, int], tuple[int, int, int]], Array, Array, float, bool]]:
    """Return edge endpoint indices, physical difference, monitor, and weight."""

    edges = []
    shape = positions.shape[:3]
    spacings = [np.diff(axis) for axis in axes]
    for axis in range(3):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    index = (i, j, k)
                    if axis < 2:
                        if index[axis] >= shape[axis] - 1:
                            continue
                        other = list(index)
                        other[axis] += 1
                        other = tuple(other)
                        h = spacings[axis][index[axis]]
                        q = positions[other]
                        m = 0.5 * (positions[index] + q)
                        if monitor_nodes is None:
                            mm = np.asarray(monitor(m[None, :])[0], dtype=float)
                        else:
                            mm = 0.5 * (monitor_nodes[index] + monitor_nodes[other])
                        weight = np.prod(
                            [spacings[a][min(index[a], len(spacings[a]) - 1)] for a in range(3) if a != axis]
                        )
                    else:
                        if index[axis] < shape[axis] - 1:
                            other = (i, j, k + 1)
                            h = spacings[2][k]
                            q = positions[other]
                            m = 0.5 * (positions[index] + q)
                            if monitor_nodes is None:
                                mm = np.asarray(monitor(m[None, :])[0], dtype=float)
                            else:
                                mm = 0.5 * (monitor_nodes[index] + monitor_nodes[other])
                        else:
                            other = (i, j, 0)
                            h = axes[2][-1] - axes[2][-2]
                            q = np.asarray(periodic_image(positions[other][None, :], 1), dtype=float)[0]
                            m = 0.5 * (positions[index] + q)
                            if monitor_nodes is None:
                                mm = np.asarray(monitor(m[None, :])[0], dtype=float)
                            else:
                                other_monitor = periodic_rotation @ monitor_nodes[other] @ periodic_rotation.T
                                mm = 0.5 * (monitor_nodes[index] + other_monitor)
                        weight = spacings[0][min(i, len(spacings[0]) - 1)] * spacings[1][min(j, len(spacings[1]) - 1)]
                    if mm.shape != (3, 3) or not np.all(np.isfinite(mm)):
                        _fail("monitor callback returned an invalid matrix")
                    if np.any(np.linalg.eigvalsh(0.5 * (mm + mm.T)) <= 0):
                        _fail("monitor callback must return SPD matrices")
                    edges.append(((index, other), np.asarray(q - positions[index]), mm, float(weight / h), axis == 2 and index[2] == shape[2] - 1))
    return edges


def _energy_gradient(
    positions: Array,
    edges: list,
    fixed: Array,
    periodic_image: PeriodicImage,
    periodic_rotation: Array,
) -> tuple[float, Array]:
    gradient = np.zeros_like(positions)
    energy = 0.0
    for (pidx, qidx), _, monitor, coefficient, wraps in edges:
        if wraps:
            q = np.asarray(periodic_image(positions[qidx][None, :], 1), dtype=float)[0]
        else:
            q = positions[qidx]
        difference = q - positions[pidx]
        edge_gradient = coefficient * monitor @ difference
        energy += 0.5 * coefficient * float(difference @ monitor @ difference)
        gradient[pidx] -= edge_gradient
        gradient[qidx] += periodic_rotation.T @ edge_gradient if wraps else edge_gradient
    gradient[fixed] = 0.0
    return energy, gradient


def _cell_jacobians(positions: Array, axes: tuple[Array, Array, Array], periodic_image: PeriodicImage) -> Array:
    nu, nv, neta = positions.shape[:3]
    jac = np.empty((nu - 1, nv - 1, neta), dtype=float)
    du, dv = np.diff(axes[0]), np.diff(axes[1])
    for i in range(nu - 1):
        for j in range(nv - 1):
            for k in range(neta):
                p = positions[i, j, k]
                a = (positions[i + 1, j, k] - p) / du[i]
                b = (positions[i, j + 1, k] - p) / dv[j]
                if k + 1 < neta:
                    q = positions[i, j, k + 1]
                else:
                    q = np.asarray(periodic_image(positions[i, j, 0][None, :], 1), dtype=float)[0]
                c = (q - p) / (axes[2][-1] - axes[2][-2])
                jac[i, j, k] = np.linalg.det(np.column_stack((a, b, c)))
    return jac


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
    raises ``TypeError``.  The monitor is evaluated once per iteration and
    held fixed while backtracking evaluates that iteration's candidates.
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
    energy, _ = _energy_gradient(x, initial_edges, fixed, affine_image, affine_image.rotation)
    energies = [energy]
    residuals: list[float] = []
    min_jacobians = [float(np.min(initial_jac))]
    converged = False
    iterations = 0
    for iteration in range(opts.max_iterations):
        # A callable monitor is sampled at the start of the iteration and is
        # then frozen for both the gradient and all backtracking candidates.
        iteration_monitor = _monitor_nodes(monitor, x) if callable(monitor) else nodes_monitor
        frozen_edges = _edge_data(x, axes, iteration_monitor, monitor, affine_image, affine_image.rotation)
        current_energy, gradient = _energy_gradient(
            x, frozen_edges, fixed, affine_image, affine_image.rotation
        )
        free_gradient = gradient[~fixed]
        norm = float(np.max(np.linalg.norm(free_gradient, axis=-1))) if free_gradient.size else 0.0
        if norm <= opts.tolerance:
            converged = True
            break
        step = opts.initial_step
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
                candidate_energy, _ = _energy_gradient(
                    candidate, frozen_edges, fixed, affine_image, affine_image.rotation
                )
                if candidate_energy < current_energy:
                    accepted = True
                    break
            step *= opts.backtracking_factor
        if not accepted:
            break
        update = float(np.max(np.linalg.norm(candidate[~fixed] - x[~fixed], axis=-1))) if np.any(~fixed) else 0.0
        x = candidate
        iterations = iteration + 1
        energies.append(candidate_energy)
        residuals.append(update)
        min_jacobians.append(float(np.min(candidate_jac)))
        if update <= opts.tolerance:
            converged = True
            break
    return MMPDEResult(
        positions=x,
        converged=converged,
        iterations=iterations,
        energy_history=np.asarray(energies),
        residual_history=np.asarray(residuals),
        minimum_jacobian_history=np.asarray(min_jacobians),
    )


__all__ = ["CandidateValidator", "MMPDEOptions", "MMPDEResult", "solve_mmpde"]
