"""Axis-regular Poisson-bracket experiment (no production integration).

This is a small NumPy/pytest prototype for the non-conservative Poisson
bracket at a polar axis.  It mirrors the existing Cartesian axis-core
infrastructure in ``drbx.geometry.fci_geometry``:

* observations are cell-centred values on a fixed number of inner radial
  rings;
* a triangular Cartesian monomial fit is factored once with a pseudoinverse;
* runtime evaluation is a matrix multiply, independently for every toroidal
  (``zeta``) slice; and
* Cartesian gradient target functionals replace the polar gradient only in the
  selected core rings.

The current branch is the centered polar-gradient formula

    {f, g} = (f_r g_theta - f_theta g_r) / r,

with the scalar half-turn continuation used at the lower radial ghost point.
The prototype intentionally does not import or modify a production operator:
it isolates the axis closure question and makes its assumptions explicit.

Run as a diagnostic with::

    python3 DRBX/tests/axis_regular_poisson_cartesian_core_experiment.py

or as tests with::

    python3 -m pytest -q DRBX/tests/axis_regular_poisson_cartesian_core_experiment.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest


Array = np.ndarray


def _basis(x: Array, y: Array, degree: int) -> Array:
    """Triangular Cartesian monomial basis, in the production ordering."""

    columns = []
    for total_degree in range(degree + 1):
        for x_degree in range(total_degree, -1, -1):
            y_degree = total_degree - x_degree
            columns.append(x**x_degree * y**y_degree)
    return np.stack(columns, axis=-1)


def _basis_gradient(x: Array, y: Array, degree: int) -> tuple[Array, Array]:
    """Cartesian derivatives of ``_basis`` at point arrays."""

    dx_columns = []
    dy_columns = []
    for total_degree in range(degree + 1):
        for x_degree in range(total_degree, -1, -1):
            y_degree = total_degree - x_degree
            dx_columns.append(
                x_degree * x ** max(x_degree - 1, 0) * y**y_degree
                if x_degree
                else np.zeros_like(x)
            )
            dy_columns.append(
                y_degree * x**x_degree * y ** max(y_degree - 1, 0)
                if y_degree
                else np.zeros_like(x)
            )
    return np.stack(dx_columns, axis=-1), np.stack(dy_columns, axis=-1)


@dataclass(frozen=True)
class PolarMesh:
    """Cell-centred polar mesh with a periodic toroidal coordinate."""

    r: Array
    theta: Array
    zeta: Array
    dr: float
    dtheta: float
    dzeta: float

    @property
    def x(self) -> Array:
        return self.r * np.cos(self.theta)

    @property
    def y(self) -> Array:
        return self.r * np.sin(self.theta)


def make_mesh(nx: int, ntheta: int, nzeta: int = 4) -> PolarMesh:
    """Construct ``(r, theta, zeta)`` cell centres on a unit polar disk."""

    dr = 1.0 / nx
    dtheta = 2.0 * np.pi / ntheta
    dzeta = 2.0 * np.pi / nzeta
    r1 = (np.arange(nx, dtype=float) + 0.5) * dr
    theta1 = (np.arange(ntheta, dtype=float) + 0.5) * dtheta
    zeta1 = np.arange(nzeta, dtype=float) * dzeta
    r, theta, zeta = np.meshgrid(r1, theta1, zeta1, indexing="ij")
    return PolarMesh(r=r, theta=theta, zeta=zeta, dr=dr, dtheta=dtheta, dzeta=dzeta)


@dataclass(frozen=True)
class ManufacturedFields:
    f: Array
    g: Array
    bracket: Array


def manufactured_fields(mesh: PolarMesh) -> ManufacturedFields:
    """Smooth Cartesian polynomials with a known bracket on every zeta slice."""

    x, y, z = mesh.x, mesh.y, mesh.zeta
    ax = 0.7 + 0.10 * np.cos(z)
    ay = -0.4 + 0.05 * np.sin(z)
    bx = 0.2 + 0.03 * np.sin(z)
    by = 0.9 + 0.04 * np.cos(z)

    f = 1.25 + ax * x + ay * y + 0.20 * x**2 + 0.15 * x * y - 0.10 * y**2
    g = -0.30 + bx * x + by * y - 0.25 * x**2 + 0.35 * x * y + 0.18 * y**2

    f_x = ax + 0.40 * x + 0.15 * y
    f_y = ay + 0.15 * x - 0.20 * y
    g_x = bx - 0.50 * x + 0.35 * y
    g_y = by + 0.35 * x + 0.36 * y
    bracket = f_x * g_y - f_y * g_x
    return ManufacturedFields(f=f, g=g, bracket=bracket)


def _evaluate_field(field: Callable[[Array, Array, Array], Array], r: Array, theta: Array, zeta: Array) -> Array:
    """Evaluate a field on possibly signed-radius polar coordinates."""

    return np.asarray(field(r * np.cos(theta), r * np.sin(theta), zeta), dtype=float)


def _field_functions() -> tuple[Callable[[Array, Array, Array], Array], Callable[[Array, Array, Array], Array]]:
    """Return the manufactured fields as Cartesian-coordinate callables."""

    def f(x: Array, y: Array, z: Array) -> Array:
        ax = 0.7 + 0.10 * np.cos(z)
        ay = -0.4 + 0.05 * np.sin(z)
        return 1.25 + ax * x + ay * y + 0.20 * x**2 + 0.15 * x * y - 0.10 * y**2

    def g(x: Array, y: Array, z: Array) -> Array:
        bx = 0.2 + 0.03 * np.sin(z)
        by = 0.9 + 0.04 * np.cos(z)
        return -0.30 + bx * x + by * y - 0.25 * x**2 + 0.35 * x * y + 0.18 * y**2

    return f, g


def centered_polar_bracket_from_callables(
    f_callable: Callable[[Array, Array, Array], Array],
    g_callable: Callable[[Array, Array, Array], Array],
    mesh: PolarMesh,
) -> Array:
    """Evaluate the centered polar bracket, including its axis ghost rule."""

    f = _evaluate_field(f_callable, mesh.r, mesh.theta, mesh.zeta)
    g = _evaluate_field(g_callable, mesh.r, mesh.theta, mesh.zeta)

    def gradients(callable_field: Callable[[Array, Array, Array], Array], values: Array) -> tuple[Array, Array]:
        lower = _evaluate_field(
            callable_field,
            -0.5 * mesh.dr * np.ones_like(mesh.theta[0]),
            mesh.theta[0],
            mesh.zeta[0],
        )
        upper = _evaluate_field(
            callable_field,
            (mesh.r.shape[0] + 0.5) * mesh.dr * np.ones_like(mesh.theta[0]),
            mesh.theta[-1],
            mesh.zeta[-1],
        )
        extended = np.concatenate((lower[None, ...], values, upper[None, ...]), axis=0)
        radial = (extended[2:] - extended[:-2]) / (2.0 * mesh.dr)
        angular = (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * mesh.dtheta)
        return radial, angular

    f_r, f_theta = gradients(f_callable, f)
    g_r, g_theta = gradients(g_callable, g)
    return (f_r * g_theta - f_theta * g_r) / mesh.r


@dataclass(frozen=True)
class CartesianCoreGradientWeights:
    """Precomputed observation-to-gradient weights for one mesh shape."""

    degree: int
    observation_rings: int
    gradient_x: Array  # (target_ring, theta, observation,)
    gradient_y: Array


def build_cartesian_core_gradient_weights(
    mesh: PolarMesh,
    *,
    degree: int = 3,
    observation_rings: int = 3,
    target_rings: int = 3,
) -> CartesianCoreGradientWeights:
    """Build static LS observation-to-gradient matrices.

    The layout intentionally follows ``AxisCoreFaceReconstruction3D``: rows
    are theta-major/ring-minor observations, and the pseudoinverse is formed
    once outside the field evaluation path.
    """

    degree = int(degree)
    observation_rings = int(observation_rings)
    target_rings = int(target_rings)
    if degree < 0 or observation_rings < 1 or target_rings < 1:
        raise ValueError("degree and ring counts must be positive (degree may be zero)")
    if observation_rings > mesh.r.shape[0] or target_rings > mesh.r.shape[0]:
        raise ValueError("ring count exceeds mesh radial extent")

    r_obs = mesh.r[:observation_rings, :, 0]
    theta_obs = mesh.theta[:observation_rings, :, 0]
    x_obs = (r_obs * np.cos(theta_obs)).T.reshape(-1)
    y_obs = (r_obs * np.sin(theta_obs)).T.reshape(-1)
    design = _basis(x_obs, y_obs, degree)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Cartesian core design matrix is rank deficient")
    coefficient_from_observations = np.linalg.pinv(design)

    r_target = mesh.r[:target_rings, :, 0]
    theta_target = mesh.theta[:target_rings, :, 0]
    x_target = r_target * np.cos(theta_target)
    y_target = r_target * np.sin(theta_target)
    basis_x, basis_y = _basis_gradient(x_target, y_target, degree)
    # Target functionals: (target, theta, coefficient) @ (coefficient, obs).
    gradient_x = np.einsum("tjp,po->tjo", basis_x, coefficient_from_observations)
    gradient_y = np.einsum("tjp,po->tjo", basis_y, coefficient_from_observations)
    return CartesianCoreGradientWeights(
        degree=degree,
        observation_rings=observation_rings,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
    )


def cartesian_core_gradient(field: Array, weights: CartesianCoreGradientWeights) -> tuple[Array, Array]:
    """Apply precomputed core gradient weights independently per zeta slice."""

    observations = np.transpose(field[: weights.observation_rings], (1, 0, 2))
    observations = observations.reshape(observations.shape[0] * observations.shape[1], observations.shape[2])
    grad_x = np.einsum("tjo,oz->tjz", weights.gradient_x, observations)
    grad_y = np.einsum("tjo,oz->tjz", weights.gradient_y, observations)
    return grad_x, grad_y


def cartesian_core_bracket(
    f_callable: Callable[[Array, Array, Array], Array],
    g_callable: Callable[[Array, Array, Array], Array],
    mesh: PolarMesh,
    weights: CartesianCoreGradientWeights,
) -> Array:
    """Use Cartesian gradients in the core and centered polar gradients outside."""

    f = _evaluate_field(f_callable, mesh.r, mesh.theta, mesh.zeta)
    g = _evaluate_field(g_callable, mesh.r, mesh.theta, mesh.zeta)
    f_r, f_theta = _centered_polar_gradients(f_callable, f, mesh)
    g_r, g_theta = _centered_polar_gradients(g_callable, g, mesh)
    f_x, f_y = cartesian_core_gradient(f, weights)
    g_x, g_y = cartesian_core_gradient(g, weights)
    polar = (f_r * g_theta - f_theta * g_r) / mesh.r
    core = f_x * g_y - f_y * g_x
    result = polar.copy()
    result[: weights.gradient_x.shape[0]] = core
    return result


def _centered_polar_gradients(
    field_callable: Callable[[Array, Array, Array], Array],
    values: Array,
    mesh: PolarMesh,
) -> tuple[Array, Array]:
    lower = _evaluate_field(
        field_callable,
        -0.5 * mesh.dr * np.ones_like(mesh.theta[0]),
        mesh.theta[0],
        mesh.zeta[0],
    )
    # The upper boundary is outside the axis experiment.  A centered ghost
    # there keeps the diagnostic array-shaped; ring-wise conclusions are made
    # from the lower/core rings only.
    upper = _evaluate_field(
        field_callable,
        (mesh.r.shape[0] + 0.5) * mesh.dr * np.ones_like(mesh.theta[0]),
        mesh.theta[-1],
        mesh.zeta[-1],
    )
    extended = np.concatenate((lower[None, ...], values, upper[None, ...]), axis=0)
    radial = (extended[2:] - extended[:-2]) / (2.0 * mesh.dr)
    angular = (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * mesh.dtheta)
    return radial, angular


def error_by_ring(actual: Array, exact: Array) -> Array:
    """Return max-norm error by radial ring, over theta and zeta."""

    return np.max(np.abs(actual - exact), axis=(1, 2))


def _results(nx: int, ntheta: int | None = None, nzeta: int = 4) -> dict[str, object]:
    ntheta = 4 * nx if ntheta is None else int(ntheta)
    mesh = make_mesh(nx, ntheta, nzeta)
    fields = manufactured_fields(mesh)
    f_callable, g_callable = _field_functions()
    current = centered_polar_bracket_from_callables(f_callable, g_callable, mesh)
    weights = build_cartesian_core_gradient_weights(mesh)
    core = cartesian_core_bracket(f_callable, g_callable, mesh, weights)
    return {
        "mesh": mesh,
        "exact": fields.bracket,
        "current": current,
        "core": core,
        "current_error": error_by_ring(current, fields.bracket),
        "core_error": error_by_ring(core, fields.bracket),
    }


def test_cartesian_core_reproduces_the_manufactured_axis_core() -> None:
    result = _results(32)
    # Both fields are degree-two Cartesian polynomials, so degree-three LS
    # reconstruction should reproduce their Cartesian gradients to roundoff.
    assert np.max(result["core_error"][:3]) < 2.0e-12
    assert np.max(result["core_error"][3:]) < 2.0e-2


def test_core_improves_the_first_rings_and_converges_with_resolution() -> None:
    coarse = _results(16)
    fine = _results(32)
    assert coarse["core_error"][0] < 2.0e-12
    assert fine["core_error"][0] < 2.0e-12
    assert fine["current_error"][0] < coarse["current_error"][0]
    assert fine["current_error"][1] < coarse["current_error"][1]
    ring0_order = np.log(coarse["current_error"][0] / fine["current_error"][0]) / np.log(2.0)
    ring1_order = np.log(coarse["current_error"][1] / fine["current_error"][1]) / np.log(2.0)
    assert ring0_order > 1.8
    assert ring1_order > 1.8


@pytest.mark.parametrize("closure", ["current", "core"])
def test_constant_preservation_and_antisymmetry(closure: str) -> None:
    mesh = make_mesh(24, 96, 3)
    f_callable, g_callable = _field_functions()
    constant = lambda x, y, z: np.full_like(x, 2.75, dtype=float)
    field_g = _evaluate_field(g_callable, mesh.r, mesh.theta, mesh.zeta)
    weights = build_cartesian_core_gradient_weights(mesh)

    if closure == "current":
        bracket_fg = centered_polar_bracket_from_callables(constant, g_callable, mesh)
        bracket_gf = centered_polar_bracket_from_callables(g_callable, constant, mesh)
    else:
        bracket_fg = cartesian_core_bracket(constant, g_callable, mesh, weights)
        bracket_gf = cartesian_core_bracket(g_callable, constant, mesh, weights)
    assert np.max(np.abs(bracket_fg)) < 1.0e-12
    assert np.max(np.abs(bracket_fg + bracket_gf)) < 1.0e-12
    # Keep the nonconstant field live in this test so accidental unused-field
    # simplifications cannot hide a broken core application.
    assert np.isfinite(field_g).all()


def print_diagnostic(resolutions: tuple[int, ...], ring_count: int) -> None:
    print("Axis-regular Poisson bracket: centered polar vs Cartesian LS core")
    print("manufactured fields: degree-2 Cartesian polynomials, zeta-dependent coefficients")
    print("resolution       current ring-0       core ring-0       current L_inf(core)   core L_inf(core)")
    for nx in resolutions:
        result = _results(nx)
        current_error = result["current_error"]
        core_error = result["core_error"]
        print(
            f"{nx:4d}^2           {current_error[0]: .6e}       {core_error[0]: .6e}       "
            f"{np.max(current_error[:ring_count]): .6e}       {np.max(core_error[:ring_count]): .6e}"
        )

    result = _results(resolutions[-1])
    print(f"\nring errors at nx={resolutions[-1]} (max over theta,zeta)")
    print("ring       current              Cartesian-core")
    for ring, (current, core) in enumerate(
        zip(result["current_error"][:ring_count], result["core_error"][:ring_count])
    ):
        print(f"{ring:4d}       {current: .6e}       {core: .6e}")
    mesh = result["mesh"]
    f_callable, g_callable = _field_functions()
    weights = build_cartesian_core_gradient_weights(mesh)
    constant = lambda x, y, z: np.full_like(x, 2.75, dtype=float)
    for name, evaluator in (
        ("current", lambda a, b: centered_polar_bracket_from_callables(a, b, mesh)),
        ("core", lambda a, b: cartesian_core_bracket(a, b, mesh, weights)),
    ):
        fg = evaluator(constant, g_callable)
        gf = evaluator(g_callable, constant)
        print(f"{name:7s} constant max={np.max(np.abs(fg)):.3e}, antisymmetry max={np.max(np.abs(fg + gf)):.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[16, 24, 32, 48])
    parser.add_argument("--ring-count", type=int, default=6)
    args = parser.parse_args()
    print_diagnostic(tuple(args.resolutions), args.ring_count)


if __name__ == "__main__":
    main()
