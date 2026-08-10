"""Manufactured-field experiment for non-modal polar Poisson brackets.

This file is intentionally standalone: it does not import or modify the
production FCI implementation.  It models the two-dimensional transverse
part of the toroidal bracket with a uniform polar cell grid,

    {f, g} = f_x g_y - f_y g_x
            = (f_r g_theta - f_theta g_r) / r.

The three closures are:

``signed_radius``
    The current centered polar derivative with the lower-radius ghost filled
    by ``f(-r, theta) = f(r, theta + pi)``.

``cartesian_core``
    A fixed local least-squares Cartesian polynomial fit using the first few
    radial rings.  The fit is performed independently for each scalar and
    produces Cartesian derivatives directly in the core; the signed-radius
    derivative is retained outside the core.

``paired_average``
    A cheap topology-aware output average on the first rings,
    ``P q(theta) = (q(theta) + q(theta + pi))/2``.  It is included as a
    deliberately conservative heuristic/stability filter, not as a claimed
    axis-regular discretization.  The diagnostics expose the angular content
    it destroys.

The script reports exact manufactured-bracket error by ring, constant
preservation, pointwise antisymmetry, a global integral proxy, and a discrete
integration-by-parts/conservation proxy.  It raises no accuracy assertions:
bad alternatives are printed as failures rather than hidden by tuned tests.

Run, for example:

    python3 tests/axis_regular_poisson_alternatives_experiment.py \
        --radial-points 32 --theta-points 64 --core-rings 3 --degree 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Grid:
    radial_points: int
    theta_points: int
    radius: float

    @property
    def dr(self) -> float:
        return self.radius / self.radial_points

    @property
    def dtheta(self) -> float:
        return 2.0 * np.pi / self.theta_points

    @property
    def r(self) -> Array:
        return (np.arange(self.radial_points, dtype=float) + 0.5) * self.dr

    @property
    def theta(self) -> Array:
        return np.arange(self.theta_points, dtype=float) * self.dtheta

    @property
    def rr(self) -> Array:
        return self.r[:, None]

    @property
    def tt(self) -> Array:
        return self.theta[None, :]

    @property
    def x(self) -> Array:
        return self.rr * np.cos(self.tt)

    @property
    def y(self) -> Array:
        return self.rr * np.sin(self.tt)

    @property
    def volume_weights(self) -> Array:
        return self.rr * self.dr * self.dtheta


@dataclass(frozen=True)
class FieldPair:
    f: Array
    g: Array
    exact_bracket: Array


def manufactured_fields(grid: Grid) -> FieldPair:
    """Return smooth Cartesian fields and their analytic bracket.

    The fourth-power radial envelope is a polynomial in x and y, so the
    fields are smooth at the axis.  It vanishes at the physical outer circle,
    making the continuous integral of a bracket over the disk zero.
    """

    x, y = grid.x, grid.y
    r2 = x * x + y * y
    rho = 1.0 - r2 / grid.radius**2
    envelope = rho**4
    envelope_x = -8.0 * x * rho**3 / grid.radius**2
    envelope_y = -8.0 * y * rho**3 / grid.radius**2

    pf = (
        1.0
        + 0.25 * x
        - 0.18 * y
        + 0.13 * x**2
        + 0.09 * x * y
        - 0.07 * y**2
        + 0.03 * x**3
        - 0.02 * x**2 * y
        + 0.04 * x * y**2
    )
    pf_x = 0.25 + 0.26 * x + 0.09 * y + 0.09 * x**2 - 0.04 * x * y + 0.04 * y**2
    pf_y = -0.18 + 0.09 * x - 0.14 * y - 0.02 * x**2 + 0.08 * x * y

    pg = (
        0.17
        - 0.21 * x
        + 0.31 * y
        + 0.12 * x**2
        - 0.08 * x * y
        + 0.10 * y**2
        + 0.05 * x**3
        + 0.03 * x**2 * y
        - 0.04 * x * y**2
    )
    pg_x = -0.21 + 0.24 * x - 0.08 * y + 0.15 * x**2 + 0.06 * x * y - 0.04 * y**2
    pg_y = 0.31 - 0.08 * x + 0.20 * y + 0.03 * x**2 - 0.08 * x * y

    f = envelope * pf
    g = envelope * pg
    fx = envelope_x * pf + envelope * pf_x
    fy = envelope_y * pf + envelope * pf_y
    gx = envelope_x * pg + envelope * pg_x
    gy = envelope_y * pg + envelope * pg_y
    return FieldPair(f=f, g=g, exact_bracket=fx * gy - fy * gx)


def periodic_shift(values: Array, shift: int) -> Array:
    return np.roll(values, shift, axis=-1)


def signed_radius_derivatives(values: Array, grid: Grid) -> tuple[Array, Array]:
    """Centered polar derivatives with the current signed-radius ghost rule."""

    nr, nt = values.shape
    radial = np.empty_like(values)
    # The explicit construction below keeps the axis rule visible and avoids
    # assuming that a generic array pad knows about the theta half-turn.
    axis_ghost = periodic_shift(values[0], nt // 2)
    radial[0] = (values[1] - axis_ghost) / (2.0 * grid.dr) if nr > 1 else 0.0
    if nr > 2:
        radial[1:-1] = (values[2:] - values[:-2]) / (2.0 * grid.dr)
    if nr > 1:
        # A second-order one-sided derivative at the physical outer edge.
        radial[-1] = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * grid.dr) if nr > 2 else (values[-1] - values[-2]) / grid.dr

    theta = (periodic_shift(values, -1) - periodic_shift(values, 1)) / (2.0 * grid.dtheta)
    return radial, theta


def polar_bracket_from_derivatives(
    f_radial: Array,
    f_theta: Array,
    g_radial: Array,
    g_theta: Array,
    grid: Grid,
) -> Array:
    return (f_radial * g_theta - f_theta * g_radial) / grid.rr


def signed_radius_bracket(f: Array, g: Array, grid: Grid) -> Array:
    fr, ft = signed_radius_derivatives(f, grid)
    gr, gt = signed_radius_derivatives(g, grid)
    return polar_bracket_from_derivatives(fr, ft, gr, gt, grid)


def monomial_exponents(degree: int) -> list[tuple[int, int]]:
    return [
        (px, degree_total - px)
        for degree_total in range(degree + 1)
        for px in range(degree_total, -1, -1)
    ]


def monomial_matrix(
    x: Array,
    y: Array,
    exponents: list[tuple[int, int]],
    *,
    coordinate_scale: float = 1.0,
) -> Array:
    """Evaluate a scaled Cartesian basis to avoid tiny-core conditioning."""

    x_scaled = np.asarray(x) / coordinate_scale
    y_scaled = np.asarray(y) / coordinate_scale
    return np.stack([x_scaled**px * y_scaled**py for px, py in exponents], axis=-1)


def monomial_gradient_matrix(
    x: Array,
    y: Array,
    exponents: list[tuple[int, int]],
    *,
    coordinate_scale: float = 1.0,
) -> tuple[Array, Array]:
    x_scaled = np.asarray(x) / coordinate_scale
    y_scaled = np.asarray(y) / coordinate_scale
    dx_columns = []
    dy_columns = []
    for px, py in exponents:
        dx_columns.append(
            px * x_scaled ** max(px - 1, 0) * y_scaled**py / coordinate_scale
            if px
            else np.zeros_like(x)
        )
        dy_columns.append(
            py * x_scaled**px * y_scaled ** max(py - 1, 0) / coordinate_scale
            if py
            else np.zeros_like(y)
        )
    return np.stack(dx_columns, axis=-1), np.stack(dy_columns, axis=-1)


def cartesian_core_bracket(
    f: Array,
    g: Array,
    grid: Grid,
    *,
    core_rings: int,
    degree: int,
) -> tuple[Array, dict[str, float]]:
    """Use one fixed Cartesian LS fit for each scalar in the core."""

    fr, ft = signed_radius_derivatives(f, grid)
    gr, gt = signed_radius_derivatives(g, grid)
    polar_result = polar_bracket_from_derivatives(fr, ft, gr, gt, grid)

    core_rings = min(int(core_rings), grid.radial_points)
    exponents = monomial_exponents(int(degree))
    coordinate_scale = float(grid.r[core_rings - 1] + 0.5 * grid.dr)
    observations = monomial_matrix(
        grid.x[:core_rings].reshape(-1),
        grid.y[:core_rings].reshape(-1),
        exponents,
        coordinate_scale=coordinate_scale,
    )
    condition_number = float(np.linalg.cond(observations))
    if np.linalg.matrix_rank(observations) < observations.shape[1]:
        raise ValueError(
            f"Cartesian core design is rank deficient: shape={observations.shape}, "
            f"degree={degree}, rings={core_rings}"
        )
    f_coeff = np.linalg.lstsq(observations, f[:core_rings].reshape(-1), rcond=None)[0]
    g_coeff = np.linalg.lstsq(observations, g[:core_rings].reshape(-1), rcond=None)[0]
    f_dx, f_dy = monomial_gradient_matrix(
        grid.x,
        grid.y,
        exponents,
        coordinate_scale=coordinate_scale,
    )
    g_dx, g_dy = f_dx, f_dy
    f_x = np.einsum("...p,p->...", f_dx, f_coeff)
    f_y = np.einsum("...p,p->...", f_dy, f_coeff)
    g_x = np.einsum("...p,p->...", g_dx, g_coeff)
    g_y = np.einsum("...p,p->...", g_dy, g_coeff)
    result = polar_result.copy()
    result[:core_rings] = f_x[:core_rings] * g_y[:core_rings] - f_y[:core_rings] * g_x[:core_rings]
    return result, {"design_condition_number": condition_number}


def paired_average_bracket(f: Array, g: Array, grid: Grid, *, core_rings: int) -> Array:
    """Apply a cheap diametric-pair average to the signed-radius result."""

    result = signed_radius_bracket(f, g, grid)
    half_turn = grid.theta_points // 2
    if grid.theta_points % 2:
        raise ValueError("paired_average requires an even theta_points count")
    averaged = 0.5 * (result + periodic_shift(result, half_turn))
    result[: min(core_rings, grid.radial_points)] = averaged[: min(core_rings, grid.radial_points)]
    return result


def ring_rms(values: Array) -> Array:
    return np.sqrt(np.mean(values * values, axis=1))


def relative_error_by_ring(result: Array, exact: Array) -> tuple[Array, Array]:
    absolute = ring_rms(result - exact)
    scale = np.maximum(ring_rms(exact), 1.0e-14)
    return absolute, absolute / scale


def scalar_metrics(
    bracket: Callable[[Array, Array], Array],
    fields: FieldPair,
    grid: Grid,
    *,
    core_rings: int = 3,
) -> dict[str, object]:
    f, g, exact = fields.f, fields.g, fields.exact_bracket
    result = bracket(f, g)
    reverse = bracket(g, f)
    constant = np.ones_like(f)
    one_g = bracket(constant, g)
    f_one = bracket(f, constant)
    abs_error, rel_error = relative_error_by_ring(result, exact)
    core_extent = max(1, min(int(core_rings), grid.radial_points))
    weights = grid.volume_weights
    global_integral = float(np.sum(weights * result))
    exact_global_integral = float(np.sum(weights * exact))

    # For compactly supported smooth fields, integration by parts gives
    # <f,{g,h}> + <g,{f,h}> = 0.  This is a useful conservation proxy even
    # though the pointwise bracket is not itself a finite-volume divergence.
    h = 0.7 * f - 0.35 * g + np.roll(f, 3, axis=1)
    fg_h = bracket(g, h)
    gf_h = bracket(f, h)
    ibp_defect = float(np.sum(weights * (f * fg_h + g * gf_h)))
    ibp_scale = float(
        np.sum(weights * (np.abs(f * fg_h) + np.abs(g * gf_h)))
    )

    return {
        "result": result,
        "abs_error_by_ring": abs_error,
        "rel_error_by_ring": rel_error,
        "max_abs_error": float(np.max(np.abs(result - exact))),
        "core_max_abs_error": float(np.max(abs_error[:core_extent])),
        "outer_max_abs_error": float(np.max(abs_error[core_extent:])),
        "constant_preservation_max": float(max(np.max(np.abs(one_g)), np.max(np.abs(f_one)))),
        "antisymmetry_max": float(np.max(np.abs(result + reverse))),
        "global_integral": global_integral,
        "exact_sampled_global_integral": exact_global_integral,
        "global_integral_abs": abs(global_integral),
        "ibp_conservation_defect": ibp_defect,
        "ibp_conservation_relative": abs(ibp_defect) / max(ibp_scale, 1.0e-14),
    }


def print_method_report(name: str, metrics: dict[str, object], extra: dict[str, float] | None = None) -> None:
    absolute = np.asarray(metrics["abs_error_by_ring"])
    relative = np.asarray(metrics["rel_error_by_ring"])
    print(f"\n{name}")
    print(f"  max abs exact-bracket error : {metrics['max_abs_error']:.6e}")
    print(f"  core max abs error          : {metrics['core_max_abs_error']:.6e}")
    print(f"  outside-core max abs error  : {metrics['outer_max_abs_error']:.6e}")
    print(f"  constant preservation max  : {metrics['constant_preservation_max']:.6e}")
    print(f"  antisymmetry max            : {metrics['antisymmetry_max']:.6e}")
    print(f"  global integral             : {metrics['global_integral']:.6e}")
    print(f"  exact sampled integral      : {metrics['exact_sampled_global_integral']:.6e}")
    print(f"  IBP/conservation relative   : {metrics['ibp_conservation_relative']:.6e}")
    print("  ring | abs RMS | relative RMS")
    for ring, (abs_value, rel_value) in enumerate(zip(absolute, relative)):
        print(f"  {ring:4d} | {abs_value:.6e} | {rel_value:.6e}")
    if extra:
        for key, value in extra.items():
            print(f"  {key.replace('_', ' '):28s}: {value:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radial-points", type=int, default=32)
    parser.add_argument("--theta-points", type=int, default=64)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--core-rings", type=int, default=3)
    parser.add_argument("--degree", type=int, default=3)
    args = parser.parse_args()
    if args.radial_points < 4 or args.theta_points < 8:
        parser.error("need at least 4 radial and 8 theta points")
    if args.theta_points % 2:
        parser.error("theta-points must be even for the diametric signed-radius topology")
    grid = Grid(args.radial_points, args.theta_points, args.radius)
    fields = manufactured_fields(grid)

    signed_metrics = scalar_metrics(
        lambda f, g: signed_radius_bracket(f, g, grid),
        fields,
        grid,
        core_rings=args.core_rings,
    )
    cartesian_metrics = scalar_metrics(
        lambda f, g: cartesian_core_bracket(
            f, g, grid, core_rings=args.core_rings, degree=args.degree
        )[0],
        fields,
        grid,
        core_rings=args.core_rings,
    )
    _, cartesian_metadata = cartesian_core_bracket(
        fields.f,
        fields.g,
        grid,
        core_rings=args.core_rings,
        degree=args.degree,
    )
    paired_metrics = scalar_metrics(
        lambda f, g: paired_average_bracket(f, g, grid, core_rings=args.core_rings),
        fields,
        grid,
        core_rings=args.core_rings,
    )

    print("Axis-regular Poisson-bracket alternatives experiment")
    print(f"grid: radial={args.radial_points}, theta={args.theta_points}, R={args.radius:g}")
    print(f"core: rings={args.core_rings}, Cartesian degree={args.degree}")
    print("exact bracket: analytic Cartesian derivatives of a smooth compact-envelope field pair")
    print_method_report("[1] signed-radius centered polar derivatives", signed_metrics)
    print_method_report("[2] local Cartesian polynomial core", cartesian_metrics, cartesian_metadata)
    print_method_report("[3] diametric-pair output average (heuristic)", paired_metrics)

    print("\nInterpretation flags (diagnostic, not pass/fail assertions)")
    for name, metrics in (
        ("signed_radius", signed_metrics),
        ("cartesian_core", cartesian_metrics),
        ("paired_average", paired_metrics),
    ):
        core_error = float(np.max(np.asarray(metrics["rel_error_by_ring"])[: args.core_rings]))
        outer_error = float(np.max(np.asarray(metrics["rel_error_by_ring"])[args.core_rings :]))
        print(
            f"  {name:16s}: core max relative error={core_error:.3e}, "
            f"outside-core max relative error={outer_error:.3e}, "
            f"global IBP defect={metrics['ibp_conservation_relative']:.3e}"
        )
    print(
        "\nNote: pointwise antisymmetry is expected to be near roundoff for all three "
        "because each uses the same cross-product ordering in both argument orders; "
        "it does not establish conservation."
    )


if __name__ == "__main__":
    main()
