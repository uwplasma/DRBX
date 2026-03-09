from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.legacy_hermes import (
    GuardLayout,
    Stencil1D,
    apply_free_o2_field3d,
    apply_neumann_boundary_average_z,
    apply_neumann_field3d,
    limit_free,
    mc_limiter,
    minmod,
    set_boundary_to_midpoint,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_limit_free_modes_match_hermes_cases() -> None:
    assert np.isclose(float(limit_free(4.0, 2.0, 0)), 1.0)
    assert np.isclose(float(limit_free(1.0, 2.0, 0)), 2.0)
    assert np.isclose(float(limit_free(1.0e-12, 2.0, 1)), 2.0)
    assert np.isclose(float(limit_free(5.0, 2.0, 2)), -1.0)


def test_limit_free_is_vectorized_and_differentiable() -> None:
    fm = jnp.array([2.0, 4.0, 8.0], dtype=jnp.float64)

    def total(fc: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(limit_free(fm, fc, 1.0))

    fc = jnp.array([1.5, 2.0, 3.0], dtype=jnp.float64)
    grad = jax.grad(total)(fc)

    np.testing.assert_allclose(
        np.asarray(grad),
        np.asarray(2.0 * fc / fm),
        rtol=1e-12,
        atol=1e-12,
    )


def test_minmod_and_mc_limiter_match_hermes_formula() -> None:
    a = jnp.array([2.0, 1.0, -1.0], dtype=jnp.float64)
    b = jnp.array([3.0, -1.0, -2.0], dtype=jnp.float64)
    c = jnp.array([1.0, 2.0, -3.0], dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(minmod(a, b, c)), np.asarray([1.0, 0.0, -1.0]))

    stencil = Stencil1D(
        c=jnp.array([2.0], dtype=jnp.float64),
        m=jnp.array([1.0], dtype=jnp.float64),
        p=jnp.array([4.0], dtype=jnp.float64),
    )
    limited = mc_limiter(stencil)
    np.testing.assert_allclose(np.asarray(limited.L), np.asarray([1.25]))
    np.testing.assert_allclose(np.asarray(limited.R), np.asarray([2.75]))


def test_apply_neumann_boundary_average_z_matches_density_transform_impl() -> None:
    layout = GuardLayout(xstart=2, xend=4, ystart=2, yend=5)
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8)

    out = apply_neumann_boundary_average_z(field, layout=layout)
    avg_lower = np.asarray(field[:, layout.xstart, :]).mean(axis=0, keepdims=True)
    avg_upper = np.asarray(field[:, layout.xend, :]).mean(axis=0, keepdims=True)
    expect_lower = 2.0 * avg_lower - np.asarray(field[:, layout.xstart, :])
    expect_upper = 2.0 * avg_upper - np.asarray(field[:, layout.xend, :])

    np.testing.assert_allclose(np.asarray(out[:, layout.xstart - 1, :]), expect_lower)
    np.testing.assert_allclose(np.asarray(out[:, layout.xstart - 2, :]), expect_lower)
    np.testing.assert_allclose(np.asarray(out[:, layout.xend + 1, :]), expect_upper)
    np.testing.assert_allclose(np.asarray(out[:, layout.xend + 2, :]), expect_upper)
    np.testing.assert_allclose(
        np.asarray(out[:, layout.xstart : layout.xend + 1, :]),
        np.asarray(field[:, layout.xstart : layout.xend + 1, :]),
    )


def test_apply_neumann_boundary_average_z_is_differentiable() -> None:
    layout = GuardLayout(xstart=2, xend=4, ystart=2, yend=5)

    def total(field: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(apply_neumann_boundary_average_z(field, layout=layout))

    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8) / 10.0
    grad = jax.grad(total)(field)
    assert np.isfinite(np.asarray(grad)).all()


def test_apply_neumann_boundary_average_z_matches_dump_backed_fixture() -> None:
    fixture_path = _FIXTURE_DIR / "hermes_mirror_ne_local_rank0_t1.npz"
    with np.load(fixture_path, allow_pickle=False) as data:
        layout = GuardLayout(
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
            ystart=int(np.asarray(data["ystart"])),
            yend=int(np.asarray(data["yend"])),
        )
        field = jnp.asarray(data["Ne"])
        work = field.at[:, : layout.xstart, :].set(0.0)
        work = work.at[:, layout.xend + 1 :, :].set(0.0)
        out = apply_neumann_boundary_average_z(work, layout=layout)

        np.testing.assert_allclose(
            np.asarray(out[:, layout.xstart - 1, :]),
            np.asarray(data["Ne__neumann_lower"]),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(out[:, layout.xstart - 2, :]),
            np.asarray(data["Ne__neumann_lower"]),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(out[:, layout.xend + 1, :]),
            np.asarray(data["Ne__neumann_upper"]),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(out[:, layout.xend + 2, :]),
            np.asarray(data["Ne__neumann_upper"]),
            rtol=1e-12,
            atol=1e-12,
        )


def test_apply_neumann_field3d_zero_gradient_matches_centered_bout_rule() -> None:
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8)

    out = apply_neumann_field3d(field, axis=1, interior_start=2, interior_end=4, guard_width=2)

    np.testing.assert_allclose(np.asarray(out[:, 1, :]), np.asarray(field[:, 2, :]))
    np.testing.assert_allclose(np.asarray(out[:, 0, :]), np.asarray(field[:, 3, :]))
    np.testing.assert_allclose(np.asarray(out[:, 5, :]), np.asarray(field[:, 4, :]))
    np.testing.assert_allclose(np.asarray(out[:, 6, :]), np.asarray(field[:, 3, :]))


def test_apply_neumann_field3d_nonzero_gradient_matches_centered_formula() -> None:
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8)
    spacing = jnp.ones_like(field) * 2.0

    out = apply_neumann_field3d(
        field,
        axis=1,
        interior_start=2,
        interior_end=4,
        spacing=spacing,
        lower_gradient=1.5,
        upper_gradient=-0.5,
        guard_width=2,
    )

    np.testing.assert_allclose(np.asarray(out[:, 1, :]), np.asarray(field[:, 2, :] - 3.0))
    np.testing.assert_allclose(np.asarray(out[:, 0, :]), np.asarray(field[:, 3, :] - 9.0))
    np.testing.assert_allclose(np.asarray(out[:, 5, :]), np.asarray(field[:, 4, :] - 1.0))
    np.testing.assert_allclose(np.asarray(out[:, 6, :]), np.asarray(field[:, 3, :] - 3.0))


def test_apply_neumann_field3d_supports_last_axis_boundaries() -> None:
    field = jnp.arange(3 * 5 * 7, dtype=jnp.float64).reshape(3, 5, 7)

    out = apply_neumann_field3d(
        field,
        axis=2,
        interior_start=2,
        interior_end=4,
        spacing=1.0,
        lower_gradient=2.0,
        upper_gradient=-1.0,
        guard_width=2,
    )

    np.testing.assert_allclose(np.asarray(out[:, :, 1]), np.asarray(field[:, :, 2] - 2.0))
    np.testing.assert_allclose(np.asarray(out[:, :, 0]), np.asarray(field[:, :, 3] - 6.0))
    np.testing.assert_allclose(np.asarray(out[:, :, 5]), np.asarray(field[:, :, 4] - 1.0))
    np.testing.assert_allclose(np.asarray(out[:, :, 6]), np.asarray(field[:, :, 3] - 3.0))


def test_apply_neumann_field3d_is_differentiable() -> None:
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8) / 10.0

    grad = jax.grad(
        lambda arr: jnp.sum(
            apply_neumann_field3d(
                arr,
                axis=1,
                interior_start=2,
                interior_end=4,
                spacing=1.0,
                lower_gradient=0.25,
                upper_gradient=-0.5,
            )
        )
    )(field)

    assert np.isfinite(np.asarray(grad)).all()


def test_apply_free_o2_field3d_matches_bout_recursive_linear_extrapolation() -> None:
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8)

    out = apply_free_o2_field3d(field, axis=1, interior_start=2, interior_end=4, guard_width=2)

    expect = np.asarray(field).copy()
    expect[:, 1, :] = (2.0 * expect[:, 2, :]) - expect[:, 3, :]
    expect[:, 0, :] = (2.0 * expect[:, 1, :]) - expect[:, 2, :]
    expect[:, 5, :] = (2.0 * expect[:, 4, :]) - expect[:, 3, :]
    expect[:, 6, :] = (2.0 * expect[:, 5, :]) - expect[:, 4, :]
    np.testing.assert_allclose(np.asarray(out), expect)


def test_apply_free_o2_field3d_dump_backed_values_follow_bout_formula() -> None:
    fixture_path = _FIXTURE_DIR / "hermes_mirror_ne_local_rank0_t1.npz"
    with np.load(fixture_path, allow_pickle=False) as data:
        field = jnp.asarray(data["Ne"], dtype=jnp.float64)
        layout = GuardLayout(
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
            ystart=int(np.asarray(data["ystart"])),
            yend=int(np.asarray(data["yend"])),
        )
        work = field.at[:, : layout.xstart, :].set(0.0)
        work = work.at[:, layout.xend + 1 :, :].set(0.0)
        out = apply_free_o2_field3d(
            work,
            axis=1,
            interior_start=layout.xstart,
            interior_end=layout.xend,
            guard_width=layout.x_guards,
        )
        expect = np.asarray(work).copy()
        expect[:, layout.xstart - 1, :] = (
            2.0 * expect[:, layout.xstart, :] - expect[:, layout.xstart + 1, :]
        )
        expect[:, layout.xstart - 2, :] = (
            2.0 * expect[:, layout.xstart - 1, :] - expect[:, layout.xstart, :]
        )
        expect[:, layout.xend + 1, :] = (
            2.0 * expect[:, layout.xend, :] - expect[:, layout.xend - 1, :]
        )
        expect[:, layout.xend + 2, :] = (
            2.0 * expect[:, layout.xend + 1, :] - expect[:, layout.xend, :]
        )
        np.testing.assert_allclose(np.asarray(out), expect, rtol=1e-12, atol=1e-12)


def test_apply_free_o2_field3d_is_differentiable() -> None:
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8) / 10.0
    grad = jax.grad(
        lambda arr: jnp.sum(
            apply_free_o2_field3d(
                arr,
                axis=1,
                interior_start=2,
                interior_end=4,
                guard_width=2,
            )
        )
    )(field)
    assert np.isfinite(np.asarray(grad)).all()


def test_set_boundary_to_midpoint_matches_bout_recursive_x_update() -> None:
    layout = GuardLayout(xstart=2, xend=4, ystart=2, yend=5)
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8)
    reference = field + 100.0

    out = set_boundary_to_midpoint(field, reference, layout=layout, apply_x=True, apply_y=False)

    expect = np.asarray(field).copy()
    ref = np.asarray(reference)
    for x in range(layout.xstart - 1, layout.xstart - layout.x_guards - 1, -1):
        interior = x + 1
        expect[:, x, :] = ref[:, x, :] + ref[:, interior, :] - expect[:, interior, :]
    for x in range(layout.xend + 1, layout.xend + layout.x_guards + 1):
        interior = x - 1
        expect[:, x, :] = ref[:, x, :] + ref[:, interior, :] - expect[:, interior, :]

    np.testing.assert_allclose(np.asarray(out), expect)


def test_set_boundary_to_midpoint_matches_bout_recursive_y_update() -> None:
    layout = GuardLayout(xstart=2, xend=4, ystart=2, yend=5)
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8)
    reference = field - 50.0

    out = set_boundary_to_midpoint(field, reference, layout=layout, apply_x=False, apply_y=True)

    expect = np.asarray(field).copy()
    ref = np.asarray(reference)
    for y in range(layout.ystart - 1, layout.ystart - layout.y_guards - 1, -1):
        interior = y + 1
        expect[:, :, y] = ref[:, :, y] + ref[:, :, interior] - expect[:, :, interior]
    for y in range(layout.yend + 1, layout.yend + layout.y_guards + 1):
        interior = y - 1
        expect[:, :, y] = ref[:, :, y] + ref[:, :, interior] - expect[:, :, interior]

    np.testing.assert_allclose(np.asarray(out), expect)


def test_set_boundary_to_midpoint_is_differentiable_in_field_and_reference() -> None:
    layout = GuardLayout(xstart=2, xend=4, ystart=2, yend=5)
    field = jnp.arange(3 * 7 * 8, dtype=jnp.float64).reshape(3, 7, 8) / 10.0
    reference = field + 1.0

    grad_field = jax.grad(
        lambda f: jnp.sum(
            set_boundary_to_midpoint(f, reference, layout=layout, apply_x=True, apply_y=False)
        )
    )(field)
    grad_ref = jax.grad(
        lambda ref: jnp.sum(
            set_boundary_to_midpoint(field, ref, layout=layout, apply_x=True, apply_y=False)
        )
    )(reference)

    assert np.isfinite(np.asarray(grad_field)).all()
    assert np.isfinite(np.asarray(grad_ref)).all()
