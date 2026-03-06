from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import (
    FieldAlignedLocalLayout,
    GuardLayout,
    ddx_centered_guarded,
    ddy_centered_guarded_local,
    ddy_index_centered_guarded_local,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_phi_metric_local_rank0_t1.npz"
)
_LOCAL_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank0_t1.npz"
)


def test_ddx_centered_guarded_linear_field_has_unit_interior_slope() -> None:
    layout = GuardLayout(xstart=2, xend=5, ystart=2, yend=3)
    x = jnp.arange(8, dtype=jnp.float64)[None, :, None]
    field = jnp.broadcast_to(x, (4, 8, 6))
    dx = jnp.ones((8, 6), dtype=jnp.float64)

    ddx = ddx_centered_guarded(field, dx, layout=layout)

    np.testing.assert_allclose(np.asarray(ddx[:, layout.xstart : layout.xend + 1, :]), 1.0)


def test_ddx_centered_guarded_is_differentiable() -> None:
    layout = GuardLayout(xstart=2, xend=5, ystart=2, yend=3)
    field = jax.random.normal(jax.random.PRNGKey(40), (4, 8, 6), dtype=jnp.float64)
    dx = jnp.ones((8, 6), dtype=jnp.float64)

    grad = jax.grad(lambda arr: jnp.sum(ddx_centered_guarded(arr, dx, layout=layout)))(field)

    assert np.isfinite(np.asarray(grad)).all()


def test_ddx_centered_guarded_matches_dump_backed_boundary_rms() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)
        dx = jnp.asarray(data["dx"], dtype=jnp.float64)
        layout = GuardLayout(
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
            ystart=int(np.asarray(data["ystart"])),
            yend=int(np.asarray(data["yend"])),
        )

    ddx = ddx_centered_guarded(phi, dx, layout=layout)
    lower = np.sqrt(
        np.mean(np.asarray(ddx[:, layout.xstart, layout.ystart : layout.yend + 1]) ** 2)
    )
    upper = np.sqrt(np.mean(np.asarray(ddx[:, layout.xend, layout.ystart : layout.yend + 1]) ** 2))
    interior = np.sqrt(
        np.mean(
            np.asarray(ddx[:, layout.xstart : layout.xend + 1, layout.ystart : layout.yend + 1])
            ** 2
        )
    )

    np.testing.assert_allclose(lower, 3.6083240491965174e-04, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(upper, 4.5560790413198235e-04, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(interior, 2.5071541900451667e-04, rtol=1e-12, atol=1e-12)


def test_ddy_centered_guarded_local_linear_field_has_unit_interior_slope() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    y = jnp.arange(8, dtype=jnp.float64)[:, None, None]
    field = jnp.broadcast_to(y, (8, 8, 6))
    dy = jnp.ones((8, 8), dtype=jnp.float64)

    ddy = ddy_centered_guarded_local(field, dy, layout=layout)

    np.testing.assert_allclose(np.asarray(ddy[layout.pstart : layout.pend + 1, :, :]), 1.0)


def test_ddy_centered_guarded_local_matches_dump_backed_rms() -> None:
    with np.load(_LOCAL_FIXTURE, allow_pickle=False) as data:
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)
        dy = jnp.asarray(data["dy"], dtype=jnp.float64)
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )

    ddy = ddy_centered_guarded_local(phi, dy, layout=layout)
    lower = float(jnp.sqrt(jnp.mean(ddy[layout.pstart, layout.xstart : layout.xend + 1, :] ** 2)))
    upper = float(jnp.sqrt(jnp.mean(ddy[layout.pend, layout.xstart : layout.xend + 1, :] ** 2)))
    interior = float(
        jnp.sqrt(
            jnp.mean(ddy[layout.pstart : layout.pend + 1, layout.xstart : layout.xend + 1, :] ** 2)
        )
    )

    np.testing.assert_allclose(lower, 5.902675901490807e-01, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(upper, 1.0488478686555056e00, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(interior, 6.891203174242735e-01, rtol=1e-12, atol=1e-12)


def test_ddy_index_centered_guarded_local_linear_field_has_unit_index_slope() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    y = jnp.arange(8, dtype=jnp.float64)[:, None, None]
    field = jnp.broadcast_to(y, (8, 8, 6))

    ddy = ddy_index_centered_guarded_local(field, layout=layout)

    np.testing.assert_allclose(np.asarray(ddy[layout.pstart : layout.pend + 1, :, :]), 1.0)
