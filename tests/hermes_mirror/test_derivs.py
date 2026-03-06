from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import GuardLayout, ddx_centered_guarded

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_phi_metric_local_rank0_t1.npz"
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
