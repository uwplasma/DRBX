from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import FieldAlignedLocalLayout, div_a_grad_perp, div_a_grad_perp_local


def test_div_a_grad_perp_local_matches_quadratic_x_second_derivative() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    x = jnp.arange(8, dtype=jnp.float64)[None, :, None]
    f = jnp.broadcast_to(x * x, (8, 8, 6))
    a = jnp.ones_like(f)

    out = div_a_grad_perp_local(
        a,
        f,
        jacobian=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        g11=1.0,
        g23=0.0,
        g_22=1.0,
        g_23=0.0,
        g33=1.0,
        bxy=1.0,
        z_shift=0.0,
        zlength=6.0,
        layout=layout,
        interp="spectral",
        periodic_binormal=True,
    )

    interior = np.asarray(out[layout.pstart : layout.pend + 1, layout.xstart + 1 : layout.xend, :])
    np.testing.assert_allclose(interior, 2.0, rtol=1e-12, atol=1e-12)


def test_div_a_grad_perp_runtime_matches_local_slice() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5, open_field_line=False)
    key_a, key_f = jax.random.split(jax.random.PRNGKey(17))
    a_inner = 1.0 + 0.05 * jax.random.normal(key_a, (4, 4, 6), dtype=jnp.float64)
    f_inner = jax.random.normal(key_f, (4, 4, 6), dtype=jnp.float64)

    def _pad_periodic(arr: jnp.ndarray) -> jnp.ndarray:
        x_padded = jnp.concatenate([arr[:, -2:, :], arr, arr[:, :2, :]], axis=1)
        return jnp.concatenate([x_padded[-2:, :, :], x_padded, x_padded[:2, :, :]], axis=0)

    a_local = _pad_periodic(a_inner)
    f_local = _pad_periodic(f_inner)

    local = div_a_grad_perp_local(
        a_local,
        f_local,
        jacobian=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        g11=1.0,
        g23=0.15,
        g_22=1.0,
        g_23=0.15,
        g33=1.0,
        bxy=1.0,
        z_shift=0.0,
        zlength=6.0,
        layout=layout,
    )
    runtime = div_a_grad_perp(
        a_inner,
        f_inner,
        jacobian=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        g11=1.0,
        g23=0.15,
        g_22=1.0,
        g_23=0.15,
        g33=1.0,
        bxy=1.0,
        z_shift=0.0,
        zlength=6.0,
        bc_kind_x=0,
        interp="spectral",
        periodic_parallel=True,
        periodic_binormal=True,
        lower_boundary_open=False,
        upper_boundary_open=False,
    )

    runtime_np = np.asarray(runtime)
    local_np = np.asarray(
        local[layout.pstart : layout.pend + 1, layout.xstart : layout.xend + 1, :]
    )
    diff_rms = float(np.sqrt(np.mean((runtime_np - local_np) ** 2)))
    corr = float(np.corrcoef(runtime_np.ravel(), local_np.ravel())[0, 1])

    assert diff_rms < 6.0e-2, diff_rms
    assert corr > 0.999, corr


def test_div_a_grad_perp_is_differentiable() -> None:
    key_a, key_f = jax.random.split(jax.random.PRNGKey(23))
    a = 1.0 + 0.01 * jax.random.normal(key_a, (4, 4, 6), dtype=jnp.float64)
    f = jax.random.normal(key_f, (4, 4, 6), dtype=jnp.float64)

    grad = jax.grad(
        lambda arr: jnp.sum(
            div_a_grad_perp(
                a,
                arr,
                jacobian=1.0,
                dx=1.0,
                dy=1.0,
                dz=1.0,
                g11=1.0,
                g23=0.1,
                g_22=1.0,
                g_23=0.1,
                g33=1.0,
                bxy=1.0,
                z_shift=0.0,
                zlength=6.0,
                bc_kind_x=2,
                periodic_parallel=False,
            )
        )
    )(f)

    assert np.isfinite(np.asarray(grad)).all()
