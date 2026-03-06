from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import (
    FieldAlignedLocalLayout,
    div_n_bxgrad_f_b_xppm_xy_x_local,
    div_n_bxgrad_f_b_xppm_xy_x_local_from_fields,
    div_n_bxgrad_f_b_xppm_xy_x_local_from_fields_ref,
    div_n_bxgrad_f_b_xppm_xy_x_local_ref,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank0_t1.npz"


def test_exb_x_local_zero_metric_gives_zero_flux() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n = jax.random.normal(jax.random.PRNGKey(70), (8, 8, 6), dtype=jnp.float64)
    dfdy = jax.random.normal(jax.random.PRNGKey(71), (8, 8, 6), dtype=jnp.float64)
    out = div_n_bxgrad_f_b_xppm_xy_x_local(
        n,
        dfdy,
        jacobian=1.0,
        dx=1.0,
        g11=1.0,
        g23=0.0,
        bxy=1.0,
        layout=layout,
        bndry_flux=True,
    )

    np.testing.assert_allclose(np.asarray(out), 0.0, rtol=1e-12, atol=1e-12)


def test_exb_x_local_ref_matches_fused() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n = jax.random.normal(jax.random.PRNGKey(72), (8, 8, 6), dtype=jnp.float64)
    dfdy = jax.random.normal(jax.random.PRNGKey(73), (8, 8, 6), dtype=jnp.float64)
    jacobian = 1.0 + 0.1 * jnp.arange(8, dtype=jnp.float64)[:, None]
    dx = jnp.ones((8, 8), dtype=jnp.float64)
    g11 = jnp.ones((8, 8), dtype=jnp.float64)
    g23 = 0.15 * jnp.ones((8, 8), dtype=jnp.float64)
    bxy = jnp.ones((8, 8), dtype=jnp.float64)

    ref = div_n_bxgrad_f_b_xppm_xy_x_local_ref(
        n,
        dfdy,
        jacobian=jacobian,
        dx=dx,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=True,
    )
    fused = div_n_bxgrad_f_b_xppm_xy_x_local(
        n,
        dfdy,
        jacobian=jacobian,
        dx=dx,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=True,
    )

    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_exb_x_local_from_fields_is_differentiable() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n = jax.random.normal(jax.random.PRNGKey(74), (8, 8, 6), dtype=jnp.float64)
    phi = jax.random.normal(jax.random.PRNGKey(75), (8, 8, 6), dtype=jnp.float64)
    dy = jnp.ones((8, 8), dtype=jnp.float64)
    dx = jnp.ones((8, 8), dtype=jnp.float64)
    z_shift = jnp.zeros((8, 8), dtype=jnp.float64)
    jacobian = jnp.ones((8, 8), dtype=jnp.float64)
    g11 = jnp.ones((8, 8), dtype=jnp.float64)
    g23 = 0.15 * jnp.ones((8, 8), dtype=jnp.float64)
    bxy = jnp.ones((8, 8), dtype=jnp.float64)

    grad = jax.grad(
        lambda arr: jnp.sum(
            div_n_bxgrad_f_b_xppm_xy_x_local_from_fields(
                arr,
                phi,
                dy=dy,
                dx=dx,
                z_shift=z_shift,
                zlength=6.0,
                jacobian=jacobian,
                g11=g11,
                g23=g23,
                bxy=bxy,
                layout=layout,
                bndry_flux=True,
            )
        )
    )(n)

    assert np.isfinite(np.asarray(grad)).all()


def test_exb_x_local_dump_backed_ne_pe_values() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)
        ne = jnp.asarray(data["Ne"], dtype=jnp.float64)
        pe = jnp.asarray(data["Pe"], dtype=jnp.float64)
        dy = jnp.asarray(data["dy"], dtype=jnp.float64)
        dx = jnp.asarray(data["dx"], dtype=jnp.float64)
        z_shift = jnp.asarray(data["zShift"], dtype=jnp.float64)
        zlength = float(np.asarray(data["zlength"]))
        jacobian = jnp.asarray(data["J"], dtype=jnp.float64)
        g11 = jnp.asarray(data["g11"], dtype=jnp.float64)
        g23 = jnp.asarray(data["g23"], dtype=jnp.float64)
        bxy = jnp.asarray(data["Bxy"], dtype=jnp.float64)
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )

    ne_ref = div_n_bxgrad_f_b_xppm_xy_x_local_from_fields_ref(
        ne,
        phi,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=True,
    )
    ne_fused = div_n_bxgrad_f_b_xppm_xy_x_local_from_fields(
        ne,
        phi,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=True,
    )
    pe_ref = div_n_bxgrad_f_b_xppm_xy_x_local_from_fields_ref(
        pe,
        phi,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=True,
    )
    pe_fused = div_n_bxgrad_f_b_xppm_xy_x_local_from_fields(
        pe,
        phi,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=True,
    )

    np.testing.assert_allclose(np.asarray(ne_fused), np.asarray(ne_ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pe_fused), np.asarray(pe_ref), rtol=1e-12, atol=1e-12)

    ne_rms = float(jnp.sqrt(jnp.mean(ne_ref * ne_ref)))
    pe_rms = float(jnp.sqrt(jnp.mean(pe_ref * pe_ref)))

    np.testing.assert_allclose(ne_rms, 5.391187274308899e-03, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pe_rms, 5.289137581776043e-03, rtol=1e-12, atol=1e-12)
