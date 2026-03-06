from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import (
    FieldAlignedLocalLayout,
    div_n_bxgrad_f_b_xppm_xy_y_local,
    div_n_bxgrad_f_b_xppm_xy_y_local_from_fields,
    div_n_bxgrad_f_b_xppm_xy_y_local_from_fields_ref,
    div_n_bxgrad_f_b_xppm_xy_y_local_ref,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank0_t1.npz"


def test_exb_y_local_zero_metric_gives_zero_flux() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n_fa = jax.random.normal(jax.random.PRNGKey(60), (8, 8, 6), dtype=jnp.float64)
    dfdx_fa = jax.random.normal(jax.random.PRNGKey(61), (8, 8, 6), dtype=jnp.float64)
    out = div_n_bxgrad_f_b_xppm_xy_y_local(
        n_fa,
        dfdx_fa,
        jacobian=1.0,
        dy=1.0,
        g11=1.0,
        g23=0.0,
        bxy=1.0,
        layout=layout,
        bndry_flux=True,
        lower_boundary_open=True,
        upper_boundary_open=False,
    )

    np.testing.assert_allclose(np.asarray(out), 0.0, rtol=1e-12, atol=1e-12)


def test_exb_y_local_ref_matches_fused() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n_fa = jax.random.normal(jax.random.PRNGKey(62), (8, 8, 6), dtype=jnp.float64)
    dfdx_fa = jax.random.normal(jax.random.PRNGKey(63), (8, 8, 6), dtype=jnp.float64)
    jacobian = 1.0 + 0.1 * jnp.arange(8, dtype=jnp.float64)[:, None]
    dy = jnp.ones((8, 8), dtype=jnp.float64)
    g11 = jnp.ones((8, 8), dtype=jnp.float64)
    g23 = 0.2 * jnp.ones((8, 8), dtype=jnp.float64)
    bxy = jnp.ones((8, 8), dtype=jnp.float64)

    ref = div_n_bxgrad_f_b_xppm_xy_y_local_ref(
        n_fa,
        dfdx_fa,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=True,
        lower_boundary_open=True,
        upper_boundary_open=False,
    )
    fused = div_n_bxgrad_f_b_xppm_xy_y_local(
        n_fa,
        dfdx_fa,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        bndry_flux=True,
        lower_boundary_open=True,
        upper_boundary_open=False,
    )

    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_exb_y_local_from_fields_is_differentiable() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n = jax.random.normal(jax.random.PRNGKey(64), (8, 8, 6), dtype=jnp.float64)
    phi = jax.random.normal(jax.random.PRNGKey(65), (8, 8, 6), dtype=jnp.float64)
    dx = jnp.ones((8, 8), dtype=jnp.float64)
    dy = jnp.ones((8, 8), dtype=jnp.float64)
    jacobian = jnp.ones((8, 8), dtype=jnp.float64)
    g11 = jnp.ones((8, 8), dtype=jnp.float64)
    g23 = 0.2 * jnp.ones((8, 8), dtype=jnp.float64)
    bxy = jnp.ones((8, 8), dtype=jnp.float64)
    z_shift = jnp.zeros((8, 8), dtype=jnp.float64)

    grad = jax.grad(
        lambda arr: jnp.sum(
            div_n_bxgrad_f_b_xppm_xy_y_local_from_fields(
                arr,
                phi,
                dx=dx,
                z_shift=z_shift,
                zlength=6.0,
                jacobian=jacobian,
                dy=dy,
                g11=g11,
                g23=g23,
                bxy=bxy,
                layout=layout,
                interp="spectral",
                bndry_flux=True,
                lower_boundary_open=True,
                upper_boundary_open=False,
            )
        )
    )(n)

    assert np.isfinite(np.asarray(grad)).all()


def test_exb_y_local_dump_backed_ne_pe_values() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)
        ne = jnp.asarray(data["Ne"], dtype=jnp.float64)
        pe = jnp.asarray(data["Pe"], dtype=jnp.float64)
        dx = jnp.asarray(data["dx"], dtype=jnp.float64)
        dy = jnp.asarray(data["dy"], dtype=jnp.float64)
        jacobian = jnp.asarray(data["J"], dtype=jnp.float64)
        g11 = jnp.asarray(data["g11"], dtype=jnp.float64)
        g23 = jnp.asarray(data["g23"], dtype=jnp.float64)
        bxy = jnp.asarray(data["Bxy"], dtype=jnp.float64)
        z_shift = jnp.asarray(data["zShift"], dtype=jnp.float64)
        zlength = float(np.asarray(data["zlength"]))
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )
        lower = bool(np.asarray(data["lower_boundary_open"]))
        upper = bool(np.asarray(data["upper_boundary_open"]))

    ne_ref = div_n_bxgrad_f_b_xppm_xy_y_local_from_fields_ref(
        ne,
        phi,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        interp="spectral",
        bndry_flux=True,
        lower_boundary_open=lower,
        upper_boundary_open=upper,
    )
    ne_fused = div_n_bxgrad_f_b_xppm_xy_y_local_from_fields(
        ne,
        phi,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        interp="spectral",
        bndry_flux=True,
        lower_boundary_open=lower,
        upper_boundary_open=upper,
    )
    pe_ref = div_n_bxgrad_f_b_xppm_xy_y_local_from_fields_ref(
        pe,
        phi,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        interp="spectral",
        bndry_flux=True,
        lower_boundary_open=lower,
        upper_boundary_open=upper,
    )
    pe_fused = div_n_bxgrad_f_b_xppm_xy_y_local_from_fields(
        pe,
        phi,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        jacobian=jacobian,
        dy=dy,
        g11=g11,
        g23=g23,
        bxy=bxy,
        layout=layout,
        interp="spectral",
        bndry_flux=True,
        lower_boundary_open=lower,
        upper_boundary_open=upper,
    )

    np.testing.assert_allclose(np.asarray(ne_fused), np.asarray(ne_ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pe_fused), np.asarray(pe_ref), rtol=1e-12, atol=1e-12)

    ne_rms = float(jnp.sqrt(jnp.mean(ne_ref * ne_ref)))
    ne_interior_rms = float(
        jnp.sqrt(
            jnp.mean(
                ne_ref[layout.pstart : layout.pend + 1, layout.xstart : layout.xend + 1, :] ** 2
            )
        )
    )
    pe_rms = float(jnp.sqrt(jnp.mean(pe_ref * pe_ref)))
    pe_interior_rms = float(
        jnp.sqrt(
            jnp.mean(
                pe_ref[layout.pstart : layout.pend + 1, layout.xstart : layout.xend + 1, :] ** 2
            )
        )
    )

    np.testing.assert_allclose(ne_rms, 5.266245270548453e-03, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ne_interior_rms, 2.3208656645780424e-03, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pe_rms, 5.06778021563735e-03, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pe_interior_rms, 2.1455021379486773e-03, rtol=1e-12, atol=1e-12)
