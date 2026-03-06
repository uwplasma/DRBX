from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import (
    FieldAlignedLocalLayout,
    div_n_bxgrad_f_b_xppm_local,
    div_n_bxgrad_f_b_xppm_local_ref,
    div_n_bxgrad_f_b_xppm_xz,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank0_t1.npz"


def test_exb_local_full_matches_xz_when_poloidal_disabled() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n = jax.random.normal(jax.random.PRNGKey(80), (8, 8, 6), dtype=jnp.float64)
    phi = jax.random.normal(jax.random.PRNGKey(81), (8, 8, 6), dtype=jnp.float64)
    jacobian = jnp.ones((8, 8), dtype=jnp.float64)
    dx = jnp.ones((8, 8), dtype=jnp.float64)
    dy = jnp.ones((8, 8), dtype=jnp.float64)
    dz = jnp.ones((8, 8), dtype=jnp.float64)
    g11 = jnp.ones((8, 8), dtype=jnp.float64)
    g23 = 0.2 * jnp.ones((8, 8), dtype=jnp.float64)
    bxy = jnp.ones((8, 8), dtype=jnp.float64)
    z_shift = jnp.zeros((8, 8), dtype=jnp.float64)

    mirror = div_n_bxgrad_f_b_xppm_local(
        n,
        phi,
        jacobian=jacobian,
        dx=dx,
        dy=dy,
        dz=dz,
        g11=g11,
        g23=g23,
        bxy=bxy,
        z_shift=z_shift,
        zlength=6.0,
        layout=layout,
        bndry_flux=True,
        poloidal=False,
        bc_kind_x=2,
        neumann_boundary_average_z=True,
    )
    xz = div_n_bxgrad_f_b_xppm_xz(
        n,
        phi,
        jacobian=jacobian,
        dx=dx,
        dz=dz,
        periodic_x=False,
        periodic_z=True,
        bndry_flux=True,
        use_mc=True,
        bc_kind_x=2,
        neumann_boundary_average_z=True,
    )

    np.testing.assert_allclose(np.asarray(mirror), np.asarray(xz), rtol=1e-12, atol=1e-12)


def test_exb_local_full_ref_matches_fused() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n = jax.random.normal(jax.random.PRNGKey(82), (8, 8, 6), dtype=jnp.float64)
    phi = jax.random.normal(jax.random.PRNGKey(83), (8, 8, 6), dtype=jnp.float64)
    jacobian = 1.0 + 0.1 * jnp.arange(8, dtype=jnp.float64)[:, None]
    dx = jnp.ones((8, 8), dtype=jnp.float64)
    dy = jnp.ones((8, 8), dtype=jnp.float64)
    dz = jnp.ones((8, 8), dtype=jnp.float64)
    g11 = jnp.ones((8, 8), dtype=jnp.float64)
    g23 = 0.2 * jnp.ones((8, 8), dtype=jnp.float64)
    bxy = jnp.ones((8, 8), dtype=jnp.float64)
    z_shift = jnp.zeros((8, 8), dtype=jnp.float64)

    ref = div_n_bxgrad_f_b_xppm_local_ref(
        n,
        phi,
        jacobian=jacobian,
        dx=dx,
        dy=dy,
        dz=dz,
        g11=g11,
        g23=g23,
        bxy=bxy,
        z_shift=z_shift,
        zlength=6.0,
        layout=layout,
        bndry_flux=True,
        poloidal=True,
        bc_kind_x=2,
        neumann_boundary_average_z=True,
    )
    fused = div_n_bxgrad_f_b_xppm_local(
        n,
        phi,
        jacobian=jacobian,
        dx=dx,
        dy=dy,
        dz=dz,
        g11=g11,
        g23=g23,
        bxy=bxy,
        z_shift=z_shift,
        zlength=6.0,
        layout=layout,
        bndry_flux=True,
        poloidal=True,
        bc_kind_x=2,
        neumann_boundary_average_z=True,
    )

    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_exb_local_full_is_differentiable() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    n = jax.random.normal(jax.random.PRNGKey(84), (8, 8, 6), dtype=jnp.float64)
    phi = jax.random.normal(jax.random.PRNGKey(85), (8, 8, 6), dtype=jnp.float64)
    jacobian = jnp.ones((8, 8), dtype=jnp.float64)
    dx = jnp.ones((8, 8), dtype=jnp.float64)
    dy = jnp.ones((8, 8), dtype=jnp.float64)
    dz = jnp.ones((8, 8), dtype=jnp.float64)
    g11 = jnp.ones((8, 8), dtype=jnp.float64)
    g23 = 0.2 * jnp.ones((8, 8), dtype=jnp.float64)
    bxy = jnp.ones((8, 8), dtype=jnp.float64)
    z_shift = jnp.zeros((8, 8), dtype=jnp.float64)

    grad = jax.grad(
        lambda arr: jnp.sum(
            div_n_bxgrad_f_b_xppm_local(
                arr,
                phi,
                jacobian=jacobian,
                dx=dx,
                dy=dy,
                dz=dz,
                g11=g11,
                g23=g23,
                bxy=bxy,
                z_shift=z_shift,
                zlength=6.0,
                layout=layout,
                bndry_flux=True,
                poloidal=True,
                bc_kind_x=2,
                neumann_boundary_average_z=True,
            )
        )
    )(n)

    assert np.isfinite(np.asarray(grad)).all()


def test_exb_local_full_dump_backed_ne_pe_values() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)
        ne = jnp.asarray(data["Ne"], dtype=jnp.float64)
        pe = jnp.asarray(data["Pe"], dtype=jnp.float64)
        dx = jnp.asarray(data["dx"], dtype=jnp.float64)
        dy = jnp.asarray(data["dy"], dtype=jnp.float64)
        dz = jnp.asarray(data["dz"], dtype=jnp.float64)
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

    ne_ref = div_n_bxgrad_f_b_xppm_local_ref(
        ne,
        phi,
        jacobian=jacobian,
        dx=dx,
        dy=dy,
        dz=dz,
        g11=g11,
        g23=g23,
        bxy=bxy,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        bndry_flux=True,
        poloidal=True,
        lower_boundary_open=lower,
        upper_boundary_open=upper,
        bc_kind_x=2,
        neumann_boundary_average_z=True,
    )
    ne_fused = div_n_bxgrad_f_b_xppm_local(
        ne,
        phi,
        jacobian=jacobian,
        dx=dx,
        dy=dy,
        dz=dz,
        g11=g11,
        g23=g23,
        bxy=bxy,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        bndry_flux=True,
        poloidal=True,
        lower_boundary_open=lower,
        upper_boundary_open=upper,
        bc_kind_x=2,
        neumann_boundary_average_z=True,
    )
    pe_ref = div_n_bxgrad_f_b_xppm_local_ref(
        pe,
        phi,
        jacobian=jacobian,
        dx=dx,
        dy=dy,
        dz=dz,
        g11=g11,
        g23=g23,
        bxy=bxy,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        bndry_flux=True,
        poloidal=True,
        lower_boundary_open=lower,
        upper_boundary_open=upper,
    )
    pe_fused = div_n_bxgrad_f_b_xppm_local(
        pe,
        phi,
        jacobian=jacobian,
        dx=dx,
        dy=dy,
        dz=dz,
        g11=g11,
        g23=g23,
        bxy=bxy,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        bndry_flux=True,
        poloidal=True,
        lower_boundary_open=lower,
        upper_boundary_open=upper,
    )

    np.testing.assert_allclose(np.asarray(ne_fused), np.asarray(ne_ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pe_fused), np.asarray(pe_ref), rtol=1e-12, atol=1e-12)

    ne_rms = float(jnp.sqrt(jnp.mean(ne_ref * ne_ref)))
    pe_rms = float(jnp.sqrt(jnp.mean(pe_ref * pe_ref)))

    np.testing.assert_allclose(ne_rms, 7.204357588601792e-03, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pe_rms, 7.071453164542667e-03, rtol=1e-12, atol=1e-12)
