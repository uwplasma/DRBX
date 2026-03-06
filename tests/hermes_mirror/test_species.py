from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import FieldAlignedLocalLayout, prepare_poloidal_y_dfdx_local_ref

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "hermes_mirror_phi_field_aligned_local_rank0_t1.npz"
)


def test_prepare_poloidal_y_dfdx_local_ref_zero_shift_linear_field() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    x = jnp.arange(8, dtype=jnp.float64)[None, :, None]
    field = jnp.broadcast_to(x, (8, 8, 6))
    dx = jnp.ones((8, 8), dtype=jnp.float64)
    z_shift = jnp.zeros((8, 8), dtype=jnp.float64)

    out = prepare_poloidal_y_dfdx_local_ref(
        field,
        dx=dx,
        z_shift=z_shift,
        zlength=6.0,
        layout=layout,
        interp="spectral",
    )

    np.testing.assert_allclose(np.asarray(out), 1.0, rtol=1e-12, atol=1e-12)


def test_prepare_poloidal_y_dfdx_local_ref_is_differentiable() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    field = jax.random.normal(jax.random.PRNGKey(50), (8, 8, 6), dtype=jnp.float64)
    dx = jnp.ones((8, 8), dtype=jnp.float64)
    z_shift = jnp.zeros((8, 8), dtype=jnp.float64)

    grad = jax.grad(
        lambda arr: jnp.sum(
            prepare_poloidal_y_dfdx_local_ref(
                arr,
                dx=dx,
                z_shift=z_shift,
                zlength=6.0,
                layout=layout,
                interp="spectral",
            )
        )
    )(field)

    assert np.isfinite(np.asarray(grad)).all()


def test_prepare_poloidal_y_dfdx_local_ref_dump_backed_values() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)
        dx = jnp.asarray(data["dx"], dtype=jnp.float64)
        z_shift = jnp.asarray(data["zShift"], dtype=jnp.float64)
        zlength = float(np.asarray(data["zlength"]))
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )

    out = prepare_poloidal_y_dfdx_local_ref(
        phi,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        interp="spectral",
    )

    total_rms = float(jnp.sqrt(jnp.mean(out * out)))
    left_rms = float(
        jnp.sqrt(jnp.mean(out[layout.pstart : layout.pend + 1, layout.xstart, :] ** 2))
    )
    right_rms = float(jnp.sqrt(jnp.mean(out[layout.pstart : layout.pend + 1, layout.xend, :] ** 2)))
    interior_rms = float(
        jnp.sqrt(
            jnp.mean(out[layout.pstart : layout.pend + 1, layout.xstart : layout.xend + 1, :] ** 2)
        )
    )

    np.testing.assert_allclose(total_rms, 3.45924382389202e-04, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(left_rms, 3.6083240491965163e-04, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(right_rms, 4.556079041319823e-04, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(interior_rms, 2.5071541900451683e-04, rtol=1e-12, atol=1e-12)


def test_prepare_poloidal_y_dfdx_local_ref_differs_from_guardless_dump_path() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)
        dx = jnp.asarray(data["dx"], dtype=jnp.float64)
        z_shift = jnp.asarray(data["zShift"], dtype=jnp.float64)
        zlength = float(np.asarray(data["zlength"]))
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )

    mirror = prepare_poloidal_y_dfdx_local_ref(
        phi,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        interp="spectral",
    )

    dx3d = jnp.broadcast_to(dx[:, :, None], phi.shape)
    xp = jnp.take(phi, jnp.clip(jnp.arange(phi.shape[1]) + 1, 0, phi.shape[1] - 1), axis=1)
    xm = jnp.take(phi, jnp.clip(jnp.arange(phi.shape[1]) - 1, 0, phi.shape[1] - 1), axis=1)
    guardless = (xp - xm) / jnp.maximum(2.0 * dx3d, 1e-30)
    guardless = guardless.at[:, 0, :].set((phi[:, 1, :] - phi[:, 0, :]) / dx[:, 0, None])
    guardless = guardless.at[:, -1, :].set((phi[:, -1, :] - phi[:, -2, :]) / dx[:, -1, None])

    from jaxdrb.hermes_mirror import build_shifted_metric_fft_phases, to_field_aligned_all_fft

    phases = build_shifted_metric_fft_phases(
        z_shift,
        nx=phi.shape[1],
        npar=phi.shape[0],
        nbinorm=phi.shape[2],
        zlength=zlength,
        open_field_line=True,
    )
    guardless = to_field_aligned_all_fft(guardless, phases)

    rel = float(
        jnp.sqrt(jnp.mean((mirror - guardless) ** 2))
        / jnp.maximum(jnp.sqrt(jnp.mean(mirror * mirror)), 1e-30)
    )

    assert rel > 0.5
