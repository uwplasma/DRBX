from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import (
    density_final_global,
    FieldAlignedLocalLayout,
    density_transform_global,
    density_transform_impl,
    prepare_reduced_species_state_global,
    prepare_poloidal_x_dfdy_local_ref,
    prepare_poloidal_y_dfdx_local_ref,
    pressure_final_global,
    pressure_transform_global,
    pressure_transform_impl,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "hermes_mirror_phi_field_aligned_local_rank0_t1.npz"
)
_EXB_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank0_t1.npz"
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


def test_prepare_poloidal_x_dfdy_local_ref_applies_lower_parallel_neumann() -> None:
    with np.load(_EXB_FIXTURE, allow_pickle=False) as data:
        phi = jnp.asarray(data["phi"], dtype=jnp.float64)
        dx = jnp.asarray(data["dx"], dtype=jnp.float64)
        dy = jnp.asarray(data["dy"], dtype=jnp.float64)
        z_shift = jnp.asarray(data["zShift"], dtype=jnp.float64)
        zlength = float(np.asarray(data["zlength"]))
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )

    out = prepare_poloidal_x_dfdy_local_ref(
        phi,
        dy=dy,
        dx=dx,
        z_shift=z_shift,
        zlength=zlength,
        layout=layout,
        interp="spectral",
        lower_boundary_open=True,
        upper_boundary_open=False,
    )

    np.testing.assert_allclose(
        np.asarray(out[layout.pstart - 1, :, :]),
        np.asarray(out[layout.pstart, :, :]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(out[layout.pstart - 2, :, :]),
        np.asarray(out[layout.pstart + 1, :, :]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_density_transform_global_matches_local_interior() -> None:
    with np.load(_EXB_FIXTURE, allow_pickle=False) as data:
        density = jnp.asarray(data["Ne"], dtype=jnp.float64)
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )
        sl = (
            slice(layout.pstart, layout.pend + 1),
            slice(layout.xstart, layout.xend + 1),
            slice(None),
        )

    local = density_transform_impl(
        density,
        layout=layout,
        neumann_boundary_average_z=True,
    )[sl]
    global_prepped = density_transform_global(density[sl])
    np.testing.assert_allclose(
        np.asarray(global_prepped), np.asarray(local), rtol=1e-12, atol=1e-12
    )


def test_pressure_transform_global_matches_local_interior() -> None:
    with np.load(_EXB_FIXTURE, allow_pickle=False) as data:
        density = jnp.asarray(data["Ne"], dtype=jnp.float64)
        pressure = jnp.asarray(data["Pe"], dtype=jnp.float64)
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )
        sl = (
            slice(layout.pstart, layout.pend + 1),
            slice(layout.xstart, layout.xend + 1),
            slice(None),
        )

    density_local = density_transform_impl(
        density,
        layout=layout,
        neumann_boundary_average_z=True,
    )
    pressure_local, temperature_local = pressure_transform_impl(
        pressure,
        density_local,
        density_floor=1e-6,
        layout=layout,
        neumann_boundary_average_z=True,
    )
    pressure_global, temperature_global = pressure_transform_global(
        pressure[sl],
        density[sl],
        density_floor=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(pressure_global),
        np.asarray(pressure_local[sl]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(temperature_global),
        np.asarray(temperature_local[sl]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_prepare_reduced_species_state_global_matches_density_pressure_sequences() -> None:
    with np.load(_EXB_FIXTURE, allow_pickle=False) as data:
        density = jnp.asarray(data["Ne"], dtype=jnp.float64)
        Te = jnp.asarray(data["Pe"], dtype=jnp.float64) / jnp.maximum(density, 1e-12)
        Ti = 0.9 * Te
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )
        sl = (
            slice(layout.pstart, layout.pend + 1),
            slice(layout.xstart, layout.xend + 1),
            slice(None),
        )

    prepared = prepare_reduced_species_state_global(
        density[sl],
        Te[sl],
        Ti[sl],
        density_floor=1e-6,
    )

    density_state = density_final_global(density_transform_global(density[sl]))
    pe_transform, Te_transform = pressure_transform_global(
        density_state * Te[sl],
        density_state,
        density_floor=1e-6,
    )
    pe_state, Te_state, _ = pressure_final_global(pe_transform, Te_transform, density_state)
    pi_transform, Ti_transform = pressure_transform_global(
        density_state * Ti[sl],
        density_state,
        density_floor=1e-6,
    )
    pi_state, Ti_state, _ = pressure_final_global(pi_transform, Ti_transform, density_state)

    np.testing.assert_allclose(np.asarray(prepared.density), np.asarray(density_state))
    np.testing.assert_allclose(np.asarray(prepared.electron_pressure), np.asarray(pe_state))
    np.testing.assert_allclose(np.asarray(prepared.electron_temperature), np.asarray(Te_state))
    np.testing.assert_allclose(np.asarray(prepared.ion_pressure), np.asarray(pi_state))
    np.testing.assert_allclose(np.asarray(prepared.ion_temperature), np.asarray(Ti_state))
