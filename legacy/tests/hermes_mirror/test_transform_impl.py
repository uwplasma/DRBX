from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.legacy_hermes import (
    FieldAlignedLocalLayout,
    density_transform_impl,
    pressure_transform_impl,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mirror_exb_local_rank0_t1.npz"


def test_density_transform_impl_restores_dump_backed_x_guards() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        ne = jnp.asarray(data["Ne"], dtype=jnp.float64)
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )

    clobbered = ne.at[:, : layout.xstart, :].set(-7.0)
    clobbered = clobbered.at[:, layout.xend + 1 :, :].set(-9.0)
    expected = density_transform_impl(
        ne,
        layout=layout,
        neumann_boundary_average_z=True,
        lower_x=True,
        upper_x=True,
    )
    out = density_transform_impl(
        clobbered,
        layout=layout,
        neumann_boundary_average_z=True,
        lower_x=True,
        upper_x=True,
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-12, atol=1e-12)

    rms = float(jnp.sqrt(jnp.mean(out * out)))
    interior_rms = float(
        jnp.sqrt(
            jnp.mean(out[layout.pstart : layout.pend + 1, layout.xstart : layout.xend + 1, :] ** 2)
        )
    )
    np.testing.assert_allclose(rms, 1.7785461475277795, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(interior_rms, 1.8245458655422153, rtol=1e-12, atol=1e-12)


def test_density_transform_impl_floors_negative_values() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    density = -jnp.ones((8, 8, 6), dtype=jnp.float64)
    out = density_transform_impl(density, layout=layout, neumann_boundary_average_z=False)
    np.testing.assert_allclose(np.asarray(out), 0.0, rtol=1e-12, atol=1e-12)


def test_pressure_transform_impl_restores_dump_backed_pressure_and_temperature() -> None:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        ne = jnp.asarray(data["Ne"], dtype=jnp.float64)
        pe = jnp.asarray(data["Pe"], dtype=jnp.float64)
        layout = FieldAlignedLocalLayout(
            pstart=int(np.asarray(data["pstart"])),
            pend=int(np.asarray(data["pend"])),
            xstart=int(np.asarray(data["xstart"])),
            xend=int(np.asarray(data["xend"])),
        )

    clobbered = pe.at[:, : layout.xstart, :].set(-3.0)
    clobbered = clobbered.at[:, layout.xend + 1 :, :].set(-5.0)
    density_state = density_transform_impl(
        ne,
        layout=layout,
        neumann_boundary_average_z=True,
        lower_x=True,
        upper_x=True,
    )
    expected_pressure, expected_temperature = pressure_transform_impl(
        pe,
        density_state,
        density_floor=1e-8,
        layout=layout,
        neumann_boundary_average_z=True,
        lower_x=True,
        upper_x=True,
    )
    pressure, temperature = pressure_transform_impl(
        clobbered,
        density_state,
        density_floor=1e-8,
        layout=layout,
        neumann_boundary_average_z=True,
        lower_x=True,
        upper_x=True,
    )

    np.testing.assert_allclose(
        np.asarray(pressure), np.asarray(expected_pressure), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(temperature), np.asarray(expected_temperature), rtol=1e-12, atol=1e-12
    )
    temp_rms = float(jnp.sqrt(jnp.mean(temperature * temperature)))
    interior_rms = float(
        jnp.sqrt(
            jnp.mean(
                temperature[layout.pstart : layout.pend + 1, layout.xstart : layout.xend + 1, :]
                ** 2
            )
        )
    )

    np.testing.assert_allclose(temp_rms, 5.928697471001826e-01, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(interior_rms, 6.081834468193776e-01, rtol=1e-12, atol=1e-12)


def test_pressure_transform_impl_is_differentiable() -> None:
    layout = FieldAlignedLocalLayout(pstart=2, pend=5, xstart=2, xend=5)
    pressure = jax.random.normal(jax.random.PRNGKey(90), (8, 8, 6), dtype=jnp.float64)
    density = 1.0 + jnp.abs(jax.random.normal(jax.random.PRNGKey(91), (8, 8, 6), dtype=jnp.float64))

    grad = jax.grad(
        lambda arr: jnp.sum(
            pressure_transform_impl(
                arr,
                density,
                density_floor=1e-6,
                layout=layout,
                neumann_boundary_average_z=True,
            )[0]
        )
    )(pressure)

    assert np.isfinite(np.asarray(grad)).all()
