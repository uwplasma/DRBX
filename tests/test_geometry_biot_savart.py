from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.geometry import biot_savart_field, build_fourier_coil_set, load_essos_biot_savart_json


def _circular_x_y_fourier_dofs(radius: float) -> jnp.ndarray:
    dofs = jnp.zeros((1, 3, 3), dtype=jnp.float64)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    return dofs


def test_biot_savart_circle_matches_center_field() -> None:
    radius = 0.7
    current = 1.8
    coils = build_fourier_coil_set(
        base_dofs=_circular_x_y_fourier_dofs(radius),
        base_currents=jnp.asarray([current], dtype=jnp.float64),
        n_segments=256,
    )

    field = np.asarray(biot_savart_field(coils, jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)))
    expected_bz = 2.0 * np.pi * 1.0e-7 * current / radius

    assert np.allclose(field[:2], 0.0, atol=2.0e-14)
    assert np.isclose(field[2], expected_bz, rtol=2.0e-12, atol=0.0)


def test_essos_json_loader_uses_fourier_convention_and_sanitized_source(tmp_path: Path) -> None:
    coil_json = tmp_path / "ESSOS_biot_savart_circle.json"
    payload = {
        "nfp": 1,
        "stellsym": False,
        "order": 1,
        "n_segments": 128,
        "dofs_curves": np.asarray(_circular_x_y_fourier_dofs(1.1)).tolist(),
        "dofs_currents": [2.5],
    }
    coil_json.write_text(json.dumps(payload), encoding="utf-8")

    coils = load_essos_biot_savart_json(coil_json)
    field = np.asarray(biot_savart_field(coils, jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)))

    assert coils.metadata["source"] == coil_json.name
    assert coils.metadata["order"] == 1
    assert np.isclose(field[2], 2.0 * np.pi * 1.0e-7 * 2.5 / 1.1, rtol=3.0e-12)


def test_biot_savart_sums_over_coils_before_segment_average() -> None:
    radius = 0.8
    dofs = jnp.concatenate([_circular_x_y_fourier_dofs(radius), _circular_x_y_fourier_dofs(radius)], axis=0)
    coils = build_fourier_coil_set(
        base_dofs=dofs,
        base_currents=jnp.asarray([1.2, 0.8], dtype=jnp.float64),
        n_segments=192,
    )

    field = np.asarray(biot_savart_field(coils, jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)))
    expected_bz = 2.0 * np.pi * 1.0e-7 * 2.0 / radius

    assert np.isclose(field[2], expected_bz, rtol=2.0e-12, atol=0.0)


def test_essos_symmetry_expansion_flips_current_for_stellarator_symmetry() -> None:
    coils = build_fourier_coil_set(
        base_dofs=_circular_x_y_fourier_dofs(1.0),
        base_currents=jnp.asarray([3.0], dtype=jnp.float64),
        n_segments=16,
        nfp=2,
        stellsym=True,
    )

    assert coils.n_coils == 4
    assert np.count_nonzero(np.asarray(coils.currents) > 0.0) == 2
    assert np.count_nonzero(np.asarray(coils.currents) < 0.0) == 2


def test_biot_savart_field_is_jvp_transformable() -> None:
    coils = build_fourier_coil_set(
        base_dofs=_circular_x_y_fourier_dofs(0.9),
        base_currents=jnp.asarray([1.0], dtype=jnp.float64),
        n_segments=128,
    )

    point = jnp.asarray([0.02, -0.03, 0.04], dtype=jnp.float64)
    tangent = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float64)
    field, field_jvp = jax.jvp(lambda xyz: biot_savart_field(coils, xyz), (point,), (tangent,))

    assert field.shape == (3,)
    assert field_jvp.shape == (3,)
    assert np.all(np.isfinite(np.asarray(field)))
    assert np.all(np.isfinite(np.asarray(field_jvp)))
