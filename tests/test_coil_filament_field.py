"""MAKEGRID coil filaments as a field provider for the HSX geometry builder."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.fci_braginskii.geometry_build.provider_bfield import (  # noqa: E402
    JaxProviderBFieldEvaluator,
    ProviderBFieldEvaluator,
)
from drbx.geometry.field_provider import FilamentCoilField, grad_B  # noqa: E402

MU0 = 4.0e-7 * np.pi


def _loop(radius: float, n: int, z: float = 0.0) -> np.ndarray:
    t = 2.0 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.full(n, z)], axis=-1)


def test_segment_biot_savart_is_exact_for_a_polygon_and_converges_to_a_circle() -> None:
    radius, current = 1.3, 2.0e4
    z = np.array([0.0, 0.4, -0.9])
    exact = MU0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)
    errors = []
    for n in (32, 64, 128):
        field = FilamentCoilField(nodes=jnp.asarray(_loop(radius, n))[None], currents=jnp.asarray([current]))
        bz = np.asarray(field.B(jnp.asarray(np.stack([0 * z, 0 * z, z], axis=-1))))[:, 2]
        errors.append(np.max(np.abs(bz / exact - 1.0)))
    # A regular inscribed n-gon carries the circle's current through a
    # shorter path: the on-axis error is O(n^-2).
    assert errors[0] < 1.0e-2
    assert 3.5 < errors[0] / errors[1] < 4.5 and 3.5 < errors[1] / errors[2] < 4.5


def test_filament_field_is_divergence_and_curl_free() -> None:
    nodes = jnp.asarray(np.stack([_loop(1.0, 60, 0.2), _loop(1.2, 60, -0.3)]))
    field = FilamentCoilField(nodes=nodes, currents=jnp.asarray([1.0e4, -6.0e3]))
    points = jnp.asarray([[0.3, 0.1, 0.0], [0.9, -0.2, 0.05], [0.0, 0.5, -0.1]])
    jac = np.asarray(grad_B(field, points))
    scale = np.max(np.abs(jac), axis=(-2, -1))
    assert np.max(np.abs(np.trace(jac, axis1=-2, axis2=-1)) / scale) < 1.0e-11
    assert np.max(np.abs(jac - np.swapaxes(jac, -1, -2)).max(axis=(-2, -1)) / scale) < 1.0e-11


def test_filament_field_is_jittable_and_differentiable_in_nodes() -> None:
    field = FilamentCoilField(nodes=jnp.asarray(_loop(1.0, 40))[None], currents=jnp.asarray([1.0e4]))
    point = jnp.asarray([[0.0, 0.0, 0.3]])
    np.testing.assert_allclose(jax.jit(lambda f, x: f.B(x))(field, point), field.B(point), rtol=1e-14)

    def bz(scale):
        return FilamentCoilField(nodes=field.nodes * scale, currents=field.currents).B(point)[0, 2]

    grad = float(jax.grad(bz)(1.0))
    fd = float((bz(1.0 + 1e-6) - bz(1.0 - 1e-6)) / 2e-6)
    assert abs(grad - fd) < 1.0e-6 * abs(fd)


def test_makegrid_reader_groups_currents_and_drops_zero_current_coils(tmp_path: Path) -> None:
    lines = ["periods  2", "begin filament", "mirror NUL"]
    for group, radius in ((1, 1.0), (2, 1.5)):
        pts = _loop(radius, 6)
        for point in pts:
            lines.append(f"  {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}  -1")
        first = pts[0]
        lines.append(f"  {first[0]:.8f} {first[1]:.8f} {first[2]:.8f}  0  {group} Coil{group}")
    lines.append("end")
    path = tmp_path / "coils.test"
    path.write_text("\n".join(lines) + "\n")

    field = FilamentCoilField.from_makegrid(path, {1: 500.0, 2: 0.0})
    assert field.nfp == 2
    assert field.nodes.shape == (1, 6, 3)
    np.testing.assert_allclose(field.currents, [-500.0])
    assert field.provenance["names"] == ("Coil1",)


def test_provider_evaluator_matches_the_provider_in_both_frames() -> None:
    field = FilamentCoilField(nodes=jnp.asarray(_loop(1.0, 48, 0.1))[None], currents=jnp.asarray([1.0e4]), nfp=4)
    evaluator = ProviderBFieldEvaluator(field, R_bounds=(0.5, 1.5), Z_bounds=(-0.4, 0.4))
    assert evaluator.nfp == 4 and np.isclose(evaluator.period, np.pi / 2)
    rphiz = np.array([[0.8, 0.3, 0.05], [1.1, -2.0, -0.1]])
    xyz = np.stack([rphiz[:, 0] * np.cos(rphiz[:, 1]), rphiz[:, 0] * np.sin(rphiz[:, 1]), rphiz[:, 2]], -1)
    b = np.asarray(field.B(jnp.asarray(xyz)))
    cyl = evaluator.evaluate_cylindrical(rphiz)
    c, s = np.cos(rphiz[:, 1]), np.sin(rphiz[:, 1])
    np.testing.assert_allclose(cyl[:, 0] * c - cyl[:, 1] * s, b[:, 0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(evaluator(xyz), b, rtol=1e-13)
    jax_view = evaluator.jax_evaluator()
    assert isinstance(jax_view, JaxProviderBFieldEvaluator)
    np.testing.assert_allclose(jax_view.evaluate_cartesian(jnp.asarray(xyz)), b, rtol=1e-13)
    identity = evaluator.content_identity()
    assert identity["scheme"] == "field-provider-v1" and len(identity["sha256"]) == 64


HSX_INPUTS = Path(os.environ.get("DRBX_HSX_INPUTS", "artifacts")).expanduser()


@pytest.mark.slow
@pytest.mark.skipif(
    not (HSX_INPUTS / "inputs/hsx/coils.hsx").exists()
    or not (HSX_INPUTS / "inputs/hsx/mgrid_res2p5cm_180pln.nc").exists(),
    reason="HSX coils/mgrid inputs not present (set DRBX_HSX_INPUTS)",
)
def test_hsx_coil_field_matches_the_mgrid_at_bundle_cell_centres() -> None:
    from drbx.fci_braginskii.geometry_build.Bfield_evaluator import bfield_evaluator_from_makegrid

    currents = [10722.0] * 6 + [0.0] * 6
    coils = FilamentCoilField.from_makegrid(
        HSX_INPUTS / "inputs/hsx/coils.hsx", {g + 1: c for g, c in enumerate(currents)}
    )
    mgrid = bfield_evaluator_from_makegrid(
        HSX_INPUTS / "inputs/hsx/mgrid_res2p5cm_180pln.nc", currents=currents, method="cubic"
    )
    positions = np.load(HSX_INPUTS / "geometry/hsx_fci_32x32x32/cell_positions.npz")["cell_positions"]
    points = positions.reshape(-1, 3)[::7]
    b_coil = np.asarray(coils.B(jnp.asarray(points)))
    b_mgrid = mgrid.evaluate_cartesian(points)
    rel = np.linalg.norm(b_coil - b_mgrid, axis=-1) / np.linalg.norm(b_mgrid, axis=-1)
    assert np.median(rel) < 1.0e-3 and rel.max() < 1.0e-2
