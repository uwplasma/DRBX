"""ESSOS coil-field adapter: numerical checks on the installed ``essos`` package.

These replace the retired ``drbx.validation.essos_*`` campaign modules. The
campaign reports wrapped the same quantities in JSON ledgers; here each
quantity is asserted directly.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry import (  # noqa: E402
    EssosCoilField,
    build_essos_imported_fci_geometry,
    essos_runtime_available,
    grad_B,
    resolve_essos_landreman_qa_json,
    resolve_essos_landreman_qa_wout,
    trace_essos_coil_field_lines,
)
import drbx.geometry.essos_import as essos_import  # noqa: E402

pytestmark = pytest.mark.skipif(not essos_runtime_available(), reason="essos is not installed")


def _qa_wout_available() -> bool:
    try:
        resolve_essos_landreman_qa_wout()
    except FileNotFoundError:
        return False
    return True


@pytest.fixture(scope="module")
def qa_field() -> EssosCoilField:
    return EssosCoilField.from_json(resolve_essos_landreman_qa_json())


@pytest.fixture(scope="module")
def sample_points() -> jnp.ndarray:
    rng = np.random.default_rng(3)
    R = rng.uniform(1.15, 1.35, 6)
    phi = rng.uniform(0.0, 2.0 * np.pi, 6)
    Z = rng.uniform(-0.08, 0.08, 6)
    return jnp.asarray(np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1))


def test_bundled_coils_and_installed_essos_are_used_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DRBX_ESSOS_ROOT", raising=False)
    path = resolve_essos_landreman_qa_json()
    assert path.is_file() and "drbx" in path.parts
    before = list(sys.path)
    modules = essos_import._import_essos_modules()
    assert sys.path == before
    assert modules["BiotSavart"].__module__ == "essos.fields"
    with pytest.raises(FileNotFoundError, match="not bundled"):
        resolve_essos_landreman_qa_wout()


def test_provider_is_a_jittable_pytree(qa_field, sample_points) -> None:
    jitted = jax.jit(lambda provider, x: provider.B(x))
    np.testing.assert_allclose(jitted(qa_field, sample_points), qa_field.B(sample_points), rtol=1.0e-13)
    assert qa_field.provenance["kind"] == "essos_coils"


def test_qa_coil_field_is_divergence_and_curl_free(qa_field, sample_points) -> None:
    jac = np.asarray(grad_B(qa_field, sample_points))
    scale = np.max(np.abs(jac), axis=(-2, -1))
    div = np.trace(jac, axis1=-2, axis2=-1)
    curl = np.stack(
        [jac[:, 2, 1] - jac[:, 1, 2], jac[:, 0, 2] - jac[:, 2, 0], jac[:, 1, 0] - jac[:, 0, 1]], axis=-1
    )
    # ESSOS integrates each coil with the rectangle rule on its Fourier curve,
    # spectrally accurate, so the vacuum identities hold to near round-off.
    assert np.max(np.abs(div) / scale) < 1.0e-8
    assert np.max(np.linalg.norm(curl, axis=-1) / scale) < 1.0e-8


def test_qa_coil_field_has_field_period_and_stellarator_symmetry(qa_field, sample_points) -> None:
    angle = 2.0 * np.pi / qa_field.nfp
    rot = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    b = np.asarray(qa_field.B(sample_points))
    b_rot = np.asarray(qa_field.B(jnp.asarray(np.asarray(sample_points) @ rot.T)))
    np.testing.assert_allclose(b_rot, b @ rot.T, rtol=0.0, atol=1.0e-10 * np.max(np.abs(b)))
    # Stellarator symmetry: (x, y, z) -> (x, -y, -z) maps B to (-Bx, By, Bz).
    flip = np.array([1.0, -1.0, -1.0])
    b_flip = np.asarray(qa_field.B(jnp.asarray(np.asarray(sample_points) * flip)))
    np.testing.assert_allclose(b_flip, b * np.array([-1.0, 1.0, 1.0]), rtol=0.0, atol=1.0e-10 * np.max(np.abs(b)))


def test_qa_coil_field_derivative_with_respect_to_coil_dofs(qa_field, sample_points) -> None:
    coils = qa_field.coils
    direction = jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf) if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf, coils)

    def objective(c):
        return jnp.sum(EssosCoilField(coils=c, nfp=qa_field.nfp, stellsym=qa_field.stellsym).B(sample_points) ** 2)

    _, tangent = jax.jvp(objective, (coils,), (direction,))
    eps = 1.0e-6
    shifted = jax.tree_util.tree_map(lambda a, d: a + eps * d if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a, coils, direction)
    back = jax.tree_util.tree_map(lambda a, d: a - eps * d if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a, coils, direction)
    fd = (objective(shifted) - objective(back)) / (2.0 * eps)
    assert np.isfinite(float(tangent)) and abs(float(tangent)) > 0.0
    assert abs(float(tangent) - float(fd)) < 1.0e-5 * abs(float(fd))


def test_traced_qa_field_lines_are_finite_and_cross_the_poincare_sections() -> None:
    bundle = trace_essos_coil_field_lines(
        r_min=1.21, r_max=1.30, n_field_lines=2, maxtime=60.0, times_to_trace=200, field_sample_count=16
    )
    assert bundle.trajectories_xyz.shape == (2, 200, 3)
    assert np.all(np.isfinite(bundle.trajectories_xyz))
    assert bundle.poincare_point_count > 0
    radius = np.hypot(bundle.trajectories_xyz[..., 0], bundle.trajectories_xyz[..., 1])
    assert 0.5 < radius.min() and radius.max() < 1.6


def test_target_exit_lengths_are_masked_to_fci_endpoint_cells() -> None:
    raw_exit_length = np.array([0.5, 1.5, np.nan, 3.0], dtype=np.float64)
    boundary = np.array([True, False, True, True])
    masked = essos_import._mask_exit_length_to_boundary(raw_exit_length, boundary)
    np.testing.assert_allclose(masked, np.array([0.5, np.nan, np.nan, 3.0]), equal_nan=True)
    with pytest.raises(ValueError, match="shapes must match"):
        essos_import._mask_exit_length_to_boundary(raw_exit_length, boundary[:-1])


@pytest.mark.skipif(not _qa_wout_available(), reason="QA VMEC wout needs DRBX_ESSOS_ROOT")
def test_imported_map_sources_keep_coil_vmec_and_hybrid_semantics() -> None:
    geometries = {
        source: build_essos_imported_fci_geometry(
            map_source=source, nx=3, ny=4, nz=6, rho_min=0.12, rho_max=0.34, maxtime=40.0, times_to_trace=160
        )
        for source in ("coil", "vmec", "hybrid")
    }
    for source, geometry in geometries.items():
        assert geometry.metadata["map_source"] == source
        for name in ("forward_x", "forward_z", "backward_x", "backward_z"):
            assert np.all(np.isfinite(np.asarray(getattr(geometry.maps, name))))
        assert np.all(np.isfinite(np.asarray(geometry.connection_length)))
    coil, vmec, hybrid = geometries["coil"], geometries["vmec"], geometries["hybrid"]
    coil_open = float(np.mean(np.asarray(coil.maps.forward_boundary, dtype=bool)))
    # Coil maps leave the scaled VMEC shell; VMEC maps are closed; hybrid keeps
    # VMEC map coordinates with the coil endpoint masks.
    assert 0.05 < coil_open < 0.95
    assert not np.asarray(vmec.maps.forward_boundary, dtype=bool).any()
    assert float(np.mean(np.asarray(hybrid.maps.forward_boundary, dtype=bool))) == coil_open
    np.testing.assert_allclose(np.asarray(hybrid.maps.forward_x), np.asarray(vmec.maps.forward_x))
    assert np.any(np.isfinite(np.asarray(coil.target_exit_length)))
