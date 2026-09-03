from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from drbx.geometry import HaloLayout3D
from drbx.native import FciDrbEBState, LocalFciDrbEBPhysicalWallBundle
from drbx.native.fci_boundaries import BC_DIRICHLET, BC_NEUMANN


DRIVER = Path(__file__).parents[1] / "simulate_hsx_blob.py"


def _load_driver():
    name = "_test_simulate_hsx_blob_parallel_wall_bcs"
    spec = importlib.util.spec_from_file_location(name, DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(phi=0.0):
    shape = (3, 4, 2)
    layout = HaloLayout3D(shape, 2)
    state = FciDrbEBState(
        density=jnp.ones(shape, dtype=jnp.float64),
        phi=jnp.full(shape, phi, dtype=jnp.float64),
        Te=jnp.full(shape, 4.0, dtype=jnp.float64),
        Ti=jnp.ones(shape, dtype=jnp.float64),
        Vi=jnp.zeros(shape, dtype=jnp.float64),
        Ve=jnp.zeros(shape, dtype=jnp.float64),
        vorticity=jnp.zeros(shape, dtype=jnp.float64),
    )

    def face_bfield(axis):
        face_shape = list(shape)
        face_shape[axis] += 1
        return SimpleNamespace(
            B_contra_owned=jnp.zeros(tuple(face_shape) + (3,), dtype=jnp.float64)
        )

    x = face_bfield(0)
    signs = jnp.where(jnp.indices(shape[1:]).sum(axis=0) % 2 == 0, 1.0, -1.0)
    x.B_contra_owned = x.B_contra_owned.at[-1, ..., 0].set(signs)
    geometry = SimpleNamespace(
        layout=layout,
        face_bfield=SimpleNamespace(axes=(x, face_bfield(1), face_bfield(2))),
    )
    domain = SimpleNamespace(
        runtime_has_physical_lower=lambda axis: False,
        runtime_has_physical_upper=lambda axis: axis == 0,
    )
    parameters = SimpleNamespace(tau=2.0, Te0=1.0, Ti0=1.0, mi_over_me=1836.0)
    return state, geometry, domain, parameters, signs


def test_legacy_velocity_trace_remains_neumann_by_default():
    driver = _load_driver()
    state, geometry, domain, parameters, _ = _inputs()
    bundle = driver.build_face_bc_bundle(state, geometry, domain, parameters)
    assert isinstance(bundle, LocalFciDrbEBPhysicalWallBundle)
    assert np.all(np.asarray(bundle.Vi.kind_x[-1]) == BC_NEUMANN)
    assert np.all(np.asarray(bundle.Ve.kind_x[-1]) == BC_NEUMANN)
    assert np.all(np.asarray(bundle.phi.kind_x[-1]) == BC_DIRICHLET)


def test_no_flow_model_supplies_zero_velocity_face_trace():
    driver = _load_driver()
    state, geometry, domain, parameters, _ = _inputs()
    bundle = driver.build_face_bc_bundle(
        state, geometry, domain, parameters, physical_wall_model="no-flow"
    )
    assert np.all(np.asarray(bundle.Vi.kind_x[-1]) == BC_DIRICHLET)
    assert np.all(np.asarray(bundle.Ve.kind_x[-1]) == BC_DIRICHLET)
    np.testing.assert_array_equal(bundle.Vi.value_x[-1], 0.0)
    np.testing.assert_array_equal(bundle.Ve.value_x[-1], 0.0)
    assert np.all(np.asarray(bundle.density.kind_x[-1]) == BC_NEUMANN)
    assert np.all(np.asarray(bundle.Te.kind_x[-1]) == BC_NEUMANN)
    assert np.all(np.asarray(bundle.Ti.kind_x[-1]) == BC_NEUMANN)


def test_simple_conducting_sheath_supplies_warm_ion_and_electron_targets():
    driver = _load_driver()
    state, geometry, domain, parameters, signs = _inputs()
    bundle = driver.build_face_bc_bundle(
        state,
        geometry,
        domain,
        parameters,
        physical_wall_model="simple-conducting-sheath",
        conducting_sheath_wall_potential=0.0,
    )
    ion = np.asarray(bundle.Vi.value_x[-1])
    electron = np.asarray(bundle.Ve.value_x[-1])
    np.testing.assert_allclose(ion, np.asarray(signs) * np.sqrt(6.0))
    np.testing.assert_allclose(
        electron,
        np.asarray(signs) * np.sqrt(1836.0 * 4.0 / (2.0 * np.pi)),
    )
    assert np.all(np.asarray(bundle.Vi.kind_x[-1]) == BC_DIRICHLET)
    assert np.all(np.asarray(bundle.Ve.kind_x[-1]) == BC_DIRICHLET)


def test_simple_conducting_sheath_uses_owner_potential_and_optional_wall_potential():
    driver = _load_driver()
    state, geometry, domain, parameters, signs = _inputs(phi=2.0)
    bundle = driver.build_face_bc_bundle(
        state,
        geometry,
        domain,
        parameters,
        physical_wall_model="simple-conducting-sheath",
        conducting_sheath_wall_potential=1.0,
    )
    expected = (
        np.asarray(signs)
        * np.sqrt(1836.0 * 4.0 / (2.0 * np.pi))
        * np.exp(-0.25)
    )
    np.testing.assert_allclose(bundle.Ve.value_x[-1], expected)


def test_unknown_physical_wall_model_is_rejected():
    driver = _load_driver()
    state, geometry, domain, parameters, _ = _inputs()
    try:
        driver.build_face_bc_bundle(
            state,
            geometry,
            domain,
            parameters,
            physical_wall_model="resolved-physical-wall",
        )
    except ValueError as exc:
        assert "physical_wall_model must be one of" in str(exc)
    else:
        raise AssertionError("obsolete resolver selector was accepted")
