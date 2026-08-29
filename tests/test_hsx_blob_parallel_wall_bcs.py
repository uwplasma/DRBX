from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from drbx.geometry import HaloLayout3D
from drbx.native import FciDrbEBState
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


def _inputs():
    shape = (3, 4, 2)
    layout = HaloLayout3D(shape, 2)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    state = FciDrbEBState(
        density=jnp.ones(shape, dtype=jnp.float64),
        phi=zeros,
        Te=jnp.full(shape, 4.0, dtype=jnp.float64),
        Ti=jnp.ones(shape, dtype=jnp.float64),
        Vi=zeros,
        Ve=zeros,
        vorticity=zeros,
    )

    def face_bfield(axis):
        face_shape = list(shape)
        face_shape[axis] += 1
        B = jnp.zeros(tuple(face_shape) + (3,), dtype=jnp.float64)
        return SimpleNamespace(B_contra_owned=B)

    x = face_bfield(0)
    upper_sign = jnp.where(
        jnp.indices(shape[1:]).sum(axis=0) % 2 == 0,
        1.0,
        -1.0,
    )
    x.B_contra_owned = x.B_contra_owned.at[-1, ..., 0].set(upper_sign)
    geometry = SimpleNamespace(
        layout=layout,
        face_bfield=SimpleNamespace(
            axes=(x, face_bfield(1), face_bfield(2))
        ),
    )
    domain = SimpleNamespace(
        runtime_has_physical_lower=lambda axis: False,
        runtime_has_physical_upper=lambda axis: axis == 0,
    )
    parameters = SimpleNamespace(tau=2.0)
    return state, geometry, domain, parameters, upper_sign


def test_velocity_neumann_mode_extrapolates_vi_and_ve():
    driver = _load_driver()
    state, geometry, domain, parameters, _ = _inputs()
    bundle = driver.build_face_bc_bundle(
        state,
        geometry,
        domain,
        parameters,
        parallel_velocity_wall_bc="neumann",
    )
    assert np.all(np.asarray(bundle.Vi.kind_x[-1]) == BC_NEUMANN)
    assert np.all(np.asarray(bundle.Ve.kind_x[-1]) == BC_NEUMANN)
    assert np.all(np.asarray(bundle.phi.kind_x[-1]) == BC_DIRICHLET)
    assert np.all(np.asarray(bundle.vorticity.kind_x[-1]) == BC_DIRICHLET)


def test_bohm_mode_sets_outward_zero_current_velocity():
    driver = _load_driver()
    state, geometry, domain, parameters, upper_sign = _inputs()
    bundle = driver.build_face_bc_bundle(
        state,
        geometry,
        domain,
        parameters,
        parallel_velocity_wall_bc="bohm",
    )
    expected = np.asarray(upper_sign) * np.sqrt(4.0 + 2.0)
    np.testing.assert_allclose(bundle.Vi.value_x[-1], expected)
    np.testing.assert_allclose(bundle.Ve.value_x[-1], expected)
    assert np.all(np.asarray(bundle.Vi.kind_x[-1]) == BC_DIRICHLET)
    np.testing.assert_allclose(
        bundle.Vi.value_x[-1] - bundle.Ve.value_x[-1],
        0.0,
    )


def test_default_parallel_velocity_bc_is_neumann():
    driver = _load_driver()
    state, geometry, domain, parameters, _ = _inputs()
    bundle = driver.build_face_bc_bundle(
        state, geometry, domain, parameters
    )
    assert np.all(np.asarray(bundle.Vi.kind_x[-1]) == BC_NEUMANN)
    assert np.all(np.asarray(bundle.Ve.kind_x[-1]) == BC_NEUMANN)
