"""Focused tests for augmented-polarization multiplier handling."""

from types import SimpleNamespace

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

import drbx.native.fci_drb_EB_rhs as rhs_module
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs, FciDrbEBState
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_operators import LocalPerpLaplacianInverseSolver
from drbx.native.fci_sharding import assemble_local_fci_geometry

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fci_drb_eb_test_helpers import _build_rhs, _context_and_sharded_inputs
from test_mms_shifted_torus_EB_sharded import _build_face_bcs


class _WeightsSolver:
    def __init__(self, weights):
        self.weights = jnp.asarray(weights)

    def _operator_mass_weights(self):
        return jnp.ones_like(self.weights, dtype=bool), self.weights


class _FakeModel:
    def __init__(self, *, simplified=True):
        self.physical_wall_model_name = (
            "simplified-gbs-mpe" if simplified else "simple-conducting-sheath"
        )
        self.parameters = SimpleNamespace(tau=2.0)
        self.domain = SimpleNamespace(mesh_axis_names=())
        self.gmres_config = SolvaxGmresConfig(regularization_epsilon=0.0)

    def _polarization_solver(self, face_bc, config=None):
        del face_bc, config
        return _WeightsSolver([1.0, 2.0, 1.0])

    _positive_polarization_action = lambda self, solver, values, face_bc: (
        jnp.asarray(values) + 1.0
    )

    recover_polarization_multiplier = LocalFciDrbEBRhs.recover_polarization_multiplier
    _vorticity_from_polarization = LocalFciDrbEBRhs._vorticity_from_polarization


def _state(values):
    zeros = jnp.zeros((3,))
    return FciDrbEBState(
        density=zeros,
        phi=zeros,
        Te=zeros,
        Ti=jnp.asarray(values["Ti"], dtype=jnp.float64),
        Vi=zeros,
        Ve=zeros,
        vorticity=jnp.asarray(values["omega"], dtype=jnp.float64),
    )


def test_recover_polarization_multiplier_uses_weighted_raw_residual(monkeypatch):
    model = _FakeModel()
    state = _state({"Ti": [1.0, 2.0, 3.0], "omega": [4.0, 5.0, 6.0]})
    face_bc = SimpleNamespace(phi=object(), Ti=object())
    monkeypatch.setattr(rhs_module, "_spmd_sum", lambda value, domain: value)
    lam = model.recover_polarization_multiplier(
        state, phi_owned=jnp.asarray([2.0, 3.0, 4.0]), face_bc=face_bc
    )
    raw = (jnp.asarray([3.0, 4.0, 5.0])
           + 2.0 * jnp.asarray([2.0, 3.0, 4.0])
           + jnp.asarray([4.0, 5.0, 6.0]))
    expected = -float(jnp.dot(jnp.asarray([1.0, 2.0, 1.0]), raw) / 4.0)
    np.testing.assert_allclose(lam, expected)


def test_non_augmented_model_has_zero_polarization_multiplier():
    model = _FakeModel(simplified=False)
    state = _state({"Ti": [1.0, 2.0, 3.0], "omega": [4.0, 5.0, 6.0]})
    assert float(model.recover_polarization_multiplier(state)) == 0.0


def test_raw_polarization_image_is_not_multiplier_corrected():
    model = _FakeModel()
    model.parameters = SimpleNamespace(tau=2.0)
    state = _state({"Ti": [1.0, 2.0, 3.0], "omega": [4.0, 5.0, 6.0]})
    face_bc = SimpleNamespace(phi=object(), Ti=object())
    raw = model._vorticity_from_polarization(
        state.phi, state.Ti, face_bc.phi, face_bc.Ti
    )
    np.testing.assert_allclose(raw, -state.phi - 1.0 - 2.0 * (state.Ti + 1.0))


def test_real_small_geometry_derived_face_trace_applies_only_mpe_multiplier(monkeypatch):
    """A manufactured raw image shifts by exactly the supplied MPE multiplier."""
    with jax.disable_jit(False):
        context, mesh, local, partition, fields, cell_fields = (
            _context_and_sharded_inputs()
        )

    def fake_apply(
        self,
        values_owned,
        *,
        face_bc=None,
        control_volume_boundary_bc=None,
        project_mean_zero=False,
    ):
        del self, face_bc, control_volume_boundary_bc, project_mean_zero
        return jnp.asarray(values_owned, dtype=jnp.float64)

    monkeypatch.setattr(
        LocalPerpLaplacianInverseSolver, "apply_positive_operator", fake_apply
    )

    def face_builder(state, geometry, domain, parameters, **kwargs):
        del kwargs
        return _build_face_bcs(state, geometry, domain, parameters)

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells):
        geometry = assemble_local_fci_geometry(local, cells)
        model = replace(
            _build_rhs(context, local, geometry),
            physical_wall_model_name="simplified-gbs-mpe",
            polarization_operator_form="weighted-symmetric",
            face_bc_builder=face_builder,
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        face_bc = model._face_bcs(state)
        raw_bc = model._derived_vorticity_face_bc_from_polarization(
            state, state.phi, face_bc, polarization_multiplier=0.0
        )
        shifted_bc = model._derived_vorticity_face_bc_from_polarization(
            state, state.phi, face_bc, polarization_multiplier=2.0
        )
        errors = []
        for mask, raw, shifted in (
            (raw_bc.mask_x, raw_bc.value_x, shifted_bc.value_x),
            (raw_bc.mask_y, raw_bc.value_y, shifted_bc.value_y),
            (raw_bc.mask_z, raw_bc.value_z, shifted_bc.value_z),
        ):
            errors.append(jnp.max(jnp.abs(jnp.where(mask, shifted - raw + 2.0, 0.0))))
        return jax.lax.pmin(
            jnp.max(jnp.stack(errors)),
            axis_name=("x", "y", "z"),
        )

    mapped = shard_map(
        kernel,
        mesh=mesh,
        in_specs=(*((partition,) * 7), partition),
        out_specs=P(),
    )
    np.testing.assert_allclose(mapped(*fields, cell_fields), 0.0, rtol=0.0, atol=1e-12)
