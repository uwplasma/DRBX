"""Focused tests for the native boundary-compatible startup helper."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import SIDE_AXIS_REGULAR
from drbx.native.fci_drb_EB_rhs import FciDrbEBState
from drbx.native.fci_halo import (
    neumann_face_trace_physical_affine,
    physical_normal_derivative_for_face_trace,
)
from drbx.native.fci_initialization import (
    BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES,
    boundary_compatibility_acceptance_failures,
    initialize_boundary_compatible_rung3,
    quiet_outer_trace_patch,
    validate_boundary_compatible_initialization_support,
)
from test_neumann_face_trace_affine import _fixture, _halo


def _patch_model():
    layout, domain, geometry, _filler = _fixture(2, shape=(12, 5, 3))
    return SimpleNamespace(
        geometry=geometry,
        domain=domain,
        halo_exchange=lambda value, _domain: value,
        topology_filler=lambda value, _domain: _halo(
            value[layout.owned_slices_cell], layout
        ),
    ), layout, domain, geometry


def test_quiet_outer_trace_patch_matches_skew_metric_trace_and_jit():
    model, layout, domain, geometry = _patch_model()
    x = jnp.arange(layout.owned_shape[0], dtype=jnp.float64)[:, None, None]
    y = jnp.arange(layout.owned_shape[1], dtype=jnp.float64)[None, :, None]
    z = jnp.arange(layout.owned_shape[2], dtype=jnp.float64)[None, None, :]
    field = jnp.broadcast_to(
        1.0 + 0.02 * x + 0.01 * jnp.sin(y) + 0.005 * jnp.cos(z),
        layout.owned_shape,
    )
    target = 1.2 + 0.05 * jnp.sin(y[0]) + 0.03 * jnp.cos(z[0])
    eager = quiet_outer_trace_patch(model, field, target, layer_count=4)
    compiled = jax.jit(
        lambda value, face: quiet_outer_trace_patch(
            model, value, face, layer_count=4
        )
    )(field, target)
    np.testing.assert_allclose(compiled, eager, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(compiled[-1], target, rtol=0.0, atol=1.0e-11)
    trace, _ = neumann_face_trace_physical_affine(
        _halo(compiled, layout), geometry, domain, 0, "upper"
    )
    np.testing.assert_allclose(trace, target, rtol=0.0, atol=1.0e-11)
    np.testing.assert_array_equal(compiled[:8], field[:8])


class _SmallWallModel:
    """Small closure using the real skew-metric face-trace operators."""

    def __init__(self):
        layout, raw_domain, geometry, _filler = _fixture(2, shape=(12, 5, 3))
        shard_spec = replace(
            raw_domain.shard_spec,
            side_kind_lower=(
                SIDE_AXIS_REGULAR,
                raw_domain.shard_spec.side_kind_lower[1],
                raw_domain.shard_spec.side_kind_lower[2],
            ),
        )
        self.domain = replace(raw_domain, shard_spec=shard_spec)
        self.geometry = geometry
        self.control_volume_geometry = None
        self.physical_wall_model_name = "simplified-gbs-mpe"
        self.parameters = SimpleNamespace(tau=1.7)
        self.halo_exchange = lambda value, _domain: value
        self.topology_filler = lambda value, _domain: _halo(
            value[layout.owned_slices_cell], layout
        )
        object.__setattr__(
            self.geometry,
            "active_cell_mask",
            jnp.ones(layout.owned_shape, dtype=bool),
        )
        face_b = jnp.zeros(layout.face_control_shape(0) + (3,), dtype=jnp.float64)
        face_b = face_b.at[..., 0].set(1.0)
        object.__setattr__(
            self.geometry,
            "face_bfield",
            SimpleNamespace(axes=(SimpleNamespace(B_contra_owned=face_b),)),
        )

    def _owner_state(self, state):
        return state

    def _materialized_wall_state(self, state):
        return state

    def _topology(self, value):
        halo = self.halo_exchange(
            jnp.pad(value, ((2, 2), (2, 2), (2, 2))), self.domain
        )
        return self.topology_filler(halo, self.domain)

    def _upper_trace(self, value):
        return neumann_face_trace_physical_affine(
            self._topology(value), self.geometry, self.domain, 0, "upper"
        )[0]

    def _face_field(self, upper):
        shape = self.domain.layout.face_control_shape(0)
        return SimpleNamespace(
            value_x=jnp.zeros(shape, dtype=jnp.float64).at[-1].set(upper)
        )

    def _face_bcs(self, state):
        te_face = self._upper_trace(state.Te)
        ti_face = self._upper_trace(state.Ti)
        cb = jnp.sqrt(te_face + self.parameters.tau * ti_face)
        vi_target = cb
        gvi = physical_normal_derivative_for_face_trace(
            self._topology(state.Vi),
            vi_target,
            self.geometry,
            self.domain,
            0,
            "upper",
        )
        gphi = -te_face / cb * gvi
        phi_face = self._upper_trace(state.phi)
        ve_target = vi_target + 0.2 * phi_face - 0.1 * te_face
        return SimpleNamespace(
            density=self._face_field(jnp.zeros_like(te_face)),
            phi=self._face_field(gphi),
            Te=self._face_field(jnp.zeros_like(te_face)),
            Ti=self._face_field(jnp.zeros_like(te_face)),
            Vi=self._face_field(vi_target),
            Ve=self._face_field(ve_target),
        )

    def _simplified_gbs_mpe_phi_gauge_data(self, state, face_bc):
        del state, face_bc
        shape = self.domain.layout.owned_shape
        weights = jnp.zeros(shape, dtype=jnp.float64)
        weights = weights.at[-1].set(1.0 / float(shape[1] * shape[2]))
        return weights, jnp.asarray(0.0), jnp.asarray(0.25)

    def _vorticity_from_polarization(self, phi, ti, face_phi, face_ti):
        del face_phi, face_ti
        defect = self._upper_trace(phi) - phi[-1]
        defect = defect + self.parameters.tau * (self._upper_trace(ti) - ti[-1])
        return jnp.zeros_like(phi).at[-1].set(defect)


def _small_state(shape):
    i, j, k = jnp.indices(shape, dtype=jnp.float64)
    return FciDrbEBState(
        density=1.1 + 0.01 * i + 0.015 * j,
        phi=-0.4 + 0.03 * i + 0.02 * j - 0.01 * k,
        Te=2.0 + 0.02 * i + 0.01 * k,
        Ti=1.3 + 0.01 * i + 0.005 * j,
        Vi=-0.15 + 0.01 * i,
        Ve=0.3 - 0.02 * i + 0.01 * k,
        vorticity=jnp.full(shape, 9.0),
    )


def test_boundary_compatible_initializer_preserves_core_and_passes_wall_contracts():
    model = _SmallWallModel()
    state = _small_state(model.domain.layout.owned_shape)
    initialize = jax.jit(
        lambda value: initialize_boundary_compatible_rung3(
            model, value, layer_count=4
        )
    )
    result, diagnostics = initialize(state)
    diagnostics_host = np.asarray(diagnostics)
    assert diagnostics_host.shape == (len(BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES),)
    assert boundary_compatibility_acceptance_failures(diagnostics_host) == ()

    named = dict(
        zip(BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES, diagnostics_host, strict=True)
    )
    assert named["minimum_density"] > 0.0
    assert named["minimum_Te"] > 0.0
    assert named["minimum_Ti"] > 0.0
    assert named["phi_normal_bc_max"] < 1.0e-10
    assert named["Vi_wall_target_defect_max"] < 1.0e-10
    assert named["Ve_wall_target_defect_max"] < 1.0e-10
    assert named["vorticity_abs_max"] < 1.0e-10
    assert named["gauge_residual_abs"] < 1.0e-12

    shift = named["phi_gauge_shift"]
    for name in ("density", "Te", "Ti", "Vi", "Ve"):
        np.testing.assert_array_equal(
            np.asarray(getattr(result, name)[:8]),
            np.asarray(getattr(state, name)[:8]),
        )
    np.testing.assert_allclose(
        np.asarray(result.phi[:8]),
        np.asarray(state.phi[:8]) + shift,
        rtol=0.0,
        atol=1.0e-12,
    )
    for name in ("density", "Te", "Ti"):
        np.testing.assert_allclose(
            np.asarray(getattr(result, name)[-1]),
            np.asarray(getattr(state, name)[-1]),
            rtol=0.0,
            atol=1.0e-12,
        )


def test_initializer_rejects_halo_too_narrow_for_penultimate_response():
    _layout, domain, _geometry, _filler = _fixture(1, shape=(8, 4, 3))
    with pytest.raises(ValueError, match="halo_width >= 2"):
        validate_boundary_compatible_initialization_support(domain, 3)


def test_host_acceptance_allows_nonzero_derived_vorticity():
    diagnostics = np.zeros(
        len(BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES), dtype=np.float64
    )
    named_index = {
        name: index
        for index, name in enumerate(BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES)
    }
    diagnostics[named_index["minimum_density"]] = 1.0
    diagnostics[named_index["minimum_Te"]] = 2.0
    diagnostics[named_index["minimum_Ti"]] = 1.5
    diagnostics[named_index["gauge_constant_response_abs"]] = 1.0
    diagnostics[named_index["trace_response_min_abs"]] = 0.1
    diagnostics[named_index["vorticity_abs_max"]] = 42.0
    assert boundary_compatibility_acceptance_failures(diagnostics) == ()
