"""Regression for splitting affine phi/Ti polarization actions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drbx.native.fci_boundaries import BC_NEUMANN  # noqa: E402
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs  # noqa: E402
from test_fci_projected_fine_grid_control_volume import _setup  # noqa: E402


def test_vorticity_polarization_splits_support_paired_affine_boundary_actions():
    """Each field's affine face data must be applied by its own action."""

    # The (3, 8, 4) projected fixture is small but exercises the actual RLP
    # support-paired implementation, including physical radial face closure.
    data = _setup(shape=(3, 8, 4))
    solver_base = data[8]
    active = data[9]
    solver = replace(solver_base, operator_form="support-paired")
    base_bc = data[7]
    phi_bc = replace(
        base_bc,
        kind_x=base_bc.kind_x.at[-1].set(BC_NEUMANN),
        value_x=base_bc.value_x.at[-1].set(0.13),
    )
    ti_bc = replace(
        base_bc,
        kind_x=base_bc.kind_x.at[-1].set(BC_NEUMANN),
        value_x=base_bc.value_x.at[-1].set(-0.21),
    )

    rng = np.random.default_rng(20260905)
    phi = np.zeros(active.shape, dtype=float)
    phi[active] = rng.normal(size=np.count_nonzero(active))
    # A constant Ti makes its nonzero Neumann affine payload unambiguous.
    ti = np.zeros(active.shape, dtype=float)
    ti[active] = 1.0
    tau = 2.3

    def polarization_solver(_face_bc, *, config=None):
        del config
        return solver

    harness = SimpleNamespace(
        parameters=SimpleNamespace(tau=tau),
        gmres_config=solver.config,
        physical_wall_model_name="simplified-gbs-mpe",
        _polarization_solver=polarization_solver,
        _positive_polarization_action=(
            lambda selected, values, face_bc: selected.apply_positive_operator(
                jnp.asarray(values, dtype=jnp.float64),
                face_bc=face_bc,
                control_volume_boundary_bc=data[6],
                project_mean_zero=False,
            )
        ),
    )

    actual = np.asarray(
        LocalFciDrbEBRhs._vorticity_from_polarization(
            harness, phi, ti, phi_bc, ti_bc
        )
    )
    phi_action = np.asarray(
        solver.apply_positive_operator(
            jnp.asarray(phi),
            face_bc=phi_bc,
            control_volume_boundary_bc=data[6],
            project_mean_zero=False,
        )
    )
    ti_action = np.asarray(
        solver.apply_positive_operator(
            jnp.asarray(ti),
            face_bc=ti_bc,
            control_volume_boundary_bc=data[6],
            project_mean_zero=False,
        )
    )
    expected = -phi_action - tau * ti_action
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-11)

    # Historical behavior combined fields and boundary payloads before one
    # action.  Distinct affine data must make that result measurably differ.
    # This is the historical expression: the combined field was closed with
    # phi's face payload, silently discarding Ti's distinct affine data.
    combined_bc = phi_bc
    combined = np.asarray(
        solver.apply_positive_operator(
            jnp.asarray(phi + tau * ti),
            face_bc=combined_bc,
            control_volume_boundary_bc=data[6],
            project_mean_zero=False,
        )
    )
    old_combined = -combined
    difference = np.max(np.abs(actual[active] - old_combined[active]))
    assert difference > 1.0e-7
