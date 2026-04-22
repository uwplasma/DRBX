from __future__ import annotations

import numpy as np
import pytest

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.recycling_boundaries import (
    apply_neutral_target_density_guards,
    apply_open_field_dirichlet_scalar_guards,
    apply_open_field_neumann_scalar_guards,
)


def _guard_mesh() -> StructuredMesh:
    return StructuredMesh(
        nx=1,
        ny=2,
        nz=1,
        mxg=0,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=1,
        jyseps1_2=1,
        jyseps2_2=1,
        ny_inner=2,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0], dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )


def test_apply_neutral_target_density_guards_extrapolates_boundary_density() -> None:
    mesh = _guard_mesh()
    field = np.zeros((1, 6, 1), dtype=np.float64)
    field[0, 2, 0] = 1.5
    field[0, 3, 0] = 0.25

    guarded = apply_neutral_target_density_guards(
        field,
        mesh=mesh,
        lower_y=True,
        upper_y=True,
    )

    assert guarded[0, 1, 0] == pytest.approx(2.75)
    assert guarded[0, 4, 0] == pytest.approx(0.0)


def test_apply_open_field_dirichlet_scalar_guards_reflects_sign() -> None:
    mesh = _guard_mesh()
    field = np.zeros((1, 6, 1), dtype=np.float64)
    field[0, 2, 0] = 3.0
    field[0, 3, 0] = -4.0

    guarded = apply_open_field_dirichlet_scalar_guards(
        field,
        mesh=mesh,
        lower_y=True,
        upper_y=True,
    )

    assert guarded[0, 1, 0] == pytest.approx(-3.0)
    assert guarded[0, 4, 0] == pytest.approx(4.0)


def test_apply_open_field_neumann_scalar_guards_copies_edge_values() -> None:
    mesh = _guard_mesh()
    field = np.zeros((1, 6, 1), dtype=np.float64)
    field[0, 2, 0] = 3.0
    field[0, 3, 0] = -4.0

    guarded = apply_open_field_neumann_scalar_guards(
        field,
        mesh=mesh,
        lower_y=True,
        upper_y=True,
    )

    assert guarded[0, 1, 0] == pytest.approx(3.0)
    assert guarded[0, 4, 0] == pytest.approx(-4.0)
