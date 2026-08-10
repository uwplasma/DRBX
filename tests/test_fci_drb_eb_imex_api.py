"""Focused algebra/API coverage for the local EB IMEX partition.

The spatial equality is exercised by the sharded HSX/ARK integration tests.
These small tests deliberately keep the stage algebra independent of an
expensive geometry construction so regressions in field ordering, source
placement, or the polarization-row sign fail immediately.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_drb_EB_rhs import (
    FciDrbEBImplicitState,
    FciDrbEBState,
    LocalFciDrbEBRhs,
    eb_state_with_implicit_state,
    implicit_state_from_eb_state,
)


def _full(scale: float = 1.0) -> FciDrbEBState:
    values = [jnp.full((2, 2, 2), scale * (index + 1), dtype=jnp.float64) for index in range(7)]
    return FciDrbEBState(*values)


def _implicit(scale: float = 1.0) -> FciDrbEBImplicitState:
    values = [jnp.full((2, 2, 2), scale * (index + 1), dtype=jnp.float64) for index in range(6)]
    return FciDrbEBImplicitState(*values)


def test_implicit_state_round_trip_is_a_jax_pytree() -> None:
    state = _full()
    implicit = implicit_state_from_eb_state(state)
    merged = eb_state_with_implicit_state(state, implicit)
    leaves, structure = jax.tree_util.tree_flatten(implicit)
    assert len(leaves) == 6
    assert structure.num_leaves == 6
    expected_order = ("density", "phi", "Te", "Ti", "Ve", "vorticity")
    expected_values = (1, 2, 3, 4, 6, 7)
    for index, (name, value) in enumerate(zip(expected_order, expected_values)):
        np.testing.assert_array_equal(
            leaves[index],
            jnp.full((2, 2, 2), value, dtype=jnp.float64),
        )
        np.testing.assert_array_equal(getattr(merged, name), getattr(state, name))
    np.testing.assert_array_equal(merged.Vi, state.Vi)


def test_explicit_complement_preserves_the_additive_split(monkeypatch) -> None:
    """The public complement must leave sources in the explicit partition."""

    state = _full(1.0)
    source = _full(0.1)
    implicit = _implicit(0.25)
    full_rhs = _full(2.0)

    # The method under test only requires these two operations.  Keeping this
    # mocked avoids geometry/JAX collectives and directly checks its contract.
    fake = object.__new__(LocalFciDrbEBRhs)
    monkeypatch.setattr(
        LocalFciDrbEBRhs,
        "evaluate_stage",
        lambda self, state_owned, source_owned=None, *, phi_owned=None: full_rhs,
    )
    monkeypatch.setattr(
        LocalFciDrbEBRhs,
        "evaluate_implicit_rhs",
        lambda self, state_owned, *, phi_owned=None: implicit,
    )
    monkeypatch.setattr(
        "drbx.native.fci_drb_EB_rhs._mask_local_eb_state_inactive",
        lambda value, geometry: value,
    )
    object.__setattr__(fake, "geometry", SimpleNamespace())

    explicit = fake.evaluate_explicit_rhs(state, source, phi_owned=state.phi)
    np.testing.assert_allclose(explicit.density + implicit.density, full_rhs.density)
    np.testing.assert_allclose(explicit.Te + implicit.Te, full_rhs.Te)
    np.testing.assert_allclose(explicit.Ti + implicit.Ti, full_rhs.Ti)
    np.testing.assert_allclose(explicit.Ve + implicit.Ve, full_rhs.Ve)
    np.testing.assert_allclose(explicit.vorticity + implicit.vorticity, full_rhs.vorticity)
    np.testing.assert_allclose(explicit.Vi, full_rhs.Vi)


def test_dirk_stage_residual_has_unscaled_algebraic_phi_row(monkeypatch) -> None:
    stage = _implicit(2.0)
    predictor = _implicit(1.0)
    known = _full(3.0)
    implicit_rhs = _implicit(0.5)
    algebraic = jnp.full((2, 2, 2), 17.0, dtype=jnp.float64)
    fake = object.__new__(LocalFciDrbEBRhs)
    object.__setattr__(fake, "geometry", SimpleNamespace())
    monkeypatch.setattr(
        LocalFciDrbEBRhs,
        "evaluate_implicit_rhs",
        lambda self, state_owned, *, phi_owned=None: implicit_rhs,
    )
    monkeypatch.setattr(
        LocalFciDrbEBRhs,
        "polarization_residual",
        lambda self, state_owned, *, phi_owned=None: algebraic,
    )
    monkeypatch.setattr(
        "drbx.native.fci_drb_EB_rhs._mask_local_eb_state_inactive",
        lambda value, geometry: value,
    )

    residual = fake.implicit_stage_residual(
        stage, predictor, known, dt_gamma=0.125
    )
    np.testing.assert_allclose(
        residual.density,
        stage.density - predictor.density - 0.125 * implicit_rhs.density,
    )
    np.testing.assert_allclose(
        residual.Ti,
        stage.Ti - predictor.Ti - 0.125 * implicit_rhs.Ti,
    )
    np.testing.assert_allclose(residual.phi, algebraic)
    np.testing.assert_allclose(
        residual.vorticity,
        stage.vorticity - predictor.vorticity - 0.125 * implicit_rhs.vorticity,
    )
