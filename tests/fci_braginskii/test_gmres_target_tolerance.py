"""The inner GMRES target follows the time step unless the deck sets it."""

import pytest

from drbx.fci_braginskii.run import GMRES_TARGET_TOLERANCE, resolve_gmres_target_tolerance as resolve


def test_explicit_target_is_used_as_given():
    assert resolve(3.0e-9, 5.0e-5, 7.5e-4) == 3.0e-9
    assert resolve(1.0e-2, 5.0e-5, 7.5e-4) == 1.0e-2  # the deck owner's choice


def test_auto_target_follows_dt_squared_between_the_floor_and_the_acceptance_cap():
    assert resolve(0.0, 5.0e-5, 7.5e-4) == pytest.approx(5.625e-6)
    assert resolve(0.0, 5.0e-5, 1.0e-2) == pytest.approx(1.0e-5)  # capped at acceptance / 5
    assert resolve(0.0, 5.0e-5, 1.0e-5) == GMRES_TARGET_TOLERANCE  # floored
    assert resolve(0.0, 0.0, 7.5e-4) == GMRES_TARGET_TOLERANCE  # zero acceptance: strict
