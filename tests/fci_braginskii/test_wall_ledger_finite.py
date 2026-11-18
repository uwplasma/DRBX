"""Compiled time step of the HSX backend on the 8x8x8 test bundle: finiteness, equilibrium, restart, closed box.

The bundle is ``tests/data/hsx_8x8x8`` (set DRBX_TEST_GEOMETRY_BUNDLE to use a larger one).  The runs are shared
through the ``hsx_run`` fixture, so each test below costs one assertion, not one compile.
"""

import numpy as np
import pytest

FIELDS = ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")


def test_hsx_run_stays_finite_and_its_wall_ledger_is_finite(hsx_run):
    run = hsx_run()
    assert run["times"].shape == (5,) and run["times"][-1] == pytest.approx(0.003)
    for name in FIELDS:
        assert np.all(np.isfinite(run[name])), name
    assert run["density"].min() > 0.0 and run["Te"].min() > 0.0 and run["Ti"].min() > 0.0
    ledger = run["wall_ledger"]
    assert ledger.shape[0] == 5 and np.all(np.isfinite(ledger))
    assert not np.any(ledger[0, 19:])  # no step before the first row
    # the Bohm wall removes particles: the inventory falls monotonically
    assert np.all(np.diff(run["columns"]["particle_inventory"]) < 0.0)


def test_hsx_restart_from_a_saved_frame_reproduces_the_continuous_run(hsx_run):
    run = hsx_run()
    restarted = hsx_run(f'restart_from = "{run["path"]}"\nrestart_frame = 2\n', final_time=0.003, num_steps=2)
    np.testing.assert_allclose(restarted["times"], run["times"][2:], rtol=0, atol=1e-15)
    # the histories are stored in float32, so the restart starts from the rounded frame; Ve, the smallest field, shows it most
    for name in FIELDS:
        scale = max(float(np.max(np.abs(run[name][4]))), 1e-300)
        assert np.max(np.abs(restarted[name][-1] - run[name][4])) <= 5e-5 * scale, name  # measured 1.2e-5 (Ve)


def test_hsx_uniform_state_without_a_filament_stays_at_equilibrium(hsx_run):
    # no density filament, so uniform n and T at rest; the zero-flow wall has no flux, the potential is zero
    run = hsx_run('density_amplitude = 0.0\nwall_closure = "zero-flow"\n', final_time=0.0015, num_steps=2)
    for name in ("density", "Te", "Ti"):
        np.testing.assert_allclose(run[name][-1], run[name][0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(run[name][0], 1.0, rtol=0, atol=1e-14)
    for name in ("phi", "Vi", "Ve", "vorticity"):
        assert np.max(np.abs(run[name])) <= 1e-14, name


def test_hsx_zero_flow_wall_conserves_particles(hsx_run):
    run = hsx_run('wall_closure = "zero-flow"\n')
    inventory = run["columns"]["particle_inventory"]
    # measured 1.5e-9: the perpendicular edge flux and the GMRES tolerance, not a wall flux
    assert abs(inventory[-1] - inventory[0]) <= 1e-8 * inventory[0]
