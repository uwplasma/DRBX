"""Real-HSX residuals of the implicit current/phi stage (set DRBX_TEST_GEOMETRY_BUNDLE)."""

import os

import jax
import pytest

import simulate_hsx_blob as driver
from drbx.fci_braginskii.native.fci_drb_EB_rhs import LocalFciDrbEBRhs


@pytest.mark.slow
@pytest.mark.parametrize("preconditioner", ["line-u", "line-u-pardiag", "line-uy-pardiag"])
def test_hsx_stage_residuals_from_returned_state(tmp_path, monkeypatch, preconditioner):
    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to replay a real HSX artifact")
    import drbx.fci_braginskii.run as run_module

    monkeypatch.setattr(run_module, "GMRES_PRECONDITIONER", preconditioner)
    records = []
    solve = LocalFciDrbEBRhs.solve_implicit_current_phi_pair

    def recording_solve(self, state_owned, *, solve_dt):
        *result, residuals = solve(
            self, state_owned, solve_dt=solve_dt, return_residuals=True
        )
        jax.debug.callback(lambda **r: records.append(r), **residuals)
        return tuple(result)

    monkeypatch.setattr(LocalFciDrbEBRhs, "solve_implicit_current_phi_pair", recording_solve)
    driver.main([
        "--geometry", bundle, "--final-time", "0.0015", "--num-steps", "2",
        "--output", str(tmp_path / "history.npz"),
    ])
    assert len(records) >= 4
    for r in records:
        scale = max(float(r["omega_norm"]), 1.0e-300)
        assert float(r["continuity"]) <= 1.0e-12 * scale
        assert float(r["polarization"]) <= float(r["acceptance"])
