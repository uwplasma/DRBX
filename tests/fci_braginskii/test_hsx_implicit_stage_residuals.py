"""Real-HSX residuals of the implicit current/phi stage (set DRBX_TEST_GEOMETRY_BUNDLE)."""

import os

import jax
import pytest

from drbx.cli import main
from drbx.fci_braginskii.native.fci_drb_EB_rhs import LocalFciDrbEBRhs


@pytest.mark.slow
@pytest.mark.parametrize("preconditioner", ["line-u", "line-u-pardiag", "line-uy-pardiag"])
def test_hsx_stage_residuals_from_returned_state(tmp_path, monkeypatch, preconditioner):
    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to replay a real HSX artifact")
    records = []
    solve = LocalFciDrbEBRhs.solve_implicit_current_phi_pair

    def recording_solve(self, state_owned, *, solve_dt):
        *result, residuals = solve(
            self, state_owned, solve_dt=solve_dt, return_residuals=True
        )
        jax.debug.callback(lambda **r: records.append(r), **residuals)
        return tuple(result)

    monkeypatch.setattr(LocalFciDrbEBRhs, "solve_implicit_current_phi_pair", recording_solve)
    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
        f'geometry = "{bundle}"\noutput = "{tmp_path / "history"}"\nfinal_time = 0.0015\nnum_steps = 2\n'
        f'gmres_preconditioner = "{preconditioner}"\n'
    )
    main(["run", str(deck)])
    assert len(records) >= 4
    for r in records:
        scale = max(float(r["omega_norm"]), 1.0e-300)
        assert float(r["continuity"]) <= 1.0e-12 * scale
        assert float(r["polarization"]) <= float(r["acceptance"])
