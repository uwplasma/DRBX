"""The wall ledger of the HSX driver is finite in every row, including the initial one (no step applied: zeros)."""

import os

import numpy as np
import pytest


@pytest.mark.slow
def test_hsx_wall_ledger_is_finite(tmp_path):
    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to run the HSX wall ledger")
    from drbx.cli import main

    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
        f'geometry = "{bundle}"\noutput = "{tmp_path / "hsx"}"\nfinal_time = 0.00075\nnum_steps = 1\n'
    )
    assert main(["run", str(deck)]) in (0, None)
    ledger = np.load(tmp_path / "hsx.npz")["wall_ledger"]
    assert ledger.shape[0] == 2 and np.all(np.isfinite(ledger))
    assert not np.any(ledger[0, 19:])  # no step before the first row
