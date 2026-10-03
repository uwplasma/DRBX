"""P06N owner values computed on the fly (test helper for the slow real-closure tests).

The frozen P06N oracle files ``<p05n_p06n_upwind>/p06n/N{n}.owner_values.npz`` are no longer kept in the workspace.
They were produced by the P06N ``live_observations`` routine: the raw-midpoint, raw-volume-weighted average of every
catalogue field over the raw cells of each owner. ``p08_step6_global.fields.owner_average_chunk`` is the chunked,
bit-identical form of that routine (it reproduced the frozen N32 file to 3.3e-16, see
``scripts/p08_step6_global/README.md``, "Owner averages"); this module applies it to the P06N catalogue
(``p06n_field_derived_global.core.NAMES`` evaluated by ``p06n_fields.evaluate`` on ``env.ref``) and exposes a drop-in for
``p_shared.replay_units._load_oracle_owner_values``.

* :func:`p06n_owner_values` -- ``(n_owners, n_catalogue_fields)`` array, cached per grid within the process.
* :func:`load_oracle_owner_values` -- ``ru._load_oracle_owner_values`` with the ``"p06n"`` entry computed when the frozen
  file is absent (the saved file is used when present); every other campaign is delegated unchanged.
* :func:`patched` -- context manager that swaps ``ru._load_oracle_owner_values`` for it (so code that calls it
  internally, e.g. ``p_shared.step3_gates.build_setup``, needs no change).
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

CHUNK = 1024                      # owners per chunk (``p08_step6_global/configuration.json["owner_chunk_size"]``)
_CACHE: dict = {}


def p06n_owner_values(env) -> np.ndarray:
    """``live_observations``-style owner averages of the P06N catalogue on ``env`` (cached per grid ``env.n``)."""
    key = (int(env.n), int(len(env.t.vol)))
    if key not in _CACHE:
        import p06n_field_derived_global.core as p06n_core
        from p08_step6_global import fields as tfields
        t = env.t
        period = float(t.g.eta_period)
        fn = lambda pts: np.column_stack([p06n_core.p06n_fields.evaluate(env.ref, pts, name, period)[0]
                                          for name in p06n_core.NAMES])                          # noqa: E731
        n_owners = int(len(t.vol))
        _CACHE[key] = np.concatenate([tfields.owner_average_chunk(t, fn, s, min(s + CHUNK, n_owners))
                                      for s in range(0, n_owners, CHUNK)])
    return _CACHE[key]


def load_oracle_owner_values(env, paths: dict, campaigns: tuple, *, _original=None) -> dict:
    """``replay_units._load_oracle_owner_values`` with P06N computed on the fly when its saved file is missing."""
    original = _original if _original is not None else _original_loader()
    campaigns = tuple(campaigns)
    out = original(env, paths, tuple(c for c in campaigns if c != "p06n"))
    if "p06n" in campaigns:
        saved = Path(paths["p05n_p06n_upwind"]) / "p06n" / f"N{env.n}.owner_values.npz"
        if saved.is_file():
            out.update(original(env, paths, ("p06n",)))
        else:
            out["p06n"] = {"owner_values": p06n_owner_values(env).copy()}
    return out


def _original_loader():
    from p_shared import replay_units as ru
    if not _ORIGINAL:
        _ORIGINAL.append(ru._load_oracle_owner_values)
    return _ORIGINAL[0]


_ORIGINAL: list = []                                  # the unpatched loader, captured on first use


@contextlib.contextmanager
def patched():
    """Within the block ``ru._load_oracle_owner_values`` is :func:`load_oracle_owner_values`."""
    from p_shared import replay_units as ru
    original = _original_loader()
    previous = ru._load_oracle_owner_values
    ru._load_oracle_owner_values = lambda env, paths, campaigns: load_oracle_owner_values(
        env, paths, campaigns, _original=original)
    try:
        yield
    finally:
        ru._load_oracle_owner_values = previous
