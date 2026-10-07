#!/usr/bin/env python3
"""Compatibility entry point for the HSX FCI Braginskii blob run.

The driver lives in :mod:`drbx.fci_braginskii.run`; ``drbx run`` with a deck
such as ``examples/inputs/hsx_fci_blob.toml`` is the equivalent TOML route.
Every option, default and output of this script is unchanged.  Attribute
reads and writes on this module (including test monkeypatching) go to the
driver module.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types

_drbx_source_override = os.environ.get("DRBX_SOURCE_ROOT")
DRBX_SRC = (
    Path(_drbx_source_override).expanduser().resolve()
    if _drbx_source_override
    else Path(__file__).resolve().parent / "src"
)
if not DRBX_SRC.is_dir() or not (DRBX_SRC / "drbx").is_dir():
    source_origin = "DRBX_SOURCE_ROOT" if _drbx_source_override else "default"
    raise RuntimeError(
        f"{source_origin} DRBX source root must be a src directory containing "
        f"a drbx package, got {DRBX_SRC}"
    )
if str(DRBX_SRC) not in sys.path:
    sys.path.insert(0, str(DRBX_SRC))

from drbx.fci_braginskii import run as _driver  # noqa: E402


class _DriverModule(types.ModuleType):
    def __getattr__(self, name):
        return getattr(_driver, name)

    def __setattr__(self, name, value):
        if name.startswith("__"):
            super().__setattr__(name, value)
        else:
            setattr(_driver, name, value)

    def __delattr__(self, name):
        delattr(_driver, name)


sys.modules[__name__].__class__ = _DriverModule

if __name__ == "__main__":
    _driver.main()
