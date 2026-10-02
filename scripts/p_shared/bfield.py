"""The magnetic-field toroidal-interpolation option (P-path operator option ``bfield_toroidal``).

``bfield_toroidal="spline"`` (the historic rule) is the toroidal cubic spline of the B evaluator that
``hsx_mms_continuum_reference.build_continuum_reference_from_sidecar`` builds; ``"compact_c3"`` swaps in the compact
C3 toroidal interpolation of ``drbx.geometry.Bfield_evaluator`` (``extrapolate=False``).  That builder and
``p07_diffusion_global.numerics.reference`` are hash-pinned by frozen campaign manifests, so the swap is done here:
:func:`apply_bfield_toroidal` replaces ``reference.bfield_evaluator`` (a plain attribute, read lazily by the metric
batch and by the JAX evaluator of the autodiff curvature) right after the frozen reference is built and before
``ScriptsGeometryProvider`` wraps it.  ``"spline"`` leaves the reference untouched, hence bitwise the historic
behaviour.  Threaded like ``inner_support`` (``replay_support.build_environment`` -> ``owner_closure`` ->
``jax_replay`` / ``step3_gates`` / ``build_artifact``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_BFIELD_TOROIDAL = "spline"
BFIELD_TOROIDAL_CHOICES = ("spline", "compact_c3")


def check_bfield_toroidal(bfield_toroidal: str) -> str:
    if bfield_toroidal not in BFIELD_TOROIDAL_CHOICES:
        raise ValueError(f"bfield_toroidal must be one of {BFIELD_TOROIDAL_CHOICES}, got {bfield_toroidal!r}")
    return bfield_toroidal


def apply_bfield_toroidal(reference, sidecar, bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL):
    """Select the toroidal interpolation of ``reference.bfield_evaluator`` (in place); returns ``reference``."""
    check_bfield_toroidal(bfield_toroidal)
    if bfield_toroidal == "compact_c3":
        from drbx.geometry.Bfield_evaluator import bfield_evaluator_from_makegrid

        payload = json.loads(Path(sidecar).resolve().read_text(encoding="utf-8"))
        reference.bfield_evaluator = bfield_evaluator_from_makegrid(
            Path(payload["makegrid"]["path"]).resolve(),
            currents=np.asarray(payload["makegrid_currents"], dtype=np.float64),
            method="cubic", toroidal_method="compact_c3", extrapolate=False)
    else:
        found = getattr(reference.bfield_evaluator, "toroidal_method", "spline")
        if found != "spline":
            raise ValueError(f"bfield_toroidal='spline' but the reference evaluator uses {found!r}")
    provenance = getattr(reference, "provenance", None)
    if isinstance(provenance, dict):
        provenance["bfield_toroidal"] = bfield_toroidal
    return reference
