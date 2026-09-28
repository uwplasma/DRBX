"""Research ``GeometryProvider`` satisfying ``drbx.stencils.geometry_arrays``.

Wraps the accepted campaigns' own frozen geometry calls, by import, never by
copy: this module contributes no new arithmetic. Every method below is a
one-line forward to a function or method that already exists elsewhere in
this repository and is pinned as an oracle by the P08 step-1 consolidation
design (``work/p08_step1_consolidation_design_20260928/design.md``, §2 and
§4 "task 2: Provider").

Call sites reproduced, with the exact same argument batching the accepted
campaigns use (one call per chunk of points, not one call per point):

* P05 ``h = b_cov/B`` and ``|J|``: there is no separate frozen function for
  this -- ``scripts/p05_direct_midpoint_global/campaign.py`` computes it
  inline, identically, at both its call sites (``compute_owner:308-310`` and
  ``compute_raw_direct:436``): ``metric = ref._metric(points)``,
  ``h = metric["bcov"] / metric["B"][:, None]``, ``jac = abs(metric["J"])``.
  This module reproduces those exact two lines against the same
  ``reference._metric`` method the campaign calls.
* P06 raw-midpoint ``J``/``B``/``K`` (no wall rule):
  ``scripts/perpendicular_structured/reference_geometry.py``'s
  ``curvature_geometry``, exactly as
  ``scripts/p06_structured_global/numerics.py``'s ``_compute_cells`` and
  ``_reference_on_raw_cells`` call it (``curvature_geometry(reference, flat)``
  where ``flat`` is the points batch, reshaped to ``(-1, 3)``).
* P06 face-node ``J``/``B``/``K`` (with the finite-difference wall rule):
  ``scripts/p06_structured_global/numerics.py``'s own ``_face_geometry``
  (~:385-407), imported and called unchanged.
* The frozen reference loader: ``scripts/p07_diffusion_global/numerics.py``'s
  ``reference(sidecar, *, verify_hashes=...)``, the same wrapper around
  ``build_continuum_reference_from_sidecar`` every campaign uses to build its
  ``ref``/``reference`` object from the localized sidecar.
* P07's geometry-only perpendicular tensor: ``reference._perpendicular_flux_tensor``
  (face-node request, as ``scripts/p07_combined_global/candidate.py`` calls
  it) and ``reference._perpendicular_geometry`` (raw-midpoint request, tensor
  plus its logical divergence, as ``scripts/p07n_field_derived_global/core.py``
  and ``scripts/p07_diffusion_global/numerics.py`` call it).
* The q1/q3 quadrature nodes and weights: ``scripts/p07_diffusion_global/
  numerics.py``'s ``quadrature`` function directly. This is the single
  canonical primitive the design's own component inventory (§1) certifies as
  the same arithmetic behind P06's ``base._cell_quadrature``/
  ``_face_quadrature`` (q3) and P06's own generalized ``_cell_quadrature``
  q1 branch; a same-sample check against both of those P06 call sites (see
  ``tests/test_stencils_geometry_provider.py``) found no difference.

Do not put this package's own directory first on ``sys.path`` -- it has no
stdlib-shadowing modules today, but per the P-path convention (see
``p06n_field_derived_global/campaign.py``'s module docstring) always import
it as ``from p_shared import provider`` with ``DRBX/scripts`` on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent  # .../DRBX/scripts
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p07_diffusion_global import numerics as _refnum  # noqa: E402
from p07_diffusion_global.numerics import quadrature as _quadrature  # noqa: E402
from perpendicular_structured.reference_geometry import curvature_geometry as _curvature_geometry  # noqa: E402
import p06_structured_global.numerics as _p06numerics  # noqa: E402

from drbx.stencils.geometry_arrays import GeometryProvider  # noqa: E402


class ScriptsGeometryProvider:
    """The ``GeometryProvider`` used by the P-shared research harness.

    Structurally satisfies ``drbx.stencils.geometry_arrays.GeometryProvider``
    (a ``runtime_checkable`` ``Protocol``): every method here forwards to one
    of the frozen calls documented in this module's docstring, unchanged.
    """

    def __init__(self, reference: Any) -> None:
        self._reference = reference

    @classmethod
    def from_sidecar(cls, sidecar, *, verify_hashes: bool = False) -> "ScriptsGeometryProvider":
        """Build the frozen reference exactly as every accepted campaign
        does, via ``p07_diffusion_global.numerics.reference``, then wrap it."""
        return cls(_refnum.reference(sidecar, verify_hashes=verify_hashes))

    @property
    def reference(self) -> Any:
        return self._reference

    # -- P05 -----------------------------------------------------------
    def p05_metric(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        metric = self._reference._metric(points)
        h = metric["bcov"] / metric["B"][:, None]
        jac = np.abs(metric["J"])
        return h, jac

    # -- P06 -------------------------------------------------------------
    def p06_curvature(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        prepared = _curvature_geometry(self._reference, points)
        return np.asarray(prepared.J), np.asarray(prepared.B), np.asarray(prepared.K)

    def p06_face_curvature(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        return _p06numerics._face_geometry(self._reference, points)

    # -- P07 ---------------------------------------------------------------
    def p07_perpendicular_tensor(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self._reference._perpendicular_flux_tensor(points)

    def p07_perpendicular_tensor_and_divergence(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        return self._reference._perpendicular_geometry(points)

    # -- Quadrature (q1 / q3) ---------------------------------------------
    def raw_cell_weight(
        self, faces: tuple[np.ndarray, np.ndarray, np.ndarray], keys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        keys = np.asarray(keys, dtype=np.int64)
        return _quadrature(faces, keys, 1, face=False)

    def face_node_weight(
        self, faces: tuple[np.ndarray, np.ndarray, np.ndarray], keys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        keys = np.asarray(keys, dtype=np.int64)
        return _quadrature(faces, keys, 3, face=True)


def _assert_conforms() -> None:
    # A cheap, import-time structural sanity check (not a substitute for the
    # bitwise tests): the class must satisfy the runtime_checkable Protocol.
    uninitialized = object.__new__(ScriptsGeometryProvider)
    if not isinstance(uninitialized, GeometryProvider):
        raise AssertionError("ScriptsGeometryProvider no longer satisfies GeometryProvider")


_assert_conforms()
