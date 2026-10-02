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

Curvature option.  ``curvature="fd"`` is exactly the above (pass it explicitly to reproduce frozen campaigns).
``curvature="autodiff"`` (the default, ``DEFAULT_CURVATURE``) wraps the frozen reference in
``p_shared.curvature_reference.AutodiffCurvatureReference`` (``_curvature`` is
the autodiff ``K = (B/2|J|) curl(b_cov/B)`` of ``drbx.geometry.curvature_autodiff``;
everything else is delegated), so ``p06_curvature`` (which goes through
``curvature_geometry(reference, points)``) and ``reference`` itself use autodiff K,
and ``p06_face_curvature`` returns ``J`` and ``B`` exactly as the frozen
``_face_geometry`` does but ``K`` from autodiff at *every* node, wall nodes
included (the one-sided finite-difference wall rule is not used).

Face-quadrature option.  ``face_node_weight(faces, keys, order=3)`` is the q3 default; ``order=2`` gives the q2 face
nodes of the P05/P06 option (``p_shared.face_quadrature``).  The provider's ``face_quadrature`` attribute only
records the declared rule (checked against the environment's); P07 always asks for ``order=3``.

B-field toroidal option.  ``bfield_toroidal="spline"`` (the default) leaves the frozen reference's B evaluator untouched;
``"compact_c3"`` replaces ``reference.bfield_evaluator`` by the compact-C3 toroidal evaluator right after the frozen
reference is built (``p_shared.bfield``; the hash-pinned reference builder itself is not edited).

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

from p_shared.bfield import DEFAULT_BFIELD_TOROIDAL, apply_bfield_toroidal, check_bfield_toroidal  # noqa: E402
from p_shared.curvature_reference import check_curvature as _check_curvature  # noqa: E402
from p_shared.curvature_reference import wrap_reference as _wrap_reference  # noqa: E402
from p_shared.curvature_reference import face_geometry as _face_geometry_for  # noqa: E402
from p_shared.curvature_reference import DEFAULT_CURVATURE  # noqa: E402
from p_shared.face_quadrature import DEFAULT_FACE_QUADRATURE  # noqa: E402
from p_shared.face_quadrature import check_face_quadrature as _check_face_quadrature  # noqa: E402

from drbx.stencils.geometry_arrays import GeometryProvider  # noqa: E402


class ScriptsGeometryProvider:
    """The ``GeometryProvider`` used by the P-shared research harness.

    Structurally satisfies ``drbx.stencils.geometry_arrays.GeometryProvider``
    (a ``runtime_checkable`` ``Protocol``): every method here forwards to one
    of the frozen calls documented in this module's docstring, unchanged.
    """

    def __init__(self, reference: Any, *, curvature: str = DEFAULT_CURVATURE,
                 face_quadrature: str = DEFAULT_FACE_QUADRATURE,
                 bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL) -> None:
        self._curvature_choice = _check_curvature(curvature)
        self._face_quadrature = _check_face_quadrature(face_quadrature)
        self._bfield_toroidal = check_bfield_toroidal(bfield_toroidal)
        found = getattr(getattr(reference, "bfield_evaluator", None), "toroidal_method", "spline")
        if found != bfield_toroidal:
            raise ValueError(f"bfield_toroidal {bfield_toroidal!r} does not match the reference evaluator's "
                             f"toroidal_method {found!r} (build it via ScriptsGeometryProvider.from_sidecar)")
        self._reference = _wrap_reference(reference, curvature)

    @classmethod
    def from_sidecar(cls, sidecar, *, verify_hashes: bool = False, curvature: str = DEFAULT_CURVATURE,
                     face_quadrature: str = DEFAULT_FACE_QUADRATURE,
                     bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL) -> "ScriptsGeometryProvider":
        """Build the frozen reference exactly as every accepted campaign
        does, via ``p07_diffusion_global.numerics.reference``, apply the ``bfield_toroidal`` option
        (``p_shared.bfield``; ``"spline"`` changes nothing), then wrap it."""
        reference = apply_bfield_toroidal(_refnum.reference(sidecar, verify_hashes=verify_hashes), sidecar,
                                          bfield_toroidal)
        return cls(reference, curvature=curvature, face_quadrature=face_quadrature,
                   bfield_toroidal=bfield_toroidal)

    @property
    def reference(self) -> Any:
        return self._reference

    @property
    def curvature(self) -> str:
        """``"fd"`` (frozen finite-difference K) or ``"autodiff"``."""
        return self._curvature_choice

    @property
    def face_quadrature(self) -> str:
        """``"q3"`` (default) or ``"q2"``: the P05/P06 face-node rule this provider is declared for
        (checked against ``Environment.face_quadrature`` by ``owner_closure.build_owner_rows``).
        ``face_node_weight`` takes its ``order`` explicitly, so this attribute selects nothing itself."""
        return self._face_quadrature

    @property
    def bfield_toroidal(self) -> str:
        """``"spline"`` (frozen B evaluator) or ``"compact_c3"`` (compact-C3 toroidal interpolation of B)."""
        return self._bfield_toroidal

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
        if self._curvature_choice == "fd":
            return _p06numerics._face_geometry(self._reference, points)
        # J and B: the exact ``reference._metric(q)`` call ``_face_geometry`` makes; K: autodiff at every node
        # (wall nodes included), not the one-sided finite-difference wall rule.
        return _face_geometry_for(self._reference, points)

    # -- P07 ---------------------------------------------------------------
    def p07_perpendicular_tensor(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self._reference._perpendicular_flux_tensor(points)

    def p07_perpendicular_tensor_and_divergence(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        return self._reference._perpendicular_geometry(points)

    # -- Quadrature (q1 / q3, q2 faces) ---------------------------------------------
    def raw_cell_weight(
        self, faces: tuple[np.ndarray, np.ndarray, np.ndarray], keys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        keys = np.asarray(keys, dtype=np.int64)
        return _quadrature(faces, keys, 1, face=False)

    def face_node_weight(
        self, faces: tuple[np.ndarray, np.ndarray, np.ndarray], keys: np.ndarray, order: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """``order x order`` Gauss face nodes and weights (3, the default: q3; 2: q2, the P05/P06 option)."""
        keys = np.asarray(keys, dtype=np.int64)
        return _quadrature(faces, keys, order, face=True)


def _assert_conforms() -> None:
    # A cheap, import-time structural sanity check (not a substitute for the
    # bitwise tests): the class must satisfy the runtime_checkable Protocol.
    uninitialized = object.__new__(ScriptsGeometryProvider)
    if not isinstance(uninitialized, GeometryProvider):
        raise AssertionError("ScriptsGeometryProvider no longer satisfies GeometryProvider")


_assert_conforms()
