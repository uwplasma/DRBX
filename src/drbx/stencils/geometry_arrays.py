"""Field-independent geometry coefficients at the perpendicular stencil nodes.

This module is the data layer for the "Geometry-only coefficients (G)" of the
P08 step-1 consolidation design (``work/p08_step1_consolidation_design_20260928/
design.md``, §2-4): the metric/curvature/quadrature quantities every P05-P07(N)
row request needs, computed once per grid from the field-independent reference
geometry and reused across every field/case/BC variant.

``GeometryProvider`` is a narrow :class:`typing.Protocol` fixing the shapes of
those coefficients; it carries no arithmetic of its own. The research adapter
that actually satisfies it, by importing (never copying) the accepted
campaigns' own frozen calls, lives outside the installable package at
``scripts/p_shared/provider.py`` (see that module's docstring for the exact
call sites reproduced). ``GeometryArrays`` is the thin, explicit container that
holds one provider's output over a fixed batch of raw midpoints and face q3
nodes, with npz save/load and a sha256 identity so a stale or hand-edited file
is rejected rather than silently misread.

Nothing here imports from ``scripts/``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

SCHEMA = "drbx.p-stencil-geometry-arrays.v1"


@runtime_checkable
class GeometryProvider(Protocol):
    """Field-independent geometry the P05/P06/P07(N) stencils consume.

    Every method takes a batch of physical points ``(Q, 3)`` in
    ``(u, theta, eta)`` and returns arrays over that same batch, in the same
    order and count; nothing here selects, filters, or reorders points (that
    is the census/row-artifact builder's job). A conforming implementation
    reuses the accepted campaigns' own frozen calls verbatim -- same function,
    same argument batching -- rather than re-deriving the formulas, so that
    its output is bitwise identical to calling those functions directly.
    """

    def p05_metric(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """P05's ``h = b_cov/B`` (Q, 3) and ``|J|`` (Q,).

        The same formula serves raw midpoints and face q3 nodes; only the
        points passed in differ.
        """
        ...

    def p06_curvature(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """P06's ``J`` (Q,), ``B`` (Q,), ``K`` (Q, 3) at raw midpoints.

        Uses the analytic curvature stencil throughout (no wall rule); see
        ``p06_face_curvature`` for the face-node variant.
        """
        ...

    def p06_face_curvature(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """P06's ``J``, ``B``, ``K`` at face q3 nodes, with the finite-difference
        wall rule applied at ``u == 1`` (the accepted P06 face-geometry call)."""
        ...

    def p07_perpendicular_tensor(self, points: np.ndarray) -> np.ndarray:
        """The geometry-only P07 tensor ``J*(g^ij - b^i b^j)`` (Q, 3, 3).

        This is the face-node request: P07's combined candidate contracts it
        directly with the field-dependent Hessian/gradient at runtime.
        """
        ...

    def p07_perpendicular_tensor_and_divergence(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The same tensor (Q, 3, 3) plus its logical divergence (Q, 3).

        This is the raw-midpoint request (P07N / the P07 portable reference):
        both outputs are geometry-only, so both are precomputable.
        """
        ...

    def raw_cell_weight(
        self, faces: tuple[np.ndarray, np.ndarray, np.ndarray], keys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The single-point (q1) raw-cell quadrature node and weight.

        Returns ``(points, weight)`` with ``points`` shape ``(Q, 1, 3)`` and
        ``weight`` shape ``(Q, 1)``, from the same shared quadrature primitive
        P06 uses for its raw-midpoint volume element.
        """
        ...

    def face_node_weight(
        self, faces: tuple[np.ndarray, np.ndarray, np.ndarray], keys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The 3x3 (q3) face quadrature nodes and weights.

        Returns ``(points, weight)`` with ``points`` shape ``(Q, 9, 3)`` and
        ``weight`` shape ``(Q, 9)``.
        """
        ...


def _array(value) -> np.ndarray:
    return np.asarray(value)


def _hash_arrays(names_and_arrays) -> "hashlib._Hash":
    digest = hashlib.sha256()
    for name, array in names_and_arrays:
        digest.update(name.encode("utf-8"))
        digest.update(repr(array.dtype).encode("utf-8"))
        digest.update(repr(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest


# ---------------------------------------------------------------------------
# Split raw/face building blocks, reused (not copied) by both
# ``GeometryArrays.build`` (one-shot, single-process) and the parallel
# geometry stage (``scripts/p_shared/build_artifact.py``'s ``geometry_raw``/
# ``geometry_face`` units, run through ``scripts/p_shared/runner.run_stage``).
# Each function below is exactly the corresponding slice of ``build``'s own
# body (same provider methods, same argument batching, same reshapes) so a
# unit computing a sub-batch of raw cells or faces is bitwise/near-bitwise
# consistent with a single call over the whole batch -- see
# ``build_artifact.py``'s module docstring for why a small residual
# (batch-size-dependent metric-evaluator roundoff, not a numerics change)
# can remain between the two call shapes.
# ---------------------------------------------------------------------------
def build_raw_geometry_arrays(
    provider: GeometryProvider,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray],
    raw_keys: np.ndarray,
) -> dict[str, np.ndarray]:
    """The ``*_raw_*`` half of :meth:`GeometryArrays.build`, over one batch
    of raw ``(i, j, k)`` cell keys. Returns a plain ``{field: array}`` dict
    keyed by the raw subset of ``GeometryArrays``'s own ``_ARRAY_FIELDS``."""
    raw_keys = np.asarray(raw_keys, dtype=np.int64)
    if raw_keys.ndim != 2 or raw_keys.shape[1] != 3:
        raise ValueError("raw_keys must have shape (R, 3)")

    raw_node_points, raw_weight = provider.raw_cell_weight(faces, raw_keys)
    raw_node_points = _array(raw_node_points)
    if raw_node_points.shape[1:] != (1, 3):
        raise ValueError("raw_cell_weight must return a single (q1) node per raw cell")
    raw_points = raw_node_points.reshape(-1, 3)
    raw_weight = _array(raw_weight).reshape(-1)

    p05_raw_h, p05_raw_jacobian = provider.p05_metric(raw_points)
    p06_raw_J, p06_raw_B, p06_raw_K = provider.p06_curvature(raw_points)
    p07_raw_tensor, p07_raw_divergence = provider.p07_perpendicular_tensor_and_divergence(raw_points)

    return dict(
        raw_points=raw_points,
        p05_raw_h=_array(p05_raw_h),
        p05_raw_jacobian=_array(p05_raw_jacobian),
        p06_raw_J=_array(p06_raw_J),
        p06_raw_B=_array(p06_raw_B),
        p06_raw_K=_array(p06_raw_K),
        p06_raw_weight=raw_weight,
        p07_raw_tensor=_array(p07_raw_tensor),
        p07_raw_divergence=_array(p07_raw_divergence),
    )


def build_face_geometry_arrays(
    provider: GeometryProvider,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray],
    face_keys: np.ndarray,
) -> dict[str, np.ndarray]:
    """The ``*_face_*`` half of :meth:`GeometryArrays.build`, over one batch
    of ``(axis, i, j, k)`` face keys. Returns a plain ``{field: array}`` dict
    keyed by the face subset of ``GeometryArrays``'s own ``_ARRAY_FIELDS``."""
    face_keys = np.asarray(face_keys, dtype=np.int64)
    if face_keys.ndim != 2 or face_keys.shape[1] != 4:
        raise ValueError("face_keys must have shape (F, 4)")

    face_points, face_weight = provider.face_node_weight(faces, face_keys)
    face_points = _array(face_points)
    if face_points.shape[1:] != (9, 3):
        raise ValueError("face_node_weight must return nine (q3) nodes per face")
    n_faces = face_points.shape[0]
    flat_faces = face_points.reshape(-1, 3)
    face_weight = _array(face_weight).reshape(n_faces, 9)

    p05_face_h, p05_face_jacobian = provider.p05_metric(flat_faces)
    p06_face_J, p06_face_B, p06_face_K = provider.p06_face_curvature(flat_faces)
    p07_face_tensor = provider.p07_perpendicular_tensor(flat_faces)

    return dict(
        face_points=face_points,
        p05_face_h=_array(p05_face_h).reshape(n_faces, 9, 3),
        p05_face_jacobian=_array(p05_face_jacobian).reshape(n_faces, 9),
        p06_face_J=_array(p06_face_J).reshape(n_faces, 9),
        p06_face_B=_array(p06_face_B).reshape(n_faces, 9),
        p06_face_K=_array(p06_face_K).reshape(n_faces, 9, 3),
        p06_face_weight=face_weight,
        p07_face_tensor=_array(p07_face_tensor).reshape(n_faces, 9, 3, 3),
    )


# Field order fixes the identity hash and the npz layout; do not reorder
# without also bumping SCHEMA (identity would silently change meaning).
_ARRAY_FIELDS = (
    "raw_points",
    "p05_raw_h",
    "p05_raw_jacobian",
    "p06_raw_J",
    "p06_raw_B",
    "p06_raw_K",
    "p06_raw_weight",
    "p07_raw_tensor",
    "p07_raw_divergence",
    "face_points",
    "p05_face_h",
    "p05_face_jacobian",
    "p06_face_J",
    "p06_face_B",
    "p06_face_K",
    "p06_face_weight",
    "p07_face_tensor",
)


@dataclass(frozen=True)
class GeometryArrays:
    """Geometry-only coefficients at a fixed batch of raw midpoints and face
    q3 nodes, as produced by a :class:`GeometryProvider`.

    ``raw_points`` has shape ``(R, 3)``; every ``*_raw_*`` array shares its
    leading axis with it. ``face_points`` has shape ``(F, 9, 3)`` (nine q3
    nodes per face); every ``*_face_*`` array shares its first two axes with
    it. ``schema`` and ``identity`` are recomputed on save and checked on
    load, never trusted from a caller.
    """

    schema: str
    identity: str
    raw_points: np.ndarray
    p05_raw_h: np.ndarray
    p05_raw_jacobian: np.ndarray
    p06_raw_J: np.ndarray
    p06_raw_B: np.ndarray
    p06_raw_K: np.ndarray
    p06_raw_weight: np.ndarray
    p07_raw_tensor: np.ndarray
    p07_raw_divergence: np.ndarray
    face_points: np.ndarray
    p05_face_h: np.ndarray
    p05_face_jacobian: np.ndarray
    p06_face_J: np.ndarray
    p06_face_B: np.ndarray
    p06_face_K: np.ndarray
    p06_face_weight: np.ndarray
    p07_face_tensor: np.ndarray

    @classmethod
    def build(
        cls,
        provider: GeometryProvider,
        *,
        faces: tuple[np.ndarray, np.ndarray, np.ndarray],
        raw_keys: np.ndarray,
        face_keys: np.ndarray,
    ) -> "GeometryArrays":
        """Fill every array by calling ``provider`` once per coefficient.

        ``raw_keys`` is ``(R, 3)`` integer ``(i, j, k)`` raw-cell indices;
        ``face_keys`` is ``(F, 4)`` integer ``(axis, i, j, k)`` face indices.
        The q1/q3 node points come from the provider's own quadrature calls
        (:meth:`GeometryProvider.raw_cell_weight` / ``face_node_weight``), so
        every other coefficient is evaluated at exactly the points its weight
        was computed at. Each remaining provider method is called exactly
        once over the flattened batch, matching how the accepted campaigns
        call them (one batched call per chunk, not one per point).
        """
        raw_keys = np.asarray(raw_keys, dtype=np.int64)
        face_keys = np.asarray(face_keys, dtype=np.int64)
        if raw_keys.ndim != 2 or raw_keys.shape[1] != 3:
            raise ValueError("raw_keys must have shape (R, 3)")
        if face_keys.ndim != 2 or face_keys.shape[1] != 4:
            raise ValueError("face_keys must have shape (F, 4)")

        raw_node_points, raw_weight = provider.raw_cell_weight(faces, raw_keys)
        raw_node_points = _array(raw_node_points)
        if raw_node_points.shape[1:] != (1, 3):
            raise ValueError("raw_cell_weight must return a single (q1) node per raw cell")
        raw_points = raw_node_points.reshape(-1, 3)
        raw_weight = _array(raw_weight).reshape(-1)

        face_points, face_weight = provider.face_node_weight(faces, face_keys)
        face_points = _array(face_points)
        if face_points.shape[1:] != (9, 3):
            raise ValueError("face_node_weight must return nine (q3) nodes per face")
        n_faces = face_points.shape[0]
        flat_faces = face_points.reshape(-1, 3)
        face_weight = _array(face_weight).reshape(n_faces, 9)

        p05_raw_h, p05_raw_jacobian = provider.p05_metric(raw_points)
        p06_raw_J, p06_raw_B, p06_raw_K = provider.p06_curvature(raw_points)
        p07_raw_tensor, p07_raw_divergence = provider.p07_perpendicular_tensor_and_divergence(raw_points)

        p05_face_h, p05_face_jacobian = provider.p05_metric(flat_faces)
        p06_face_J, p06_face_B, p06_face_K = provider.p06_face_curvature(flat_faces)
        p07_face_tensor = provider.p07_perpendicular_tensor(flat_faces)

        arrays = dict(
            raw_points=raw_points,
            p05_raw_h=_array(p05_raw_h),
            p05_raw_jacobian=_array(p05_raw_jacobian),
            p06_raw_J=_array(p06_raw_J),
            p06_raw_B=_array(p06_raw_B),
            p06_raw_K=_array(p06_raw_K),
            p06_raw_weight=raw_weight,
            p07_raw_tensor=_array(p07_raw_tensor),
            p07_raw_divergence=_array(p07_raw_divergence),
            face_points=face_points,
            p05_face_h=_array(p05_face_h).reshape(n_faces, 9, 3),
            p05_face_jacobian=_array(p05_face_jacobian).reshape(n_faces, 9),
            p06_face_J=_array(p06_face_J).reshape(n_faces, 9),
            p06_face_B=_array(p06_face_B).reshape(n_faces, 9),
            p06_face_K=_array(p06_face_K).reshape(n_faces, 9, 3),
            p06_face_weight=face_weight,
            p07_face_tensor=_array(p07_face_tensor).reshape(n_faces, 9, 3, 3),
        )
        identity = cls._compute_identity(arrays)
        return cls(schema=SCHEMA, identity=identity, **arrays)

    @staticmethod
    def _compute_identity(arrays: dict) -> str:
        ordered = [(name, _array(arrays[name])) for name in _ARRAY_FIELDS]
        digest = _hash_arrays(ordered)
        digest.update(SCHEMA.encode("utf-8"))
        return digest.hexdigest()

    def _recompute_identity(self) -> str:
        return self._compute_identity({name: getattr(self, name) for name in _ARRAY_FIELDS})

    def verify(self) -> None:
        """Raise ``ValueError`` if the stored identity no longer matches the
        arrays (hand-edited, truncated, or otherwise corrupted content)."""
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported geometry-arrays schema {self.schema!r}")
        recomputed = self._recompute_identity()
        if recomputed != self.identity:
            raise ValueError(
                "geometry-arrays identity does not match its arrays "
                f"(stored {self.identity!r}, recomputed {recomputed!r})"
            )

    def save(self, path: str | Path) -> None:
        """Write every array plus ``schema``/``identity`` to a single npz file."""
        self.verify()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        payload["schema"] = np.array(self.schema)
        payload["identity"] = np.array(self.identity)
        with tmp.open("wb") as handle:
            np.savez(handle, **payload)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, *, expected_identity: str | None = None) -> "GeometryArrays":
        """Read a file written by :meth:`save`, verifying its identity.

        Raises ``ValueError`` if the file's own arrays no longer match its
        stored identity, or if ``expected_identity`` is given and differs
        from the file's identity (a caller asking for a specific, now-stale
        geometry snapshot).
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"])
            identity = str(data["identity"])
            arrays = {name: np.asarray(data[name]) for name in _ARRAY_FIELDS}
        instance = cls(schema=schema, identity=identity, **arrays)
        instance.verify()
        if expected_identity is not None and instance.identity != expected_identity:
            raise ValueError(
                f"geometry-arrays identity {instance.identity!r} at {path} does not match "
                f"the expected identity {expected_identity!r}"
            )
        return instance


assert tuple(f.name for f in fields(GeometryArrays) if f.name in _ARRAY_FIELDS) == _ARRAY_FIELDS
