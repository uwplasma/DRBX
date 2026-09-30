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

#: P05/P06 face-node rules: ``order x order`` Gauss (3: q3 nine nodes, the default; 2: q2 four nodes)
FACE_ORDERS = (3, 2)


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
        self, faces: tuple[np.ndarray, np.ndarray, np.ndarray], keys: np.ndarray, order: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """The ``order x order`` Gauss face quadrature nodes and weights (``order`` 3: q3, the default;
        2: q2). Callers pass ``order`` only when it is not 3.

        Returns ``(points, weight)`` with ``points`` shape ``(Q, order**2, 3)`` and
        ``weight`` shape ``(Q, order**2)``.
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


def _face_nodes(provider: GeometryProvider, faces, face_keys: np.ndarray, order: int):
    """``provider.face_node_weight`` at ``order``; the argument is omitted at 3 (bitwise the historic call)."""
    if order == 3:
        return provider.face_node_weight(faces, face_keys)
    return provider.face_node_weight(faces, face_keys, order=order)


def _check_face_order(face_order: int) -> int:
    if face_order not in FACE_ORDERS:
        raise ValueError(f"face_order must be one of {FACE_ORDERS}, got {face_order!r}")
    return int(face_order)


def build_face_geometry_arrays(
    provider: GeometryProvider,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray],
    face_keys: np.ndarray,
    face_order: int = 3,
) -> dict[str, np.ndarray]:
    """The ``*_face_*`` half of :meth:`GeometryArrays.build`, over one batch
    of ``(axis, i, j, k)`` face keys. Returns a plain ``{field: array}`` dict
    keyed by the face subset of ``GeometryArrays``'s own ``_ARRAY_FIELDS``.

    ``face_order`` (3, or 2) is the P05/P06 face-node rule: ``face_points`` and every
    ``p05_face_*`` / ``p06_face_*`` array carry ``face_order**2`` nodes. At 3 the result is exactly
    the historic one. At 2 the dict also holds ``p07_face_points`` / ``p07_face_weight`` (the q3
    set) and ``p07_face_tensor`` is evaluated at those nine nodes (P07 always uses q3)."""
    face_order = _check_face_order(face_order)
    nodes = face_order ** 2
    face_keys = np.asarray(face_keys, dtype=np.int64)
    if face_keys.ndim != 2 or face_keys.shape[1] != 4:
        raise ValueError("face_keys must have shape (F, 4)")

    face_points, face_weight = _face_nodes(provider, faces, face_keys, face_order)
    face_points = _array(face_points)
    if face_points.shape[1:] != (nodes, 3):
        raise ValueError("face_node_weight must return nine (q3) nodes per face" if face_order == 3
                         else f"face_node_weight must return {nodes} (q{face_order}) nodes per face")
    n_faces = face_points.shape[0]
    flat_faces = face_points.reshape(-1, 3)
    face_weight = _array(face_weight).reshape(n_faces, nodes)

    extra = {}
    if face_order == 3:
        flat_p07 = flat_faces
    else:
        p07_points, p07_weight = _face_nodes(provider, faces, face_keys, 3)
        p07_points = _array(p07_points)
        if p07_points.shape[1:] != (9, 3):
            raise ValueError("face_node_weight must return nine (q3) nodes per face")
        flat_p07 = p07_points.reshape(-1, 3)
        extra = dict(p07_face_points=p07_points, p07_face_weight=_array(p07_weight).reshape(n_faces, 9))

    p05_face_h, p05_face_jacobian = provider.p05_metric(flat_faces)
    p06_face_J, p06_face_B, p06_face_K = provider.p06_face_curvature(flat_faces)
    p07_face_tensor = provider.p07_perpendicular_tensor(flat_p07)

    return dict(
        face_points=face_points,
        p05_face_h=_array(p05_face_h).reshape(n_faces, nodes, 3),
        p05_face_jacobian=_array(p05_face_jacobian).reshape(n_faces, nodes),
        p06_face_J=_array(p06_face_J).reshape(n_faces, nodes),
        p06_face_B=_array(p06_face_B).reshape(n_faces, nodes),
        p06_face_K=_array(p06_face_K).reshape(n_faces, nodes, 3),
        p06_face_weight=face_weight,
        p07_face_tensor=_array(p07_face_tensor).reshape(n_faces, 9, 3, 3),
        **extra,
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


#: present (and hashed/saved) only at ``face_order == 2``: the P07 q3 face set
_OPTIONAL_FIELDS = ("p07_face_points", "p07_face_weight")


@dataclass(frozen=True)
class GeometryArrays:
    """Geometry-only coefficients at a fixed batch of raw midpoints and face
    q3 nodes, as produced by a :class:`GeometryProvider`.

    ``raw_points`` has shape ``(R, 3)``; every ``*_raw_*`` array shares its
    leading axis with it. ``face_points`` has shape ``(F, 9, 3)`` (nine q3
    nodes per face); every ``*_face_*`` array shares its first two axes with
    it. ``schema`` and ``identity`` are recomputed on save and checked on
    load, never trusted from a caller.

    ``face_order`` (3 default, or 2) is the P05/P06 face-node rule. At 2, ``face_points`` and every
    ``p05_face_*`` / ``p06_face_*`` array have ``(F, 4, ...)`` (q2) nodes, and the P07 q3 set lives in the
    optional ``p07_face_points (F, 9, 3)`` / ``p07_face_weight (F, 9)`` (``p07_face_tensor`` is always at nine
    nodes). At 3 those two are ``None`` (not saved, not hashed) and :attr:`p07_points` / :attr:`p07_weight`
    return ``face_points`` / ``p06_face_weight``, so the q3 schema, files and identity are unchanged.

    ``face_order``, ``p07_face_points`` and ``p07_face_weight`` are keyword-only constructor arguments and
    plain (non-dataclass-field) attributes, so the dataclass field set stays exactly ``schema``, ``identity``
    and ``_ARRAY_FIELDS`` (the schema); they are class-level defaults (3, ``None``, ``None``) unless given.
    ``dataclasses.replace`` therefore does not carry them: rebuild a q2 instance with the constructor.
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

    # not dataclass fields (no annotation): class-level defaults, set per instance by ``__init__`` below
    face_order = 3
    p07_face_points = None
    p07_face_weight = None

    def __init__(self, *args, face_order: int = 3, p07_face_points=None, p07_face_weight=None, **kwargs) -> None:
        names = ("schema", "identity") + _ARRAY_FIELDS
        if len(args) > len(names):
            raise TypeError(f"GeometryArrays takes at most {len(names)} positional arguments ({len(args)} given)")
        values = dict(zip(names, args))
        repeated = sorted(set(values) & set(kwargs))
        if repeated:
            raise TypeError(f"GeometryArrays got multiple values for {repeated}")
        values.update(kwargs)
        missing, unknown = [n for n in names if n not in values], sorted(set(values) - set(names))
        if missing or unknown:
            raise TypeError(f"GeometryArrays: missing arguments {missing}, unexpected arguments {unknown}")
        _check_face_order(face_order)
        has_p07 = p07_face_points is not None
        if has_p07 != (p07_face_weight is not None) or has_p07 != (face_order != 3):
            raise ValueError("p07_face_points / p07_face_weight are present exactly when face_order is 2")
        if face_order == 3 and np.ndim(values["face_points"]) == 3 and np.shape(values["face_points"])[1] == 4:
            raise ValueError("face_points has four (q2) nodes per face but face_order is 3")
        for name in names:
            object.__setattr__(self, name, values[name])
        if face_order != 3:
            object.__setattr__(self, "face_order", int(face_order))
            object.__setattr__(self, "p07_face_points", p07_face_points)
            object.__setattr__(self, "p07_face_weight", p07_face_weight)

    @property
    def p07_points(self) -> np.ndarray:
        """The P07 (q3) face nodes ``(F, 9, 3)``."""
        return self.face_points if self.p07_face_points is None else self.p07_face_points

    @property
    def p07_weight(self) -> np.ndarray:
        """The P07 (q3) face weights ``(F, 9)``."""
        return self.p06_face_weight if self.p07_face_weight is None else self.p07_face_weight

    @classmethod
    def build(
        cls,
        provider: GeometryProvider,
        *,
        faces: tuple[np.ndarray, np.ndarray, np.ndarray],
        raw_keys: np.ndarray,
        face_keys: np.ndarray,
        face_order: int = 3,
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

        face_order = _check_face_order(face_order)
        face = build_face_geometry_arrays(provider, faces, face_keys, face_order)

        p05_raw_h, p05_raw_jacobian = provider.p05_metric(raw_points)
        p06_raw_J, p06_raw_B, p06_raw_K = provider.p06_curvature(raw_points)
        p07_raw_tensor, p07_raw_divergence = provider.p07_perpendicular_tensor_and_divergence(raw_points)

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
            **face,
        )
        if face_order != 3:
            arrays["face_order"] = face_order
        identity = cls._compute_identity(arrays)
        return cls(schema=SCHEMA, identity=identity, **arrays)

    @staticmethod
    def _compute_identity(arrays: dict) -> str:
        """The content hash. ``face_order`` enters the identity only when it is 2 (``arrays`` then holds
        ``p07_face_points`` / ``p07_face_weight``); at 3 the hash is the historic one."""
        ordered = [(name, _array(arrays[name])) for name in _ARRAY_FIELDS]
        q2 = arrays.get("p07_face_points") is not None
        if q2:
            ordered += [(name, _array(arrays[name])) for name in _OPTIONAL_FIELDS]
        digest = _hash_arrays(ordered)
        digest.update(SCHEMA.encode("utf-8"))
        if q2:
            digest.update(f"face_order={int(arrays.get('face_order', 2))}".encode("utf-8"))
        return digest.hexdigest()

    def _recompute_identity(self) -> str:
        arrays = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        if self.face_order != 3:
            arrays.update({name: getattr(self, name) for name in _OPTIONAL_FIELDS}, face_order=self.face_order)
        return self._compute_identity(arrays)

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
        if self.face_order != 3:
            payload.update({name: getattr(self, name) for name in _OPTIONAL_FIELDS})
            payload["face_order"] = np.array(self.face_order)
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
            if "face_order" in data.files:
                arrays.update({name: np.asarray(data[name]) for name in _OPTIONAL_FIELDS})
                arrays["face_order"] = int(data["face_order"])
        instance = cls(schema=schema, identity=identity, **arrays)
        instance.verify()
        if expected_identity is not None and instance.identity != expected_identity:
            raise ValueError(
                f"geometry-arrays identity {instance.identity!r} at {path} does not match "
                f"the expected identity {expected_identity!r}"
            )
        return instance


assert tuple(f.name for f in fields(GeometryArrays) if f.name in _ARRAY_FIELDS) == _ARRAY_FIELDS
