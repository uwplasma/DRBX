"""The P05/P06 face-quadrature option (P08 operator-change bundle, item 2).

``face_quadrature="q3"`` (default, ``DEFAULT_FACE_QUADRATURE``) integrates the P05 upwind jump and the P06
characteristic side correction with 3x3 Gauss (nine nodes per face); ``"q2"`` uses 2x2 Gauss (four nodes).
P07 always uses q3.  ``"q3"`` is bitwise the historic behaviour everywhere (geometry arrays, saved schemas
and identities, artifact policy, plans and operator outputs); the option is threaded like ``curvature``
(``replay_support.build_environment`` -> ``owner_closure`` -> ``jax_replay`` / ``step3_gates`` /
``build_artifact``).  Design: ``work/p08_q2_face_quadrature_20260930/design.md``, section Q2.2.
"""
from __future__ import annotations

DEFAULT_FACE_QUADRATURE = "q3"
FACE_QUADRATURE_CHOICES = ("q3", "q2")
_ORDER = {"q3": 3, "q2": 2}


def check_face_quadrature(face_quadrature: str) -> str:
    if face_quadrature not in FACE_QUADRATURE_CHOICES:
        raise ValueError(f"face_quadrature must be one of {FACE_QUADRATURE_CHOICES}, got {face_quadrature!r}")
    return face_quadrature


def face_order(face_quadrature: str = DEFAULT_FACE_QUADRATURE) -> int:
    """The Gauss order per face direction: 3 for ``"q3"``, 2 for ``"q2"``."""
    return _ORDER[check_face_quadrature(face_quadrature)]
