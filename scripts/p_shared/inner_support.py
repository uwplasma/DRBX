"""The inner donor-support option (P08 operator-change bundle, item 3).

``inner_support="profile7"`` (C0, the historic rule) is the rule of
``drbx.geometry.fci_perpendicular_reconstruction.StructuredReconstruction``: the coupled Cartesian quartic fit is
used only while some ring of the four-layer stencil has fewer than seven owners.  ``"last_aggregate"`` (C1) also
uses it for every stencil anchored on a ring at or before the last agglomerated ring (the first full ring minus
one), for the P05/P06 point rows (R1, R2, R3) and the P07 integrated rows (R4).  ``"profile7"`` is bitwise the
historic behaviour everywhere; the option is threaded like ``face_quadrature`` (``replay_support.build_environment``
-> ``owner_closure`` -> ``jax_replay`` / ``step3_gates`` / ``build_artifact``).  Design:
``work/p08_donor_support_c1_20260930/design.md``.
"""
from __future__ import annotations

# C3 adopted 30 September 2026 (user decision) after the C0/C1/C1b/C3 comparison (work/p08_donor_support_c1_20260930);
# "profile7" reproduces the frozen step 1-3 oracles and campaigns and must be passed explicitly for that
DEFAULT_INNER_SUPPORT = "fixed_radius"
INNER_SUPPORT_CHOICES = ("profile7", "last_aggregate", "any_aggregate", "fixed_radius")


def check_inner_support(inner_support: str) -> str:
    if inner_support not in INNER_SUPPORT_CHOICES:
        raise ValueError(f"inner_support must be one of {INNER_SUPPORT_CHOICES}, got {inner_support!r}")
    return inner_support
