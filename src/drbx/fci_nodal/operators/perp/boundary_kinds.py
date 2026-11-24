"""Boundary kinds and normalization shared by the nodal operators."""

DIRICHLET, NEUMANN = "dirichlet", "neumann"


def normalize_kinds(field_kinds, n_fields: int) -> tuple[str, ...]:
    """A per-field tuple of ``"dirichlet"`` / ``"neumann"`` (a single string is broadcast)."""
    if isinstance(field_kinds, str):
        field_kinds = (field_kinds,) * n_fields
    kinds = tuple(field_kinds)
    if len(kinds) != n_fields:
        raise ValueError(f"field_kinds has {len(kinds)} entries for {n_fields} fields")
    bad = set(kinds) - {DIRICHLET, NEUMANN}
    if bad:
        raise ValueError(f"unknown field kinds: {sorted(bad)}")
    return kinds
