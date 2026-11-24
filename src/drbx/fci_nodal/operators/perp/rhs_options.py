"""Static options and wall data for the perpendicular operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax

from .curvature_absolute import ABSOLUTE_METHODS
from .boundary_kinds import DIRICHLET, NEUMANN
from .sbp_curvature import PSI_VARIANTS

FIELD_NAMES = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")          # six-field parallel state order order
TERM_NAMES = ("bracket", "curvature", "diffusion")
CURVATURE_FIELDS = ("density", "Te", "Ti", "vorticity")                  # the columns of the SBP curvature derivative and matrix wall SAT curvature state
_NEUMANN_MODES = ("physical", "conormal")
_PHI_MODES = ("solve", "prescribed")


@dataclass(frozen=True)
class PerpendicularOptions:
    """Static (hashable) options; validated on construction (see the module docstring)."""

    fields: tuple[str, ...] = ("density", "Te", "Ti", "vorticity")
    terms: tuple[str, ...] = ("bracket", "curvature", "diffusion")
    diffusion_kinds: tuple[str, ...] = ()
    neumann_mode: str = "physical"
    phi_mode: str = "solve"
    psi: str = "phi_plus_tau_pi"
    bracket_c_kappa: float = 1.0
    absolute_method: str = "closed_form"
    laplacian_c_kappa: float = 1.0
    phi_rtol: float = 1e-10
    phi_maxit: int = 200

    def __post_init__(self):
        for name in ("fields", "terms", "diffusion_kinds"):
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError(f"{name} must be a sequence of strings, got the string {value!r}")
            object.__setattr__(self, name, tuple(value))
        fields, terms, kinds = self.fields, self.terms, self.diffusion_kinds
        if not fields:
            raise ValueError("fields must not be empty")
        bad = [f for f in fields if f not in FIELD_NAMES]
        if bad:
            raise ValueError(f"unknown fields {bad}; fields must be a subset of {FIELD_NAMES}")
        order = [FIELD_NAMES.index(f) for f in fields]
        if order != sorted(set(order)):
            raise ValueError(f"fields must be distinct and in the order {FIELD_NAMES}, got {fields}")
        bad = [t for t in terms if t not in TERM_NAMES]
        if bad or len(set(terms)) != len(terms):
            raise ValueError(f"terms must be distinct entries of {TERM_NAMES}, got {terms}")
        if "curvature" in terms:
            missing = [f for f in CURVATURE_FIELDS if f not in fields]
            if missing:
                raise ValueError(f"the curvature term needs the fields {CURVATURE_FIELDS}; missing {missing}")
        if ("diffusion" in terms or kinds) and len(kinds) != len(fields):
            raise ValueError(f"diffusion_kinds needs one entry per field ({len(fields)}), got {len(kinds)}")
        if any(k not in (DIRICHLET, NEUMANN) for k in kinds):
            raise ValueError(f"diffusion_kinds entries must be 'dirichlet' or 'neumann', got {kinds}")
        if self.neumann_mode not in _NEUMANN_MODES:
            raise ValueError(f"neumann_mode must be one of {_NEUMANN_MODES}, got {self.neumann_mode!r}")
        if self.phi_mode not in _PHI_MODES:
            raise ValueError(f"phi_mode must be one of {_PHI_MODES}, got {self.phi_mode!r}")
        if self.psi not in PSI_VARIANTS:
            raise ValueError(f"psi must be one of {PSI_VARIANTS}, got {self.psi!r}")
        if self.absolute_method not in ABSOLUTE_METHODS:
            raise ValueError(f"absolute_method must be one of {ABSOLUTE_METHODS}, got {self.absolute_method!r}")
        if not self.phi_maxit >= 1:
            raise ValueError(f"phi_maxit must be >= 1, got {self.phi_maxit}")

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def curvature_index(self) -> tuple[int, ...]:
        """Positions of ``(n, Te, Ti, omega)`` in ``fields``."""
        return tuple(self.fields.index(f) for f in CURVATURE_FIELDS)


class NodalPerpendicularParams(NamedTuple):
    rho_star: jax.Array          # scalar; rho_s0 / L_ref (multiplies the bracket and the curvature, divides the polarization twice)
    tau: jax.Array               # scalar
    D: jax.Array                 # (F,) perpendicular diffusion per field in ``opts.fields`` order


class NodalWallData(NamedTuple):
    """Traced data of the (single, Zernike core-ring layout) wall: ``value (E, N, F)`` inflow (bracket, curvature) and Dirichlet (diffusion)
    data, ``normal (E, N, F)`` the Neumann datum (read for Neumann-kind fields), ``psi (E, N)`` the Dirichlet ``psi`` datum."""

    value: jax.Array
    normal: jax.Array | None = None
    psi: jax.Array | None = None
