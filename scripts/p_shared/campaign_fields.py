"""Module-level **field adapters**, one per frozen campaign (P08 step 2b, task E2).

Why this module exists
----------------------
:mod:`p_shared.replay_units` used to define each frozen campaign's
boundary-data callbacks (Dirichlet trace, physical-normal data), its
role/pairing/variant tables and its ``wall_cache`` keys as closures/locals
inside ``_cells_unit_core``, ``_faces_unit_core`` and ``_p07_unit_core``. The
JAX operator harness (design ``work/p08_step2b_operator_assembly_design_
20260929/design.md`` section 2.4) needs the *same* boundary data at arbitrary
point sets (the plan's ``dirichlet_points`` / ``neumann_points``), so those
definitions now live here, once, and both the host replay and the harness use
them. The host replay is a **bitwise no-op** of the pre-refactor closures: every
adapter method executes exactly the statements the closure it replaced did
(same functions, same argument casts, same output layout).

Uniform adapter interface
-------------------------
Every adapter (subclass of :class:`FieldAdapter`; P06-legacy is a per-field
family, see :class:`P06LegacyAdapter`) exposes:

``campaign``
    the campaign key (``"p05"``, ``"p05n_frozen"``, ``"p05n_upwind"``,
    ``"p06n"``, ``"p06_legacy"`` (per field), ``"p07"``, ``"p07n"``).
``owner_values``
    the oracle input array exactly as :func:`p_shared.replay_units.
    _load_oracle_owner_values` loads it (P06-legacy: the per-field
    ``owner_values_all[fi].T``), ``(owners, F)`` in the column order the row
    artifact is applied to.
``dirichlet(points) -> (value (Q, F), gradient (Q, 3, F))``
    exactly the host ``trace_fn`` / ``dirichlet_trace_fn`` of the campaign.
    The gradient axis order is ``(Q, 3, F)`` (P07's ``(Q, F, 3)`` -> ``(Q, 3, F)``
    swap is done here, at the source, as before).
``normal(points) -> g_N (Q, F)``
    the host ``normal_data_fn`` (physical-normal derivative) where the campaign
    has one (P05N, P06N, P07, P07N); ``None`` (attribute) for P05 and
    P06-legacy, which are Dirichlet-only.
``has_normal``
    ``normal is not None``.
``wall_keys[role] -> WallKeys(trace, normal)``
    the exact :class:`p_shared.replay_support.WallDataCache` keys the host
    passes as ``trace_key`` / ``normal_key`` for each unit role (``"cells"``,
    ``"faces"``, ``"p07"``), **including the deliberate omissions**: the
    ``"cells"``-role ``normal`` key of P05N (frozen and upwind) and of P06N is
    ``None`` on purpose. Caching those cells-role normal callbacks reproduced
    every other campaign/term bitwise but shifted P05N-upwind's ``raw_N`` at
    N64 by ~2e-16 (and P06N's ``*_phi_dirichlet``/``*_phi_neumann`` variants'
    raw material/remainder/total by ~1e-16), so they are kept live -- do not
    "fix" them. A ``None`` key means "call the callback live".
``reconstructions[name] -> Reconstruction(columns, field_kinds)``
    how the host builds the boundary-conditioned state from the columns:
    ``columns`` selects (with repeats allowed) the ``owner_values`` /
    boundary-data columns that are fed to one reconstruction and
    ``field_kinds`` gives the per-fed-column ``"dirichlet"``/``"neumann"``
    choice (the host's ``np.where(is_neumann, value_n, value_d)`` selection is
    equivalent to feeding ``owner_values[:, columns]`` with those kinds).
    ``field_kinds`` (the plain attribute) is the per-``owner_values``-column
    kind tuple of the *primary* (compared-to-oracle "N"/"D") reconstruction
    when the host applies one kind per column (P05, P06-legacy: all
    ``"dirichlet"``; P07: all ``"dirichlet"`` at conditioned faces, replay fix
    2; P07N: all ``"neumann"``); it is ``None`` for P05N and P06N, where every
    column is reconstructed in *both* kinds and a role/variant table selects
    (use ``reconstructions``).

Campaign-specific tables (everything a harness needs to select
reconstructions, all taken verbatim from the host code):

P05
    ``pairs`` = ``k.PAIRS`` (8 bracket pairs over the 8 fields), ``fields`` =
    ``k.FIELDS``. Cells centered bracket + live face jump on the p07 topology
    domain; all fields Dirichlet.
P05N (``p05n_frozen`` / ``p05n_upwind``, pinned catalogues
``p05n_catalogue.json`` / ``p05n_upwind_catalogue.json``)
    ``names`` (physical fields, columns of ``owner_values``), ``role_names`` (sorted
    role names), ``role_physical_index`` (role -> physical column),
    ``role_bc`` / ``role_is_neumann`` (per role), ``pair_names``,
    ``n_pair_index`` / ``d_pair_index`` (pairs as *role* index pairs for the
    N and D actions; ``d_pair_index`` maps every ``*_N`` role to its
    ``*_D`` counterpart), ``r_pair_index`` (pairs as *physical* index pairs
    for the MMS reference term ``raw_R``, host-only), ``action_pair_index`` =
    ``n_pair_index + d_pair_index`` (the faces block evaluates both in one
    call and splits at ``P = len(pair_names)``). The role state
    (``reconstructions["role"]``) feeds ``owner_values[:, role_physical_index]``
    with the per-role kinds; both the N and D actions are pair brackets of that
    one role state.
P06N
    ``tables`` (:data:`p06n_field_derived_global.core.CATALOGUE_TABLES`),
    ``names`` (the physical field columns, 5 primitives per variant),
    ``variant_names``, ``field_index[variant]`` / ``is_neumann[variant]`` (a
    variant's 5 columns and their kinds; ``reconstructions[variant]``), and
    ``variant_spec``. The q1 terms use all 5 fed fields; the q3 face
    correction uses only the first four (``Q3_FIELDS = 4``, the ``[:, :4]``
    slice of the fed values). ``evaluate_exact`` is the host-only MMS reference
    (``raw_R_*``).
P06-legacy (:class:`P06LegacyAdapter`, then one :class:`P06LegacyFieldAdapter`
per ``FIELD_NAMES`` entry, ``time_value = 0.37``)
    each per-field adapter has 5 primitive fields, all Dirichlet;
    ``legacy_seam_double_mask(keys)`` / ``legacy_seam_multiplier(keys)`` is the
    duplicated-census seam rule the host applies to the q3 correction
    (``face_multiplier`` 2 on the theta/eta seam faces, 1 elsewhere; the
    operator itself uses the deduplicated census). ``Q3_FIELDS = 4`` as above.
P07 / P07N
    ``dirichlet_fields`` (owner_values columns that use the Dirichlet D lift at
    a conditioned face: **every** field for plain P07, **none** for P07N),
    ``need_D`` (P07N also needs the all-D flux ``flux_D``; plain P07 does
    not), ``radial_degree_by_family`` (:data:`P07_RADIAL_DEGREE_BY_FAMILY`,
    the conditioned families ``{1: 4, 2: 4, 4: 3}`` shared by both). P07N adds
    ``names`` and the host-only ``exact_gradients`` (the ``O_q3`` reference
    term, ``(Q, F, 3)``).

All ``points`` are ``(Q, 3)`` in the campaign's native ``(u, theta, eta)``
coordinates; adapters cast with ``np.asarray(q, dtype=np.float64)`` exactly as
the closures did. Import note: this module puts ``DRBX/scripts`` on
``sys.path`` (as :mod:`p_shared.replay_units` does) so the frozen campaign
modules import lazily inside the constructors; run from ``DRBX/scripts`` (never
from a directory containing a stray ``operator.py``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent            # .../DRBX/scripts
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p_shared.replay_support import _p05n_evaluate, _tables_trace_all      # noqa: E402

__all__ = [
    "WallKeys", "Reconstruction", "FieldAdapter", "P05Adapter", "P05NAdapter", "P06NAdapter",
    "P06LegacyFieldAdapter", "P06LegacyAdapter", "P07Adapter", "P07NAdapter",
    "P05N_CATALOGUE_FILES", "P07_RADIAL_DEGREE_BY_FAMILY", "P06_LEGACY_TIME_VALUE", "Q3_FIELDS",
    "legacy_seam_double_mask", "legacy_seam_multiplier", "build_adapter",
]

#: campaign key -> pinned catalogue file name (``p05n_core._CATALOGUE_TABLES`` key).
P05N_CATALOGUE_FILES = {"p05n_frozen": "p05n_catalogue.json", "p05n_upwind": "p05n_upwind_catalogue.json"}
#: Conditioned (boundary-reaching) census families of the integrated P07 rows and their radial degree.
P07_RADIAL_DEGREE_BY_FAMILY = {1: 4, 2: 4, 4: 3}
#: The frozen P06-legacy campaign's evaluation time.
P06_LEGACY_TIME_VALUE = 0.37
#: The q3 characteristic correction uses only the first four of the five primitive fields.
Q3_FIELDS = 4

DIRICHLET = "dirichlet"
NEUMANN = "neumann"


@dataclass(frozen=True)
class WallKeys:
    """``WallDataCache`` keys for one unit role; ``None`` = live (deliberate omission)."""
    trace: Optional[str]
    normal: Optional[str] = None


@dataclass(frozen=True)
class Reconstruction:
    """``owner_values[:, columns]`` fed to one reconstruction with per-fed-column ``field_kinds``."""
    columns: np.ndarray
    field_kinds: tuple


def _all(kind: str, count: int) -> tuple:
    return (kind,) * int(count)


class FieldAdapter:
    """Base class; see the module docstring for the interface."""

    campaign: str = ""
    #: ``normal`` is a bound method in adapters with a physical-normal callback, else ``None``.
    normal = None
    has_normal: bool = False

    def __init__(self, ref, owner_values):
        self.ref = ref
        self.owner_values = owner_values
        self.wall_keys: dict = {}
        self.reconstructions: dict = {}
        self.field_kinds: Optional[tuple] = None

    @property
    def n_fields(self) -> int:
        return int(self.owner_values.shape[1])

    def dirichlet(self, points):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# P05
# ---------------------------------------------------------------------------
class P05Adapter(FieldAdapter):
    campaign = "p05"

    def __init__(self, ref, owner_values):
        import p05_structured_global.numerics as k

        super().__init__(ref, owner_values)
        self._k = k
        self.fields = tuple(k.FIELDS)
        self.pairs = tuple(k.PAIRS)
        self.field_kinds = _all(DIRICHLET, len(k.FIELDS))
        self.wall_keys = {"cells": WallKeys("p05_cells_trace", None), "faces": WallKeys("p05_faces_trace", None)}
        self.reconstructions = {"D": Reconstruction(np.arange(len(k.FIELDS), dtype=np.int64), self.field_kinds)}

    def dirichlet(self, points):
        return self._k.boundary_trace(self.ref, np.asarray(points, dtype=np.float64))


# ---------------------------------------------------------------------------
# P05N (frozen / upwind catalogues)
# ---------------------------------------------------------------------------
class P05NAdapter(FieldAdapter):
    def __init__(self, campaign: str, ref, period, owner_values):
        import p05n_field_derived_global.core as p05n_core

        super().__init__(ref, owner_values)
        self.campaign = campaign
        table = p05n_core._CATALOGUE_TABLES[P05N_CATALOGUE_FILES[campaign]]
        self.period = period
        self.names = tuple(table["names"])
        roles = dict(table["roles"])
        self.role_names = tuple(sorted(roles))
        self.role_physical_index = np.array([self.names.index(roles[r][0]) for r in self.role_names], dtype=np.int64)
        self.role_bc = {r: roles[r][1] for r in self.role_names}
        self.role_is_neumann = np.asarray([self.role_bc[r] == "neumann" for r in self.role_names])
        pairings = dict(table["pairings"])
        self.pair_names = tuple(sorted(pairings))
        role_index = {r: i for i, r in enumerate(self.role_names)}
        self.n_pair_index = [(role_index[pairings[p][0]], role_index[pairings[p][1]]) for p in self.pair_names]
        dirichlet_counterpart = {r: (r[:-2] + "_D" if r.endswith("_N") else r) for r in roles}
        self.d_pair_index = [(role_index[dirichlet_counterpart[pairings[p][0]]],
                              role_index[dirichlet_counterpart[pairings[p][1]]]) for p in self.pair_names]
        self.r_pair_index = [(int(self.role_physical_index[a]), int(self.role_physical_index[b]))
                             for a, b in self.n_pair_index]
        self.action_pair_index = tuple(self.n_pair_index) + tuple(self.d_pair_index)
        self.field_kinds = None
        self.reconstructions = {"role": Reconstruction(
            self.role_physical_index,
            tuple(NEUMANN if b else DIRICHLET for b in self.role_is_neumann))}
        # NOTE: the cells-role normal key is deliberately omitted (module docstring).
        self.wall_keys = {"cells": WallKeys(f"{campaign}_cells_trace", None),
                          "faces": WallKeys(f"{campaign}_faces_trace", f"{campaign}_faces_normal")}
        self.has_normal = True
        self.normal = self._normal

    def dirichlet(self, points):
        q = np.asarray(points, dtype=np.float64)
        v = np.empty((len(q), len(self.names))); g = np.empty((len(q), 3, len(self.names)))
        for j, name in enumerate(self.names):
            vv, gg, _ = _p05n_evaluate(name, self.ref, q, self.period)
            v[:, j] = vv; g[:, :, j] = gg
        return v, g

    def _normal(self, points):
        from p07n_field_derived_global.fields import normal as p07n_normal
        q = np.asarray(points, dtype=np.float64)
        av = p07n_normal(self.ref, q)
        outv = np.empty((len(q), len(self.names)))
        for j, name in enumerate(self.names):
            _, gg, _ = _p05n_evaluate(name, self.ref, q, self.period)
            outv[:, j] = np.einsum("qa,qa->q", av, gg)
        return outv

    def exact_gradient(self, points):
        """Host-only MMS reference (``raw_R``): analytic gradient ``(Q, 3, F)`` at ``points``."""
        exact_grad = np.empty((len(points), 3, len(self.names)))
        for j, name in enumerate(self.names):
            _vv, gg, _ = _p05n_evaluate(name, self.ref, points, self.period)
            exact_grad[:, :, j] = gg
        return exact_grad


# ---------------------------------------------------------------------------
# P06N
# ---------------------------------------------------------------------------
class P06NAdapter(FieldAdapter):
    campaign = "p06n"

    def __init__(self, ref, period, owner_values):
        import p06n_field_derived_global.core as p06n_core

        super().__init__(ref, owner_values)
        self.period = period
        self.tables = p06n_core.CATALOGUE_TABLES
        self.names = self.tables.names
        self.variant_names = self.tables.variant_names
        self.variant_spec = self.tables.variant_spec
        self.field_index = self.tables.field_index
        self.is_neumann = self.tables.is_neumann
        self.field_kinds = None
        self.reconstructions = {name: Reconstruction(
            self.field_index[name], tuple(NEUMANN if b else DIRICHLET for b in self.is_neumann[name]))
            for name in self.variant_names}
        # NOTE: the cells-role normal key is deliberately omitted (module docstring).
        self.wall_keys = {"cells": WallKeys("p06n_cells_trace", None),
                          "faces": WallKeys("p06n_faces_trace", "p06n_faces_normal")}
        self.has_normal = True
        self.normal = self._normal

    def dirichlet(self, points):
        return _tables_trace_all(self.tables, self.ref, points, self.period)

    def _normal(self, points):
        from p07n_field_derived_global.fields import normal as p07n_normal
        a = p07n_normal(self.ref, np.asarray(points, dtype=np.float64))
        _v, g = self.dirichlet(points)
        return np.einsum("qa,qaf->qf", a, g)

    def evaluate_exact(self, points, field):
        """Host-only MMS reference: ``tables.evaluate`` of one physical field name."""
        return self.tables.evaluate(self.ref, points, field, self.period)


# ---------------------------------------------------------------------------
# P06-legacy: one adapter per legacy field name (5 primitive fields each)
# ---------------------------------------------------------------------------
class P06LegacyFieldAdapter(FieldAdapter):
    campaign = "p06_legacy"

    def __init__(self, ref, field_name: str, owner_values, time_value: float = P06_LEGACY_TIME_VALUE):
        import p06_structured_global.numerics as p06numerics

        super().__init__(ref, owner_values)
        self._p06numerics = p06numerics
        self.field_name = field_name
        self.time_value = time_value
        self.field_kinds = _all(DIRICHLET, owner_values.shape[1])
        self.reconstructions = {"D": Reconstruction(np.arange(owner_values.shape[1], dtype=np.int64),
                                                    self.field_kinds)}
        self.wall_keys = {"cells": WallKeys(f"p06legacy_cells_trace_{field_name}", None),
                          "faces": WallKeys(f"p06legacy_faces_trace_{field_name}", None)}

    def dirichlet(self, points):
        v5, g5 = self._p06numerics._evaluate_fields(self.field_name, self.ref,
                                                    np.asarray(points, dtype=np.float64), self.time_value)
        return v5.T, np.moveaxis(g5, 0, -1)


class P06LegacyAdapter:
    """The P06-legacy family: ``fields[field_name]`` is a :class:`P06LegacyFieldAdapter`."""
    campaign = "p06_legacy"

    def __init__(self, ref, owner_values_all, time_value: float = P06_LEGACY_TIME_VALUE):
        import p06_structured_global.numerics as p06numerics

        self.ref = ref
        self.owner_values_all = owner_values_all
        self.time_value = time_value
        self.field_names = tuple(p06numerics.FIELD_NAMES)
        self.terms = tuple(p06numerics.TERMS)
        self.fields = {name: P06LegacyFieldAdapter(ref, name, owner_values_all[fi].T, time_value)
                       for fi, name in enumerate(self.field_names)}

    def __iter__(self):
        return iter(self.fields.items())


def legacy_seam_double_mask(keys) -> np.ndarray:
    """Faces P06-legacy's duplicated census counts twice (theta seam ``axis == 1, j == 0`` and eta seam
    ``axis == 2, k == 0``); ``keys`` is ``(F, 4)`` census face keys."""
    keys = np.asarray(keys)
    return ((keys[:, 0] == 1) & (keys[:, 2] == 0)) | ((keys[:, 0] == 2) & (keys[:, 3] == 0))


def legacy_seam_multiplier(keys) -> np.ndarray:
    """The ``face_multiplier`` (2.0 on seam faces, 1.0 elsewhere) the host applies to the legacy q3 correction."""
    return np.where(legacy_seam_double_mask(keys), 2.0, 1.0)


# ---------------------------------------------------------------------------
# P07 / P07N
# ---------------------------------------------------------------------------
class P07Adapter(FieldAdapter):
    """Plain P07 (frozen, independently hand-rolled ``p07_diffusion_global`` campaign).

    Root cause of the earlier plain-P07 wall mismatch (P08 step-1 finding, replay fix 2): the frozen
    ``p07_diffusion_global.numerics.face_chunk`` builds a wall-reaching face's flux from ITS OWN
    field-dependent boundary reconstruction (field 0, ``phi_mms``, from a Dirichlet ('value')
    ``BoundaryRelation``; fields 1-3 -- ``Ti_mms``, ``regular_neumann``, ``mixed_eta_neumann`` -- from a
    Neumann ('normal_derivative') one), but *neither* branch is the package's 28-node physical-normal
    wall-trace elimination the P07N Neumann restoration performs. A bounded id-level diff against the
    frozen kernel confirmed every one of plain P07's fields reproduces to roundoff via the row's own
    Dirichlet D lift at a conditioned (family 1/2/4) face, never via the Neumann restoration -- so
    ``dirichlet_fields`` lists every field (and ``field_kinds`` is all ``"dirichlet"``).

    ``p07_fields`` returns its gradient as ``(Q, fields, 3)``, unlike every other callback here (the
    ``(Q, 3, fields)`` convention ``apply_point_row`` / ``apply_integrated_row`` expect, ``[:, 1:]``
    slicing the *spatial* axis to theta/eta); ``dirichlet`` and ``normal`` swap it once, at the source.
    """
    campaign = "p07"
    need_D = False
    radial_degree_by_family = P07_RADIAL_DEGREE_BY_FAMILY

    def __init__(self, ref, owner_values):
        from p07_diffusion_global.numerics import fields as p07_fields

        super().__init__(ref, owner_values)
        self._fields = p07_fields
        # Every plain-P07 field uses the D lift at a conditioned face (replay fix 2; see
        # ``replay_units._p07_unit_core`` history / ``_p07_family_flux_unit`` docstring).
        self.dirichlet_fields = tuple(range(owner_values.shape[1]))
        self.field_kinds = _all(DIRICHLET, owner_values.shape[1])
        self.reconstructions = {"N": Reconstruction(np.arange(owner_values.shape[1], dtype=np.int64),
                                                    self.field_kinds)}
        self.wall_keys = {"p07": WallKeys("p07_p07_trace", "p07_p07_normal")}
        self.has_normal = True
        self.normal = self._normal

    def dirichlet(self, points):
        # ``p07_fields`` returns its gradient as (Q, fields, 3); swap once, at the source, to (Q, 3, fields).
        v, g, _h = self._fields(self.ref, np.asarray(points, dtype=np.float64))
        return v, np.swapaxes(g, 1, 2)

    def _normal(self, points):
        from p07n_field_derived_global.fields import normal as p07n_normal
        a = p07n_normal(self.ref, np.asarray(points, dtype=np.float64))
        _v, g, _h = self._fields(self.ref, np.asarray(points, dtype=np.float64))
        g = np.swapaxes(g, 1, 2)
        return np.einsum("qa,qaf->qf", a, g)


class P07NAdapter(FieldAdapter):
    campaign = "p07n"
    need_D = True
    radial_degree_by_family = P07_RADIAL_DEGREE_BY_FAMILY

    def __init__(self, ref, period, owner_values):
        import p07n_field_derived_global.fields as p07n_fields

        super().__init__(ref, owner_values)
        self._fields = p07n_fields
        self.period = period
        self.names = tuple(p07n_fields.NAMES)
        self.dirichlet_fields = ()
        self.field_kinds = _all(NEUMANN, owner_values.shape[1])
        cols = np.arange(owner_values.shape[1], dtype=np.int64)
        self.reconstructions = {"N": Reconstruction(cols, self.field_kinds),
                                "D": Reconstruction(cols, _all(DIRICHLET, owner_values.shape[1]))}
        self.wall_keys = {"p07": WallKeys("p07n_p07_trace", "p07n_p07_normal")}
        self.has_normal = True
        self.normal = self._normal

    def dirichlet(self, points):
        q = np.asarray(points, dtype=np.float64)
        f = self._fields
        v = np.column_stack([f.evaluate(self.ref, q, name, self.period)[0] for name in f.NAMES])
        g = np.stack([f.evaluate(self.ref, q, name, self.period)[1] for name in f.NAMES], axis=-1)
        return v, g

    def _normal(self, points):
        f = self._fields
        q = np.asarray(points, dtype=np.float64)
        a = f.normal(self.ref, q)
        outv = np.empty((len(q), len(f.NAMES)))
        for j, name in enumerate(f.NAMES):
            _, g, _ = f.evaluate(self.ref, q, name, self.period)
            outv[:, j] = np.einsum("qa,qa->q", a, g)
        return outv

    def exact_gradients(self, points):
        """Host-only ``O_q3`` reference: analytic gradients ``(Q, F, 3)`` (field axis 1)."""
        f = self._fields
        return np.stack([f.evaluate(self.ref, points, name, self.period)[1] for name in f.NAMES], axis=1)


def build_adapter(campaign: str, *, ref, period, owner_values):
    """Factory: the adapter for ``campaign`` (``owner_values`` as loaded by
    ``replay_units._load_oracle_owner_values``; for ``"p06_legacy"`` the full ``(fields, 5, owners)``
    array, returning a :class:`P06LegacyAdapter`)."""
    if campaign == "p05":
        return P05Adapter(ref, owner_values)
    if campaign in P05N_CATALOGUE_FILES:
        return P05NAdapter(campaign, ref, period, owner_values)
    if campaign == "p06n":
        return P06NAdapter(ref, period, owner_values)
    if campaign == "p06_legacy":
        return P06LegacyAdapter(ref, owner_values)
    if campaign == "p07":
        return P07Adapter(ref, owner_values)
    if campaign == "p07n":
        return P07NAdapter(ref, period, owner_values)
    raise KeyError(f"unknown campaign {campaign!r}")
