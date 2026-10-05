"""The eta-filtered field option (P-path operator option ``eta_filter``): a low-resolution MMS verification arm.

``eta_filter=None`` (the default) is the raw field, bitwise the historic behaviour.  Otherwise it is the dict
``{"quantity": "J*B^i", "max_harmonic_per_period": 3, "nfp": 4, "samples_per_period": 64}``: the contravariant flux density
``F^i = J B^i`` is low-passed along eta at fixed logical ``(u, theta)`` (per-field-period harmonics ``m <= 3``) and
``B~^i = F~^i / J``; ``h = b~_cov / |B~|``, ``|J|`` and ``K`` all derive from ``B~``
(:mod:`drbx.geometry.eta_filtered_field`).  Production uses the raw field; this arm only exists so a low-resolution
verification gate does not see the coil ripple (per-period ``m = 12``) that the grid cannot resolve.

``hsx_mms_continuum_reference.py`` is hash-pinned by frozen manifests, so (as ``p_shared.bfield`` does for the
toroidal interpolation) the swap is done here, right after the frozen reference is built:
:func:`apply_eta_filter` replaces the reference *instance's* ``_metric_batch`` (a bound method called by ``_metric``) by
one that takes the field from the column-exact filtered form, and attaches ``reference.eta_filter_arm`` which
:class:`p_shared.curvature_reference.AutodiffCurvatureReference` uses to feed the JAX table twin of the same field to
the autodiff curvature (instead of ``jax_bfield.evaluate_cartesian``).  The tabulated form is built lazily on the first
autodiff-curvature call; it is cached in memory and, with ``DRBX_ETA_FILTER_TABLE_CACHE=<dir>``, on disk (keyed by the
arm identity; the one-off table-vs-column equivalence check is stored with it).

The arm identity (:func:`arm_identity`) is the sha256 of the canonical JSON of ``(map sha, makegrid sha, currents,
bfield_toroidal, eta_filter, table grid)``; it is recorded in ``reference.provenance["eta_filter"]`` and, by the
P09 extraction, in the nodal-metric meta (hence in the metric identity).  Threaded like ``bfield_toroidal``:
``ScriptsGeometryProvider.from_sidecar(..., eta_filter=...)``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from drbx.geometry.eta_filtered_field import (DEFAULT_TABLE_GRID, EtaFilteredColumnField, EtaFilterSpec,
                                              EtaFilterTable, JaxEtaFilterTable, JaxMetricView)

DEFAULT_ETA_FILTER = None
#: environment variable naming a directory for the on-disk cache of the tabulated form
TABLE_CACHE_ENV = "DRBX_ETA_FILTER_TABLE_CACHE"
#: logical points used by the one-off table-vs-column equivalence check
EQUIVALENCE_POINTS = 256


def check_eta_filter(eta_filter: Mapping[str, Any] | None) -> dict | None:
    """The validated option as a canonical dict (``None`` stays ``None``)."""
    if eta_filter is None:
        return None
    return EtaFilterSpec.from_mapping(eta_filter).as_dict()


def arm_identity(sidecar: str | Path, bfield_toroidal: str, eta_filter: Mapping[str, Any]) -> str:
    """sha256 of (map sha, makegrid sha, currents, bfield_toroidal, eta_filter, table grid)."""
    payload = json.loads(Path(sidecar).resolve().read_text(encoding="utf-8"))
    core = {"map_sha256": payload["metric_cache"]["sha256"], "makegrid_sha256": payload["makegrid"]["sha256"],
            "makegrid_currents": [float(x) for x in payload["makegrid_currents"]], "bfield_toroidal": bfield_toroidal,
            "eta_filter": check_eta_filter(eta_filter), "table_grid": list(DEFAULT_TABLE_GRID)}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()


class EtaFilterArm:
    """The filtered field of one reference: the column-exact form, the (lazily built) table, the arm identity."""

    def __init__(self, reference: Any, spec: EtaFilterSpec, identity: str) -> None:
        self.spec = spec
        self.identity = identity
        self.column_field = EtaFilteredColumnField(reference.metric_evaluator, reference.bfield_evaluator, spec)
        self._table: EtaFilterTable | None = None
        self._jax: JaxEtaFilterTable | None = None
        self.table_check: dict | None = None
        self.table_seconds: float | None = None
        self.table_from_cache: bool | None = None
        #: metric used to sample the table grid (beyond ``u = 1``, which ``MetricEvaluator`` rejects); ``None``: the JAX
        #: metric of ``reference.metric_evaluator``
        self.table_metric: Any = None

    @property
    def eta_filter(self) -> dict:
        return self.spec.as_dict()

    def meta(self) -> dict:
        """The option and identity, as recorded in provenance and in the nodal-metric meta."""
        return {**self.spec.as_dict(), "arm_sha256": self.identity}

    def table(self) -> EtaFilterTable:
        """The tabulated form (built once; loaded from ``DRBX_ETA_FILTER_TABLE_CACHE`` when present)."""
        if self._table is None:
            t0 = time.perf_counter()
            cache_dir = os.environ.get(TABLE_CACHE_ENV)
            path = Path(cache_dir) / f"eta_filter_table_{self.identity[:16]}.npz" if cache_dir else None
            if path is not None and path.exists():
                table = EtaFilterTable.load(path, metric_evaluator=self.column_field.metric_evaluator)
                if table.meta.get("arm_sha256") != self.identity:
                    raise ValueError(f"{path} was built for another arm ({table.meta.get('arm_sha256')})")
                self.table_from_cache = True
            else:
                table = EtaFilterTable.from_column_field(self.column_field, sample_metric=self._table_metric())
                rng = np.random.default_rng(20261004)                  # the one-off equivalence check against columns
                pts = np.stack([rng.uniform(0.02, 1.0, EQUIVALENCE_POINTS), rng.uniform(0.0, 2.0 * np.pi, EQUIVALENCE_POINTS),
                                rng.uniform(0.0, 2.0 * np.pi, EQUIVALENCE_POINTS)], axis=1)
                check = table.check_against_columns(self.column_field, pts)
                table.meta = {"arm_sha256": self.identity, "equivalence_vs_columns": check,
                              "grid": list(DEFAULT_TABLE_GRID)}
                if path is not None:
                    table.save(path)
                self.table_from_cache = False
            self.table_check = table.meta.get("equivalence_vs_columns")
            self.table_seconds = time.perf_counter() - t0
            self._table = table
        return self._table

    def _table_metric(self):
        if self.table_metric is None:
            from drbx.geometry.jax_metric_evaluator import JaxMetricEvaluator

            self.table_metric = JaxMetricView(JaxMetricEvaluator.from_metric_evaluator(self.column_field.metric_evaluator))
        return self.table_metric

    def jax_field(self) -> JaxEtaFilterTable:
        """The JAX twin of the table: the ``logical_field`` of the autodiff curvature."""
        if self._jax is None:
            self._jax = self.table().to_jax()
        return self._jax


def _filtered_metric_batch(reference: Any, arm: EtaFilterArm):
    """``reference._metric_batch`` with the field of ``arm`` (the statements of the original, same keys and dtypes)."""

    def _metric_batch(q):                                      # an instance attribute: called as ``self._metric_batch(q)``
        metric = reference.metric_evaluator.evaluate(q, reject_nonpositive_J=False)
        magnetic = arm.column_field.project_magnetic_field(q, metric)
        J = np.asarray(metric.signed_J, dtype=np.float64)
        gcov = np.asarray(metric.covariant_metric, dtype=np.float64)
        gcontra = np.asarray(metric.contravariant_metric, dtype=np.float64)
        bcontra = np.asarray(magnetic.B_contravariant, dtype=np.float64) / reference.B0
        bmag = np.asarray(magnetic.magnitude, dtype=np.float64) / reference.B0
        bmag = np.maximum(bmag, 1.0e-30)
        bunit = bcontra / bmag[..., None]
        bcov = np.einsum("...ij,...j->...i", gcov, bunit)
        return {"J": J, "gcov": gcov, "gcontra": gcontra, "b": bunit, "bcov": bcov, "B": bmag}

    return _metric_batch


def apply_eta_filter(reference, sidecar, eta_filter: Mapping[str, Any] | None = DEFAULT_ETA_FILTER,
                     bfield_toroidal: str = "spline"):
    """Select the eta-filtered field of ``reference`` (in place); returns ``reference``.

    ``eta_filter=None`` leaves the reference untouched (bitwise the raw arm, no attribute, no provenance entry).  Call it
    after ``p_shared.bfield.apply_bfield_toroidal`` (the filter samples ``reference.bfield_evaluator``).
    """
    option = check_eta_filter(eta_filter)
    if option is None:
        if getattr(reference, "eta_filter_arm", None) is not None:
            raise ValueError("eta_filter=None but the reference carries an eta-filtered field")
        return reference
    if getattr(reference, "eta_filter_arm", None) is not None:
        raise ValueError("the reference already carries an eta-filtered field")
    spec = EtaFilterSpec.from_mapping(option)
    arm = EtaFilterArm(reference, spec, arm_identity(sidecar, bfield_toroidal, option))
    reference.eta_filter_arm = arm
    reference._metric_batch = _filtered_metric_batch(reference, arm)
    provenance = getattr(reference, "provenance", None)
    if isinstance(provenance, dict):
        provenance["eta_filter"] = arm.meta()
    return reference


def reference_eta_filter(reference) -> dict | None:
    """The option dict of ``reference`` (``None``: raw), looking through an autodiff-curvature wrapper."""
    arm = getattr(reference, "eta_filter_arm", None)
    return arm.eta_filter if isinstance(arm, EtaFilterArm) else None
