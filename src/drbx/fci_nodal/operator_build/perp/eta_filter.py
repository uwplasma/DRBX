"""ETA field selection and explicit table cache plumbing."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ...evaluators.eta_filtered_field import (EtaFilteredColumnField, EtaFilterSpec,
                                              EtaFilterTable, JaxEtaFilterTable, JaxMetricView)

#: logical points used by the one-off table-vs-column equivalence check
EQUIVALENCE_POINTS = 256


def arm_identity(provenance: dict, options) -> str:
    """Hash the original arm inputs, with the caller's grid and spec."""
    core = {"map_sha256": provenance["metric_cache"]["sha256"], "makegrid_sha256": provenance["makegrid"]["sha256"],
            "makegrid_currents": [float(x) for x in options.makegrid_currents], "bfield_toroidal": options.bfield_toroidal,
            "eta_filter": options.eta_filter.as_dict(), "table_grid": list(options.table_grid)}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()


class EtaFilterArm:
    """The filtered field of one reference: the column-exact form, the (lazily built) table, the arm identity."""

    def __init__(self, reference: Any, spec: EtaFilterSpec, identity: str, table_grid, cache_dir) -> None:
        self.spec = spec
        self.identity = identity
        self.table_grid = table_grid
        self.cache_dir = cache_dir
        self.metric_evaluator = reference.metric_evaluator
        metric_view = self.metric_evaluator
        if metric_view.nfp != spec.nfp:
            # The reference has checked that the filter spans whole periods.
            # Preserve the map evaluation at each requested absolute eta.
            metric_view = SimpleNamespace(period=spec.period, evaluate=metric_view.evaluate)
        self.column_field = EtaFilteredColumnField(metric_view, reference.bfield_evaluator, spec)
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
        """The tabulated form, cached in the caller-specified directory."""
        if self._table is None:
            t0 = time.perf_counter()
            path = Path(self.cache_dir) / f"field_table_{self.identity[:16]}.npz" if self.cache_dir is not None else None
            if path is not None and path.exists():
                table = EtaFilterTable.load(path, metric_evaluator=self.column_field.metric_evaluator)
                if table.meta.get("arm_sha256") != self.identity:
                    raise ValueError(f"{path} was built for another arm ({table.meta.get('arm_sha256')})")
                self.table_from_cache = True
            else:
                nu, ntheta, u_max = self.table_grid
                table = EtaFilterTable.from_column_field(self.column_field, nu=nu, ntheta=ntheta, u_max=u_max,
                                                         sample_metric=self._table_metric())
                rng = np.random.default_rng(20261004)                  # the one-off equivalence check against columns
                pts = np.stack([rng.uniform(0.02, 1.0, EQUIVALENCE_POINTS), rng.uniform(0.0, 2.0 * np.pi, EQUIVALENCE_POINTS),
                                rng.uniform(0.0, 2.0 * np.pi, EQUIVALENCE_POINTS)], axis=1)
                check = table.check_against_columns(self.column_field, pts)
                table.meta = {"arm_sha256": self.identity, "equivalence_vs_columns": check,
                              "grid": list(self.table_grid)}
                if path is not None:
                    table.save(path)
                self.table_from_cache = False
            self.table_check = table.meta.get("equivalence_vs_columns")
            self.table_seconds = time.perf_counter() - t0
            self._table = table
        return self._table

    def _table_metric(self):
        if self.table_metric is None:
            from ...evaluators.jax_metric_evaluator import JaxMetricEvaluator

            self.table_metric = JaxMetricView(JaxMetricEvaluator.from_metric_evaluator(self.metric_evaluator))
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


def apply_eta_filter(reference, provenance, options):
    """Install the optional column field and its lazy table twin."""
    spec = options.eta_filter
    if spec is None:
        return
    arm = EtaFilterArm(reference, spec, arm_identity(provenance, options),
                       options.table_grid, options.table_cache_directory)
    reference.eta_filter_arm = arm
    reference._metric_batch = _filtered_metric_batch(reference, arm)
