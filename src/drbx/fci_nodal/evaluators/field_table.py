"""Eta-filter table construction from the P reference and EtaFilterArm paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .._hashing import sha256_stream
from .magnetic_field_evaluator import bfield_evaluator_from_makegrid
from .metric_evaluator import MetricEvaluator
from .eta_filtered_field import EtaFilteredColumnField, EtaFilterTable, JaxMetricView
from .jax_metric_evaluator import JaxMetricEvaluator

from ..options import MetricOptions

EQUIVALENCE_POINTS = 256


FIELD_TABLE_TOROIDAL_METHODS = ("spline", "compact_hermite_c3")


def build_field_table(options: MetricOptions) -> Path | None:
    """Return the table path, or None when the caller selects the raw field."""
    spec = options.field_filter
    if spec is None:
        return None
    if options.metric_cache_path is None or options.field_table_dir is None:
        raise ValueError("metric_cache_path and field_table_dir are required for a filtered table")

    metric_cache = options.metric_cache_path.resolve()
    makegrid = options.makegrid_path.resolve()
    bfield_toroidal = options.bfield_toroidal
    if bfield_toroidal not in FIELD_TABLE_TOROIDAL_METHODS:
        raise ValueError(f"unsupported toroidal field interpolation: {bfield_toroidal!r}")
    currents = (
        None if options.makegrid_currents is None
        else np.asarray(options.makegrid_currents, dtype=np.float64)
    )
    nu, ntheta, u_max = options.table_grid
    with np.load(metric_cache, allow_pickle=False) as cached:
        evaluator = MetricEvaluator.from_cache_payload(cached, prefix="metric_evaluator_")
    bfield = bfield_evaluator_from_makegrid(
        makegrid, currents=currents, method="cubic",
        toroidal_method=bfield_toroidal, extrapolate=False,
    )
    column_field = EtaFilteredColumnField(evaluator, bfield, spec)

    with metric_cache.open("rb") as stream:
        map_sha256 = sha256_stream(stream)
    makegrid_sha256 = options.makegrid_sha256
    if makegrid_sha256 is None:
        with makegrid.open("rb") as stream:
            makegrid_sha256 = sha256_stream(stream)
    core = {"map_sha256": map_sha256, "makegrid_sha256": makegrid_sha256,
            "makegrid_currents": [float(x) for x in (bfield.currents if currents is None else currents)],
            "bfield_toroidal": bfield_toroidal,
            "eta_filter": spec.as_dict(), "table_grid": list(options.table_grid)}
    identity = hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()
    path = options.field_table_dir.resolve() / f"field_table_{identity[:16]}.npz"
    if path.exists():
        table = EtaFilterTable.load(path, metric_evaluator=column_field.metric_evaluator)
        if table.meta.get("arm_sha256") != identity:
            raise ValueError(f"{path} was built for another arm ({table.meta.get('arm_sha256')})")
        return path

    sample_metric = JaxMetricView(JaxMetricEvaluator.from_metric_evaluator(column_field.metric_evaluator))
    table = EtaFilterTable.from_column_field(
        column_field, nu=nu, ntheta=ntheta, u_max=u_max, sample_metric=sample_metric,
    )
    rng = np.random.default_rng(20261004)                  # the one-off equivalence check against columns
    pts = np.stack([rng.uniform(0.02, 1.0, EQUIVALENCE_POINTS), rng.uniform(0.0, 2.0 * np.pi, EQUIVALENCE_POINTS),
                    rng.uniform(0.0, 2.0 * np.pi, EQUIVALENCE_POINTS)], axis=1)
    check = table.check_against_columns(column_field, pts)
    table.meta = {"arm_sha256": identity, "equivalence_vs_columns": check, "grid": list(options.table_grid)}
    table.save(path)
    return path
