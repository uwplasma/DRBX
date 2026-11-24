"""Local cubic source phases, owner reductions and indexed publication."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..options import DEFAULT_POLYROUTE_OPTIONS, PolygonBuildOptions, boundary_bisection_count
from ..operator_build.parallel._case import CASE_NAME

if TYPE_CHECKING:
    from ..evaluators.eta_filtered_field import EtaFilterSpec

from . import owner_reductions as owners
from .clip_entry import entry
from .clipping import triangle_range_numpy
from .compiled_clipping import triangle_range
from .compiled_polygons import Polygons
from .context import context
from .indexed_output import Writer, publish_fits
from .numerics import fit_task, integrate, metric_task, observe, source_plan
from .production_splice import ProductionSplice, validate_polygon_case
from .rays import Rays
from .storage import FrameWriter, dump, fingerprint


@dataclass(frozen=True)
class GeometryOptions:
    output: Path
    metric_cache: Path
    field_table: Path
    polygon_geometry: Path
    polygon_merged: Path
    symmetry: str = DEFAULT_POLYROUTE_OPTIONS.symmetry
    field_periods: int = DEFAULT_POLYROUTE_OPTIONS.field_periods
    planes: int = DEFAULT_POLYROUTE_OPTIONS.resolution
    resolution: int = DEFAULT_POLYROUTE_OPTIONS.resolution
    storage_bins: int = 64
    field_filter: EtaFilterSpec | None = DEFAULT_POLYROUTE_OPTIONS.field_filter
    angular_subdivisions: int = DEFAULT_POLYROUTE_OPTIONS.disk_angles // DEFAULT_POLYROUTE_OPTIONS.resolution
    trace_substeps: int = 64
    support_widths: tuple[int, ...] = (25, 40, 64)
    expand_widths: tuple[int, ...] = (40, 64)
    reconstruction_rule: str = 'expand_fit_support'
    geometry_capacity: int = 128
    compiled_clipping: bool = True
    polygon_rings: int = DEFAULT_POLYROUTE_OPTIONS.rings
    geometry_reference_key_table: Path | None = None
    polygon_receipt: Path | None = None
    clip_bisections: int | None = None
    workers: int = 1

    def __post_init__(self):
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError('Geometry workers must be a positive integer')

    @classmethod
    def for_polygon(cls, polygon: PolygonBuildOptions, **inputs):
        bisections = polygon.bisections
        requested = inputs.pop('clip_bisections', None)
        if requested is not None and requested != bisections:
            raise ValueError('Geometry clip_bisections must match polygon boundary bisections')
        return cls(resolution=polygon.resolution, planes=polygon.resolution,
                   field_periods=polygon.field_periods, symmetry=polygon.symmetry,
                   field_filter=polygon.field_filter, polygon_rings=polygon.rings,
                   angular_subdivisions=polygon.disk_angles // polygon.resolution,
                   clip_bisections=bisections, **inputs)

    @property
    def bisections(self):
        return boundary_bisection_count(self.resolution, self.clip_bisections)


def configuration(options):
    n = options.resolution
    p = options.field_periods if options.symmetry == 'nfp' else 1
    if options.field_filter is None:
        raise ValueError(f'Raw geometry is not supported: field_filter=None, field_table={options.field_table!r}')
    if options.field_table is None:
        raise ValueError(f'Filtered geometry requires field_table: field_filter={options.field_filter!r}, field_table=None')
    if options.symmetry not in ('nfp', 'none') or options.field_periods < 1:
        raise ValueError('Specify nfp or none and a positive field-period count')
    if options.planes != n or n < 1 or n % p:
        raise ValueError('Cubic layout requires E=N and E divisible by the replication count')
    if min(options.storage_bins, options.angular_subdivisions, options.trace_substeps,
           options.geometry_capacity) < 1:
        raise ValueError('Counts and evaluator capacity must be positive')
    bisections = options.bisections
    if not 2 <= options.polygon_rings < n:
        raise ValueError('Production polygon rings must leave a positive inner radius')
    files = {name: Path(path) for name in ('metric_cache', 'field_table',
             'polygon_geometry', 'polygon_merged')
             if (path := getattr(options, name)) is not None}
    config = dict(N=n, eta_planes=options.planes, symmetry=options.symmetry,
                  field_periods=options.field_periods, replication_periods=p,
                  files=files,
                  filter_spec=options.field_filter, trace_substeps=options.trace_substeps,
                  angular_subdivisions=options.angular_subdivisions, clip_bisections=bisections,
                  support_widths=options.support_widths, expand_widths=options.expand_widths,
                  config=dict(reconstruction_rule=options.reconstruction_rule,
                              geometry_capacity=options.geometry_capacity))
    receipt = options.polygon_receipt
    if receipt is None:
        receipt = files['polygon_geometry'].parent / CASE_NAME
    config.update(polygon_rings=options.polygon_rings, geometry_reference_key_table=options.geometry_reference_key_table,
                  polygon_receipt=Path(receipt))
    config['fingerprint'] = fingerprint(dict(config, filter_spec=repr(options.field_filter),
                                         storage_bins=options.storage_bins,
                                         compiled_clipping=options.compiled_clipping))
    return config


def clipping_context(observation, compiled):
    return entry(observation, compiled)


def clip_leaf(data, bounds):
    if data['compiled_polygons'] is None:
        return triangle_range_numpy(data, bounds)
    return triangle_range(data, bounds)


def source_plane(m, ns, ev, rays, out, k, compiled):
    frame = FrameWriter(out)
    for sign in (1, -1):
        cuts = None
        for order in (3, 8):
            plan = source_plan(ns, k, order, cuts)
            if cuts is not None:
                plan['cuts'] = cuts
            parent = f'q{order}_{sign}'
            frame.write(f'{parent}/plan', plan, dict(order=order, sign=sign))
            cut = np.zeros(ns['P'], bool)
            for lo in range(0, len(plan['owner']), 64):
                bounds = (lo, min(lo + 64, len(plan['owner'])))
                result = integrate(ns, ev, rays, plan, sign, order, bounds)
                frame.write(f'{parent}/leaf_{lo:09d}', result, dict(bounds=bounds))
                if order == 3:
                    cut[result['owner'][result['hit']]] = True
            cuts = np.flatnonzero(cut)
    observation = observe(ns, rays, k, m['angular_subdivisions'], m['clip_bisections'])
    frame.write('observation', observation, dict(plane=k))
    data = clipping_context(observation, compiled)
    n = len(observation['tris'])
    for lo in range(0, n, 8):
        bounds = (lo, min(lo + 8, n))
        label = f'leaf_{lo:09d}'
        clipped = clip_leaf(data, bounds)
        frame.write(f'clip/{label}', clipped, dict(bounds=bounds))
        frame.write(f'metric/{label}', metric_task(ns, ev, k, clipped), dict(bounds=bounds))


def assembly_plan(m, ns, counts, physical_counts, polygon_prefix, storage_bins):
    n = m['N']
    p = m['replication_periods']
    k = n // p
    ad = [x['ad_per_period'] for x in counts]

    def pref(values, start=0):
        result = [start]
        for value in values:
            result.append(result[-1] + value)
        return result

    physical = pref([int(x['count']) for x, _ in physical_counts])
    return dict(N=n, E=n, p=p, K=k, plane_nodes=ns['P'], NN=ns['NN'], storage_bins=storage_bins,
                ad_per_period=ad, face_prefix=pref([p*x for x in ad], polygon_prefix),
                physical_wall_rows=physical[-1], physical_prefixes=physical, polygon_prefix=polygon_prefix,
                coverage=counts)


def build_geometry(options: GeometryOptions) -> Path:
    """Build cubic partitioned geometry in a new folder and return that folder.

    Production polygon paths are the geometry and merged products of repaired polygon and merged wall rows.
    The polygon_receipt defaults to the case record beside those products.
    The optional geometry_reference_key_table NPZ holds source keys and explicit IDs.
    Source planes are generated by the local tracer.
    """
    import jax

    if not jax.config.x64_enabled or jax.default_backend() != 'cpu':
        raise ValueError('The caller must select JAX CPU float64')
    m = configuration(options)
    validate_polygon_case(m, m['polygon_receipt'])
    ns, ev = context(m)
    m['filter_spec'] = ev.filter_spec
    output = Path(options.output).resolve()
    n = m['N']
    k = n // m['replication_periods']
    b = m['files']
    production = ProductionSplice(b, ns['NN'], options.geometry_reference_key_table)
    if options.workers > 1:
        from .sharded import build_sharded

        return build_sharded(options, m, ns, production)
    output.mkdir(parents=True)
    sources = []
    compiled = Polygons() if options.compiled_clipping else None
    for plane in range(k):
        base = output / 'source' / f'p{plane:03d}'
        rays = Rays(ev, options.trace_substeps)
        try:
            source_plane(m, ns, ev, rays, base, plane, compiled)
        finally:
            rays.close()
        sources.append(base)
    censuses = [owners.census(m, ns, plane, base) for plane, base in enumerate(sources)]
    census_frames = FrameWriter(output / 'census')
    for plane, (values, metadata) in enumerate(censuses):
        census_frames.write(f'p{plane:03d}', values, metadata)

    wall_rows = len(production.wall['wall'])
    chunks = [(lo, min(lo+8192, wall_rows)) for lo in range(0, wall_rows, 8192)]
    physical = [owners.physical_count(lo, hi) for lo, hi in chunks]
    for (lo, _), (values, metadata) in zip(chunks, physical):
        census_frames.write(f'wall_counts/chunk_{lo:012d}', values, metadata)
    polygon_count = len(production.faces)
    plan = assembly_plan(m, ns, [meta for _, meta in censuses], physical, polygon_count, options.storage_bins)
    plan['original_face_prefix'] = np.cumsum([0, *(plan['p'] * row['source_records'] for row in plan['coverage'])])

    dump(output/'assembly_plan.json', plan)
    writer = Writer(output, m, plan)
    try:
        for plane, base in enumerate(sources):
            writer.part_start(f'plane{plane:03d}')
            owners.plane(m, plan, writer, base, plane, censuses[plane][0], production)
        for lo in range(0, polygon_count, 8192):
            writer.part_start(f'polygon_{lo:012d}')
            owners.polygon(m, plan, writer, lo, min(lo+8192, polygon_count), production)
        for (lo, hi), offset in zip(chunks, plan['physical_prefixes']):
            writer.part_start(f'wall_{lo:012d}')
            owners.physical_wall(m, plan, writer, lo, hi, offset, production)
        geometry = writer.finish()
        publish_fits(output, m, geometry, writer.keys(), fit_task)
        writer.publish_reader()
    finally:
        writer.close()
    return output
