"""Connect pure geometry and caller-owned local products in bundle stage order."""

from ...geometry_build.clipping_primitives import area_centroid, cross, inpoly, tiles as build_tiles
from ...geometry_build.dual_cells import build as build_cells
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
from scipy.special import roots_legendre

from ..._hashing import sha256_stream
from . import assemble_joint, audit_tiling, build_joint_geometry, classify
from . import close_tiny_gaps, diagnostics, diagnostics_joint
from . import geometry_plane, joint, merge, mesh, partition_baseline, gap_centroid_trace_audit
from . import footprint_repair, prepare_children, project, quadrature_plane, refine, trace_plane
from ._io import dump, load, save
from ._case import case_inputs, finish_case_record, open_case_record
from ._rays import RayCache, key, sample
from ...options import PolygonBuildOptions
from . import _geometry
from ._classes import point_in_segments, segment_in_cell
from ._repair import edges, seeds
from ._support import in_triangles, relative, stats
from ...geometry_build.trace_evaluator import Evaluator

BITS = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], bool)
FIELDS = ['maps', 'arcs', 'wallhit', 'hitell']


def _context(root, ev, options, layout_builder, shared_polygon, archive,
             prior_ledgers, log):
    rays = RayCache(root, options, shared_polygon.trace, archive, prior_ledgers)
    shared = shared_polygon.bind(options, rays.trace_sides)
    N = options.resolution
    TAU = _geometry.TAU
    HOP = TAU/N

    @lru_cache(maxsize=1)
    def build():
        return build_cells(N, layout_builder)

    @lru_cache(maxsize=2)
    def tiles(sub):
        return build_tiles(build()[1], sub)

    gu = SimpleNamespace(P=build()[0].P, build=build, tiles=tiles,
                         cross=cross, inpoly=inpoly,
                         area_centroid=area_centroid,
                         metric=_geometry.metric)

    class Lookup:
        ray = staticmethod(rays.lookup)

        def observation(self, ref, k):
            transform = shared.transform
            xy,_=transform(np.asarray(ref)[None],'half');q=seeds(xy,(k+1)*HOP)[0];z=[self.ray(tuple(q),s*HOP) for s in [-1,1]];tau=[]
            for zz in z:tau.append(None if zz is None else abs((zz['wall'][2]-q[2]+np.pi)%TAU-np.pi)/HOP if zz['hit'] else 1.)
            return q,z,tau

    x8,w8=roots_legendre(8);x8=(x8+1)/2;w8=w8/2;u8=x8[:,None];v8=x8[None,:];BQ8=np.stack([np.broadcast_to(1-u8,(8,8)),np.broadcast_to(u8*(1-v8),(8,8)),np.broadcast_to(u8*v8,(8,8))],-1).reshape(-1,3);WQ8=(w8[:,None]*w8[None,:]*u8).ravel()

    return SimpleNamespace(
        **vars(shared), N=N, NP=options.source_planes, TAU=TAU, HOP=HOP,
        SUB=options.face_subdivision, RINGS=options.rings, copies=options.copies,
        period_angle=options.period_angle, disk_angles=options.disk_angles,
        BISECT=options.bisections, wall_bisections=options.wall_bisections,
        root=root, history=root/'history', evaluator=ev, gu=gu,
        Lookup=Lookup, book=rays.book, trace_cached=rays.trace_cached,
        trace_sides=rays.trace_sides, plane_charged=rays.plane_charged,
        load=load, save=save, dump=dump, log=log, relative=relative,
        key=key, sample=sample, seeds=seeds, edges=edges,
        BITS=BITS, FIELDS=FIELDS, BQ8=BQ8, WQ8=WQ8, stats=stats,
        point_in_segments=point_in_segments, segment_in_cell=segment_in_cell,
        in_triangles=in_triangles, supplemental_ray_budget=options.supplemental_ray_budget,
        prior_ledgers=prior_ledgers, merge_fraction=options.merge_fraction,
        trace_sizing_workers=options.trace_sizing_workers,
    )


def _bind_stages(context):
    def lane(name, module, *dependencies):
        context.out = context.root/name
        return module.bind(context, *dependencies)

    meshes = lane('base_polygon', mesh)
    classification = lane('base_polygon', classify)
    trace = lane('base_polygon', trace_plane, meshes['midpoint_mesh'], meshes['plane_mesh'],
                 classification['midpoint_observe'], classification['clip_class'],
                 classification['clip_plane_seeded_wall'], classification['plane_seeds'])
    project_stage = lane('polygon_repair', project)
    post = lane('polygon_repair', footprint_repair)
    joint_stage = lane('polygon_repair', joint)
    diagnostic = lane('base_polygon', diagnostics)


    return SimpleNamespace(
        trace_plane=trace['main'],
        geometry_plane=lane('base_polygon', geometry_plane),
        quadrature_plane=lane('base_polygon', quadrature_plane, meshes['midpoint_mesh'])['main'],
        diagnostics=diagnostic['main'],
        refine=lane('base_polygon', refine, meshes['plane_mesh'], classification['clip_plane_seeded_wall']),
        partition_baseline=lane('partition_audit', partition_baseline)['main'], gap_centroid_trace_audit=lane('polygon_repair', gap_centroid_trace_audit),
        prepare_children=lane('polygon_repair', prepare_children, project_stage['remainder']),
        joint_project=joint_stage['project'],
        joint_bisect=joint_stage['bisect'],
        joint_partition=joint_stage['partition'], audit_tiling=lane('polygon_repair', audit_tiling),
        close_tiny_gaps=lane('polygon_repair', close_tiny_gaps, project_stage['remainder'])['main'],
        build_joint_geometry=lane('polygon_repair', build_joint_geometry)['main'],
        assemble_joint=lane('polygon_repair', assemble_joint)['main'],
        diagnostics_joint=lane('polygon_repair', diagnostics_joint, post['stats'])['main'],
        merge=lane('polygon_repair', merge)['main'], repaired_tiling_audit=post['main'], ray_ledger=context.book,
    )


def open_route(root, *, metric_cache, field_table, layout_builder, shared_polygon,
               options=PolygonBuildOptions(), archive=None, prior_ledgers=(), log=print):
    """Bind local stages to a saved metric and its loaded ETA table.

    The layout builder is the package Zernike core-ring layout function. The shared
    polygon module is the staged polywall numerical module. A supplied
    archive must match the continuous inputs, exact key and dense kernel.
    Prior ledgers carry the archived ray charges into the supplied budget.
    """
    root = Path(root)
    table_sha256 = field_table.archive_sha256
    if table_sha256 is None:
        raise ValueError('repaired polygon and merged wall rows requires a field table loaded from a saved archive')
    from ...evaluators.metric_evaluator import MetricEvaluator
    from ...evaluators.jax_metric_evaluator import JaxMetricEvaluator

    settings = asdict(options)
    settings['field_filter'] = field_table.spec.as_dict()
    with Path(metric_cache).open('rb') as stream:
        inputs = case_inputs(settings, table_sha256, sha256_stream(stream))
        receipt = open_case_record(root, inputs)
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as payload:
            metric = MetricEvaluator.from_cache_payload(payload, prefix='metric_evaluator_')
    ev = Evaluator.from_objects(JaxMetricEvaluator.from_metric_evaluator(metric), field_table, options)
    context = _context(root, ev, options, layout_builder, shared_polygon,
                       archive, tuple(prior_ledgers), log)
    stages = _bind_stages(context)

    def run():
        """Run the ordered base polygon and wall classification/repaired polygon and merged wall rows stages."""
        open_case_record(root, inputs)
        for directory in (root/'base_polygon'/'checkpoints', root/'polygon_repair'/'checkpoints', context.history):
            directory.mkdir(parents=True, exist_ok=True)
        planes = range(options.source_planes)
        for stage in (stages.trace_plane, stages.geometry_plane, stages.quadrature_plane):
            for k in planes:
                stage(k)
        stages.diagnostics()
        for k in planes:
            stages.refine(k)
        reports = [json.loads((root/'base_polygon'/f'refinement_{k}.json').read_text()) for k in planes]
        rebuilt = [k for report in reports for k in report['rebuild_planes']]
        dump(root/'base_polygon'/'refinement.json', dict(
            complete=True, criterion=reports[0]['criterion'],
            units=[row for report in reports for row in report['units']],
            rebuild_planes=rebuilt, ledger=stages.ray_ledger()['new_rays']))
        for k in rebuilt:
            stages.geometry_plane(k)
        if rebuilt:
            stages.diagnostics()
        stages.partition_baseline()
        for stage in (stages.gap_centroid_trace_audit, stages.prepare_children):
            for k in planes:
                stage(k)
        stages.joint_project()
        for stage in (stages.joint_bisect, stages.joint_partition):
            for k in planes:
                stage(k)
        stages.audit_tiling(discover=True)
        stages.close_tiny_gaps()
        stages.audit_tiling()
        for k in planes:
            stages.build_joint_geometry(k)
        for stage in (stages.assemble_joint, stages.diagnostics_joint, stages.merge, stages.repaired_tiling_audit):
            stage()
        arrays = root/'polygon_repair'/'arrays'
        products = dict(geometry=arrays/f'polygon_geometry_N{options.resolution}.npz',
                        merged=arrays/f'merged_N{options.resolution}_c{options.merge_fraction:g}.npz',
                        face_flux=arrays/f'face_flux_N{options.resolution}.npz',
                        wall_rows=arrays/f'wall_rows_N{options.resolution}.npz')
        return dict(products, case_record=finish_case_record(receipt, inputs, products))

    stages.run = run
    return stages


