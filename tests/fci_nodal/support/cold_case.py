"""Read and reuse the current-format, one-owner reference case."""

import os
from pathlib import Path

import pytest

from .backend import backend_module


REFERENCE_CASE_ENV = 'DRBX_FCI_NODAL_REFERENCE_CASE'


def current_case_directory():
    value = os.environ.get(REFERENCE_CASE_ENV)
    if not value:
        pytest.skip(f'Set {REFERENCE_CASE_ENV} to a complete one-owner N16 reference case')
    case = Path(value).resolve(strict=True)
    receipts = backend_module('case._receipts')
    for name in receipts.PREDECESSORS:
        stage = read_stage(case, name)
        receipts.verify_code(name, receipts.binding(stage))
        for path in receipts.receipt_products(stage).values():
            path.resolve(strict=True)
    geometry = receipts.receipt_products(read_stage(case, 'geometry'))['indexed_geometry']
    if receipts.read_json(geometry/'geometry.json')['schema'] != backend_module('geometry_build.reader').GEOMETRY_SCHEMA:
        raise ValueError('Reference case requires the current indexed geometry schema')
    owner = receipts.binding(read_stage(case, 'owner_inputs'))['options']
    if owner['owner_count'] != 1 or owner['layout'][:2] != [16, 16]:
        raise ValueError('Reference case requires one owner at resolution 16 with 16 eta planes')
    return case


def read_stage(case, stage):
    return backend_module('case._receipts').stage_receipt(case, stage)[0]


def stage_products(case, stage):
    return backend_module('case._receipts').receipt_products(read_stage(case, stage))


def cold_options(case, *, owners=1):
    """Restore current typed options from the case's own receipts."""
    receipts = backend_module('case._receipts')
    opts = backend_module('options')

    def options(stage):
        return receipts.binding(read_stage(case, stage))['options']

    def record(cls, values):
        return cls(**{key: tuple(value) if isinstance(value, list) else value
                      for key, value in values.items()})

    metric = options('metric').copy()
    filter_spec = backend_module('evaluators.eta_filtered_field').EtaFilterSpec.from_mapping(metric['field_filter'])
    metric['field_filter'] = filter_spec
    if directory := os.environ.get('DRBX_FCI_NODAL_DEVICE_DIR'):
        for key in ('makegrid_path', 'vessel_path'):
            if metric[key] is not None:
                metric[key] = Path(directory) / Path(metric[key]).name

    polygon = dict(options('polygon'), field_filter=filter_spec)
    geometry = dict(options('geometry'), field_filter=filter_spec)
    parallel = dict(options('parallel_operators'), filter_spec=filter_spec, owners=owners)
    perpendicular = dict(options('perpendicular_operators'), eta_filter=filter_spec, makegrid=metric['makegrid_path'])
    # The stage receipt records owner_count alongside the builder's options.
    perpendicular.pop('owner_count')
    runtime = dict(options('mms_run')['runtime'], owner_count=owners, cpu_device_count=owners)
    return backend_module('case').CaseOptions(
        metric=record(opts.MetricOptions, metric),
        polygon=record(opts.PolygonBuildOptions, polygon),
        geometry=record(backend_module('geometry_build').GeometryOptions, geometry),
        parallel=record(backend_module('operator_build.parallel').ParallelBuildOptions, parallel),
        perpendicular=record(backend_module('operator_build.perp').PerpendicularBuildOptions, perpendicular),
        runtime=record(opts.RuntimeOptions, runtime), owners=owners,
    )


def verified_adoptions(case):
    """Verify the early reference receipts and offer those products to the driver."""
    receipts = backend_module('case._receipts')
    adoptions, stamps = {}, {}
    for name, predecessors in receipts.PREDECESSORS.items():
        stage, stamp = receipts.stage_receipt(case, name)
        identity = receipts.binding(stage)
        assert stage['stage'] == name
        assert identity['input_receipts'] == [stamps[key] for key in predecessors]
        assert receipts.file_receipts(identity['input_paths']) == identity['input_files']
        products = receipts.receipt_products(stage)
        assert receipts.file_receipts(products.values()) == stage['output_files']
        stamps[name] = stamp
        adoptions[name] = dict(products, origin_receipt=Path(stamp['receipt']))
        if name == 'geometry':
            break
    return adoptions


def mms_profile(options, reference):
    receipt = read_stage(reference, 'mms_run')
    profile = backend_module('case._receipts').binding(receipt)['options']
    return dict(profile, runtime=options.runtime)
