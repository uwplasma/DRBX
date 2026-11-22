"""repaired polygon and merged wall rows case metadata must agree and its splice files must retain their bytes."""

import hashlib
import json

import numpy as np
import pytest


def test_polygon_case_binds_splice_files(runtime_module, tmp_path):
    from dataclasses import asdict
    from dataclasses import replace

    polygon = runtime_module('operator_build.parallel')
    record = runtime_module('operator_build.parallel._case')
    geometry = runtime_module('geometry_build')
    config_module = runtime_module('geometry_build.build')
    splice = runtime_module('geometry_build.production_splice')
    filter_spec = runtime_module('evaluators.eta_filtered_field').EtaFilterSpec()
    options = polygon.PolygonBuildOptions(field_filter=filter_spec)
    stage = tmp_path/'polygon_repair'
    arrays = stage/'arrays'
    arrays.mkdir(parents=True)
    products = {role:arrays/f'{role}.npz' for role in record.PRODUCT_ROLES}
    for role, path in products.items():
        path.write_bytes(role.encode())
    for name in ('geometry', 'merge'):
        (stage/f'{name}_N{options.resolution}.json').write_text(f'{{"N":{options.resolution}}}')
    table = tmp_path/'table.npz'
    table.write_bytes(b'field table')
    metric = tmp_path/'metric.npz'
    metric.write_bytes(b'metric cache')
    settings = asdict(options)
    settings['field_filter'] = filter_spec.as_dict()
    opts = geometry.GeometryOptions.for_polygon(
        options, output=tmp_path/'output', metric_cache=metric, field_table=table,
        polygon_geometry=products['geometry'],
        polygon_merged=products['merged'],
    )
    config = config_module.configuration(opts)
    with pytest.raises(ValueError, match='requires a polygon case record'):
        splice.validate_polygon_case(config, config['polygon_receipt'])
    receipt = record.write_case_record(arrays, settings, hashlib.sha256(table.read_bytes()).hexdigest(), products,
                                       metric_cache_sha256=record.file_sha256(metric))
    assert receipt == config['polygon_receipt']
    splice.validate_polygon_case(config, receipt)
    with pytest.raises(ValueError, match='repaired polygon and merged wall rows rings'):
        splice.validate_polygon_case(dict(config, polygon_rings=options.rings-1), receipt)
    with pytest.raises(ValueError, match='Raw geometry'):
        config_module.configuration(replace(opts, field_filter=None))
    with pytest.raises(ValueError, match='boundary bisections must be a nonnegative integer'):
        config_module.configuration(replace(opts, clip_bisections=-1))
    table.write_bytes(b'other field table')
    with pytest.raises(ValueError, match='field table SHA-256'):
        splice.validate_polygon_case(config, receipt)
    table.write_bytes(b'field table')
    products['merged'].write_bytes(b'merged stream replaced in place')
    with pytest.raises(ValueError, match='merged SHA-256'):
        splice.validate_polygon_case(config, receipt)
    products['merged'].write_bytes(b'merged')

    for role in ('geometry', 'merged'):
        files = dict(config['files'], **{f'polygon_{role}':arrays/f'other_{role}.npz'})
        with pytest.raises(ValueError, match=f'repaired polygon and merged wall rows {role} product'):
            splice.validate_polygon_case(dict(config, files=files), receipt)
    baseline = receipt.read_text()
    for key, value in (('N',32), ('rings',8), ('field_periods',1), ('symmetry','none')):
        receipt.write_text(json.dumps(dict(json.loads(baseline), **{key:value})))
        with pytest.raises(ValueError, match=f'repaired polygon and merged wall rows {key}'):
            splice.validate_polygon_case(config, receipt)
    receipt.write_text(baseline)
    for name in ('geometry', 'merge'):
        sidecar = stage/f'{name}_N{options.resolution}.json'
        original = sidecar.read_text()
        sidecar.write_text('{"N":32}')
        with pytest.raises(ValueError, match=f'repaired polygon and merged wall rows {name} resolution'):
            splice.validate_polygon_case(config, receipt)
        sidecar.write_text(original)
    with pytest.raises(ValueError, match='Filtered geometry requires field_table'):
        config_module.configuration(replace(opts, field_table=None))
    metric.write_bytes(b'different metric')
    with pytest.raises(ValueError, match='metric cache SHA-256'):
        splice.validate_polygon_case(config, receipt)
    metric.write_bytes(b'metric cache')
    receipt.write_text(json.dumps(dict(json.loads(baseline), schema='drbx.pw05-case-v2')))
    with pytest.raises(ValueError, match='case record schema'):
        splice.validate_polygon_case(config, receipt)


def test_polygon_root_matches_original_inputs_before_reuse(tmp_path, runtime_module):
    record = runtime_module('operator_build.parallel._case')
    options = dict(resolution=16, rings=9, symmetry='nfp', field_periods=4)
    inputs = record.case_inputs(options, 'table', 'metric')
    path = record.open_case_record(tmp_path, inputs)
    assert json.loads(path.read_text())['products'] == {}
    assert record.open_case_record(tmp_path, inputs) == path
    for key in ('field_table_sha256', 'metric_cache_sha256', 'route_options'):
        with pytest.raises(ValueError, match=f'repaired polygon and merged wall rows root inputs differ for {key}'):
            record.open_case_record(tmp_path, dict(inputs, **{key:'changed'}))
    files = {role:path.parent/f'{role}.npz' for role in record.PRODUCT_ROLES}
    for role, file in files.items():
        file.write_text(role)
    record.finish_case_record(path, inputs, files)
    assert set(json.loads(path.read_text())['products']) == set(files)
    files['merged'].write_text('in-place replacement')
    with pytest.raises(ValueError, match='repaired polygon and merged wall rows completed products changed'):
        record.open_case_record(tmp_path, inputs)


def test_polygon_bisection_option_reaches_geometry(tmp_path, runtime_module):
    opts = runtime_module('options')
    geometry = runtime_module('geometry_build')
    polygon = opts.PolygonBuildOptions(boundary_bisections=6)
    paths = dict(output=tmp_path, metric_cache=tmp_path, field_table=tmp_path,
                 polygon_geometry=tmp_path, polygon_merged=tmp_path)
    assert geometry.GeometryOptions.for_polygon(polygon, **paths).clip_bisections == 6
    with pytest.raises(ValueError, match='clip_bisections must match'):
        geometry.GeometryOptions.for_polygon(polygon, clip_bisections=12, **paths)
    assert geometry.GeometryOptions(**paths).bisections == opts.PolygonBuildOptions().bisections == 12
    assert geometry.GeometryOptions(**paths, resolution=32).bisections == opts.PolygonBuildOptions(resolution=32).bisections == 10


def test_geometry_field_refuses_incompatible_table(tmp_path, runtime_module):
    from dataclasses import replace

    context = runtime_module('geometry_build.context').context
    field = runtime_module('evaluators.eta_filtered_field')
    spec = field.EtaFilterSpec()
    path = tmp_path/'tiny.npz'
    values = np.zeros((5, 4, spec.n_coefficients, 3))
    field.EtaFilterTable(values, spec, 1.04).save(path)
    config = dict(files={'field_table':path}, filter_spec=replace(spec,max_harmonic_per_period=2),
                  field_periods=spec.nfp)
    with pytest.raises(ValueError, match='Geometry field table spec'):
        context(config)
    with pytest.raises(ValueError, match='Geometry field table periods'):
        context(dict(config, filter_spec=spec, field_periods=1))
