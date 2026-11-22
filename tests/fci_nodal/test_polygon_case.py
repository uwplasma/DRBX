"""repaired polygon and merged wall rows case metadata must agree and its splice files must retain their bytes."""

import hashlib
import json
import multiprocessing as mp
import os
import signal
from time import monotonic, sleep
from types import SimpleNamespace

import numpy as np
import pytest


def _stub_trace(evaluator, seeds, delta, *, steps):
    evaluator.beta_min = min(evaluator.beta_min, float(seeds[:, 0].min()))
    evaluator.beta_max = max(evaluator.beta_max, float(seeds[:, 0].max()))
    evaluator.max_u = max(evaluator.max_u, float(seeds[:, 0].max()))
    return {'endpoint': seeds + delta[:, None],
            'batch_size': np.full(len(seeds), evaluator.trace_batch_size),
            'steps': np.full(len(seeds), steps)}


def test_polygon_trace_dispatch_keeps_batch_and_ledger_order(runtime_module, tmp_path):
    from functools import partial
    from types import SimpleNamespace

    ray_cache = runtime_module('operator_build.parallel._rays').RayCache
    option_type = runtime_module().PolygonBuildOptions
    make_evaluator = partial(SimpleNamespace, beta_min=np.inf, beta_max=-np.inf, max_u=0.)
    seeds = np.column_stack((np.arange(1., 41.) / 50., np.zeros((40, 2))))
    roots = [tmp_path/'serial', tmp_path/'parallel']
    observations = []
    for root, workers in zip(roots, (1, 3), strict=True):
        cache = ray_cache(root, option_type(workers=workers), _stub_trace)
        evaluator = make_evaluator()
        with cache.tracing(make_evaluator):
            first = cache.trace_cached(evaluator, seeds[:29], .1, 'first')
            second = cache.trace_cached(evaluator, seeds[20:], .1, 'second')
        observations.append((first, second, cache.book(), vars(evaluator)))
    for before, after in zip(observations[0][:2], observations[1][:2], strict=True):
        assert before.keys() == after.keys()
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
    assert observations[0][2:] == observations[1][2:]
    serial, parallel = ({path.name: path.read_bytes() for path in (root/'rays').iterdir()}
                        for root in roots)
    assert serial == parallel


def test_polygon_plane_commit_keeps_serial_files(runtime_module, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from threading import Event
    from types import SimpleNamespace

    driver = runtime_module('operator_build.parallel.driver')
    execution = runtime_module('operator_build.parallel._plane_execution')
    ray_cache = runtime_module('operator_build.parallel._rays').RayCache
    io = runtime_module('operator_build.parallel._io')
    options = runtime_module().PolygonBuildOptions(ray_budget=43)
    module = SimpleNamespace(trace=_stub_trace)

    def evaluator():
        return SimpleNamespace(beta_min=np.inf, beta_max=-np.inf, max_u=0.)

    def run_unit(root, rays, ev, log, phase, plane):
        seeds = np.column_stack((np.arange(1., 22.)/22, np.zeros(21), np.full(21, plane+.5)))
        first = phase == 'base'
        if not first:
            rays.check_projection_budget(8+plane)
            source = root/f'checkpoints/base_{plane}.json'
            archive = root/f'history/base_{plane}.json'
            archive.parent.mkdir(exist_ok=True)
            source.rename(archive)
        result = rays.trace_cached(ev, seeds[:13] if first else seeds[8:], .1, f'{phase} strip{plane}')
        io.save(root/f'outputs/{phase}_{plane}.npz', **result)
        io.dump(root/f'checkpoints/{phase}_{plane}.json', dict(plane=plane))
        log(f'{phase} plane {plane} checkpoint')

    def context(root, ev, options, layout_builder, shared, rays, prior, log):
        return root, rays, ev, log

    def bind(context):
        _, rays, _, log = context
        return SimpleNamespace(base=lambda k: run_unit(*context, 'base', k),
                               refine=lambda k: run_unit(*context, 'refine', k),
                               log=log, ray_ledger=rays.book)

    monkeypatch.setattr(driver, '_context', context)
    monkeypatch.setattr(driver, '_bind_stages', bind)
    states = []
    for parallel in (False, True):
        root = tmp_path/('parallel' if parallel else 'serial')
        root.mkdir()
        rays, ev, logs = ray_cache(root, options, _stub_trace), evaluator(), []
        stages = bind((root, rays, ev, logs.append))
        for phase in ('base', 'refine'):
            if not parallel:
                driver._run_phase(stages, phase, range(2))
                continue
            second_finished = Event()
            completion = []

            def work(root, private, name, plane, snapshot, quota, extrema):
                worker = options, None, module, evaluator(), None
                if plane == 0:
                    assert second_finished.wait(5)
                result = execution.stage_plane(root, private, name, plane, json.loads(snapshot.read_text()),
                                               quota, extrema, worker)
                assert (result is None) == (name == 'refine' and plane == 1)
                completion.append(plane)
                if plane == 1:
                    second_finished.set()
                return result

            with ThreadPoolExecutor(2) as pool:
                publish = partial(execution.run_plane_phase, pool, root, rays, ev, logs.append, work=work)
                driver._run_phase(stages, phase, range(2), publish)
            assert completion == [1, 0]
        assert logs == [
            line
            for phase, plane, count in [('base', 0, 13), ('base', 1, 26), ('refine', 0, 34), ('refine', 1, 42)]
            for line in (f'{phase} plane {plane} checkpoint', f'{phase} plane {plane} complete; ledger {count}/43')
        ]
        files = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        states.append((files, rays.book(), (ev.beta_min, ev.beta_max, ev.max_u), logs))
    assert states[0] == states[1]


_test_tracers = None


def _stub_tracer_evaluator(root):
    (root/f'tracer_{os.getppid()}_{os.getpid()}').touch()
    return SimpleNamespace(beta_min=np.inf, beta_max=-np.inf, max_u=0.)


def _start_stub_tracers(root):
    from functools import partial

    from drbx.fci_nodal.operator_build.parallel._rays import trace_pool

    global _test_tracers
    (root/f'plane_{os.getpid()}').touch()
    _test_tracers = trace_pool(2, _stub_trace, partial(_stub_tracer_evaluator, root))
    return _test_tracers


def _nested_stub_plane(root, private, phase, plane, snapshot_path, quota, extrema):
    from drbx.fci_nodal.operator_build.parallel._plane_execution import _PlaneRays, _PlaneResult, _extrema
    from drbx.fci_nodal.options import PolygonBuildOptions

    snapshot = json.loads(snapshot_path.read_text())
    options = PolygonBuildOptions(workers=2)
    evaluator = SimpleNamespace(beta_min=extrema[0], beta_max=extrema[1], max_u=extrema[2])
    rays = _PlaneRays(private, root/'rays', options, _stub_trace, snapshot, quota, _test_tracers)
    seeds = np.column_stack((np.arange(1., 26.)/26, np.zeros(25), np.full(25, plane+.5)))
    rays.trace_cached(evaluator, seeds, .1, f'{phase} strip{plane}')
    if phase == 'error':
        # Both nested pools must start before the first plane fails.
        (root/f'finished_{plane}').touch()
        if plane == 0:
            deadline = monotonic()+8.
            while not (root/'finished_1').exists() and monotonic() < deadline:
                sleep(.02)
            assert (root/'finished_1').exists()
            raise ValueError('forced stub plane failure')
    return _PlaneResult(0, rays.book()['calls'], _extrema(evaluator), (), (), ())


def _exercise_nested_plane_phase(root):
    from drbx.fci_nodal.operator_build.parallel._plane_execution import run_plane_phase
    from drbx.fci_nodal.operator_build.parallel._plane_pool import PlanePool
    from drbx.fci_nodal.operator_build.parallel._rays import RayCache
    from drbx.fci_nodal.options import PolygonBuildOptions

    def no_fallback(plane):
        raise AssertionError(f'unexpected serial fallback for plane {plane}')

    for phase in ('success', 'error'):
        folder = root/phase
        folder.mkdir()
        rays = RayCache(folder, PolygonBuildOptions(), _stub_trace)
        evaluator = SimpleNamespace(beta_min=np.inf, beta_max=-np.inf, max_u=0.)
        try:
            with PlanePool(2, _start_stub_tracers, (folder,)) as pool:
                run_plane_phase(pool, folder, rays, evaluator, print, phase, range(2),
                                no_fallback, _nested_stub_plane)
        except RuntimeError as error:
            assert phase == 'error' and 'ValueError: forced stub plane failure' in str(error)
            assert not getattr(error, '__notes__', ())
        else:
            assert phase == 'success'
        assert rays.book()['new_rays'] == (50 if phase == 'success' else 0)
        assert not mp.active_children()


def test_polygon_nested_plane_workers_return_and_exit(tmp_path):
    context = mp.get_context('spawn')
    stage = context.Process(target=_exercise_nested_plane_phase, args=(tmp_path,))
    stage.start()
    stage.join(25.)
    try:
        assert stage.exitcode == 0, 'nested polygon phase failed or did not return within 25 s'
        for phase in ('success', 'error'):
            folder = tmp_path/phase
            lanes = list(folder.glob('plane_*'))
            assert len(lanes) == 2
            for lane in lanes:
                tracers = list(folder.glob(f'tracer_{lane.name.split("_")[1]}_*'))
                assert len(tracers) == 2
                for child in [lane, *tracers]:
                    with pytest.raises(ProcessLookupError):
                        os.kill(int(child.name.rsplit('_', 1)[1]), 0)
    except BaseException:
        if stage.is_alive():
            stage.kill()
            stage.join(3.)
        for marker in tmp_path.glob('*/plane_*'):
            # Reap the separate lane groups on a failed test as well.
            try:
                os.killpg(int(marker.name.rsplit('_', 1)[1]), signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
    finally:
        if not stage.is_alive():
            stage.close()


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
