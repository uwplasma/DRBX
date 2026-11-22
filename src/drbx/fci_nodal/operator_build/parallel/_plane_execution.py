"""Stage independent polygon planes and publish each phase in serial order."""

from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ._io import dump
from ._plane_pool import PlanePool
from ._rays import RayCache, merge_extrema, trace_pool
from .job_storage import guard_rss
from ...geometry_build.trace_evaluator import Evaluator


class _PlaneQuotaExceeded(Exception):
    """The remaining budget must be resolved in serial order."""


class _PlaneRays(RayCache):
    def __init__(self, root, shared_rays, options, tracer, snapshot, quota, pool):
        super().__init__(root, options, tracer)
        self.ledger = snapshot
        self.shared_rays = shared_rays
        self.first_call = len(snapshot['calls'])
        self.limit = snapshot['new_rays'] + quota
        self._pool = pool
        for call in snapshot['calls']:
            self._index_call(call)

    def _file_path(self, filename):
        if int(filename.split('_', 1)[0]) < self.first_call:
            return self.shared_rays / filename
        return super()._file_path(filename)

    def _write_ledger(self, ledger):
        # Only the coordinator writes the durable ledger.
        pass

    def check_call_budget(self, count):
        if self.ledger['new_rays'] + count > self.limit:
            raise _PlaneQuotaExceeded

    def check_projection_budget(self, count):
        self.check_call_budget(count)


def _extrema(evaluator):
    return evaluator.beta_min, evaluator.beta_max, evaluator.max_u


def _make_evaluator(metric_cache, field_table, options, rss_limit_bytes):
    guard_rss(rss_limit_bytes, label='Polygon tracer')
    return Evaluator(metric_cache, field_table, options)


_worker = None


def initialize_plane(metric_cache, field_table, options, layout_builder, module_name,
                     tracer_children, rss_limit_bytes):
    global _worker
    guard_rss(rss_limit_bytes, label='Polygon plane')
    module = import_module(module_name)
    evaluator = Evaluator(metric_cache, field_table, options)
    pool = None
    if tracer_children > 1:
        factory = partial(_make_evaluator, metric_cache, field_table, options, rss_limit_bytes)
        pool = trace_pool(tracer_children, module.trace, factory)
    _worker = (replace(options, workers=max(1, tracer_children)), layout_builder, module, evaluator, pool)
    return pool


def _stage_file_view(root, private):
    """Hard-link inputs. Plane kernels replace or rename files without writing through a link."""
    original = {}
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name in ('rays', 'history') or directory.name.startswith('.plane-work-'):
            continue
        (private/directory.name).mkdir()
        for source in directory.rglob('*'):
            if source.is_file():
                relative = source.relative_to(root)
                target = private/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.hardlink_to(source)
                # Hold a link to the inode even if the worker removes its path.
                original[relative] = source.stat().st_ino
    return original


@dataclass(frozen=True)
class _PlaneResult:
    first_call: int
    calls: list
    extrema: tuple
    files: tuple
    removed: tuple
    log_lines: tuple


def stage_plane(root, private, name, plane, snapshot, quota, extrema, worker):
    from .driver import _bind_stages, _context

    options, layout_builder, module, evaluator, pool = worker
    original = _stage_file_view(root, private)
    messages = []
    evaluator.beta_min, evaluator.beta_max, evaluator.max_u = extrema
    rays = _PlaneRays(private, root/'rays', options, module.trace, snapshot, quota, pool)
    stages = _bind_stages(_context(private, evaluator, options, layout_builder, module, rays, (), messages.append))
    try:
        getattr(stages, name)(plane)
    except _PlaneQuotaExceeded:
        return None
    finally:
        rays._file.cache_clear()
    files = {path.relative_to(private): path.stat().st_ino for path in private.rglob('*')
             if path.is_file() and path.parent != private/'rays'}
    changed = tuple(sorted(name for name, inode in files.items() if inode != original.get(name)))
    removed = tuple(sorted(original.keys() - files.keys()))
    return _PlaneResult(rays.first_call, rays.book()['calls'][rays.first_call:],
                        _extrema(evaluator), changed, removed, tuple(messages))


def run_plane(root, private, name, plane, snapshot_path, quota, extrema):
    snapshot = json.loads(snapshot_path.read_text())
    return stage_plane(root, private, name, plane, snapshot, quota, extrema, _worker)


def _publish_plane(rays, evaluator, log, root, private, result):
    """Publish files and assign each ray batch its serial call ordinal."""
    if result.extrema[0] <= result.extrema[1]:
        merge_extrema(evaluator, result.extrema)
    for old_index, call in enumerate(result.calls, result.first_call):
        rays.check_call_budget(call['rays'])
        new_index = len(rays.book()['calls'])
        batches = []
        for batch in call['batches']:
            prefix, tail = batch['file'].split('_', 1)
            assert int(prefix) == old_index
            filename = f'{new_index:08d}_{tail}'
            (private/'rays'/batch['file']).replace(rays.root/filename)
            batches.append(dict(batch, file=filename))
        rays.append_call(dict(call, batches=batches))

    for relative in result.files:
        source, target = private/relative, root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
    for relative in result.removed:
        (root/relative).unlink()
    for line in result.log_lines:
        log(line)


def log_plane_completion(log, book, phase, plane):
    ledger = book()
    log(f'{phase} plane {plane} complete; ledger {ledger["new_rays"]}/{ledger["cap"]}')


def run_plane_phase(pool, root, rays, evaluator, log, name, planes, serial, work=run_plane):
    """Collect private results before publishing, with no other plane running during commit."""
    planes = tuple(planes)
    if not planes:
        return
    with TemporaryDirectory(prefix='.plane-work-', dir=root) as workdir:
        private_root = Path(workdir)
        snapshot = rays.book()
        remaining = snapshot['cap'] - snapshot['new_rays']
        quota = remaining // len(planes)
        snapshot_path = private_root/'phase_ledger.json'
        dump(snapshot_path, snapshot)
        extrema = _extrema(evaluator)
        tasks = []
        for plane in planes:
            private = private_root/str(plane)
            private.mkdir()
            tasks.append((root, private, name, plane, snapshot_path, quota, extrema))
        results = list(pool.map(work, *zip(*tasks, strict=True)))
        rays.root.mkdir(exist_ok=True)
        for index, (plane, result) in enumerate(zip(planes, results, strict=True)):
            if result is None or any(key in rays.entries for call in result.calls
                                     for batch in call['batches'] for key in batch['keys']):
                # A quota (or an overlapping key) can prevent speculation. The
                # serial path decides the actual cap with the complete prefix.
                for plane in planes[index:]:
                    serial(plane)
                    log_plane_completion(log, rays.book, name, plane)
                break
            _publish_plane(rays, evaluator, log, root, private_root/str(plane), result)
            log_plane_completion(log, rays.book, name, plane)


def plane_pool(workers, planes, metric_cache, field_table, options, layout_builder, module, rss_limit_bytes):
    lanes = min(workers, planes)
    children = workers // lanes - 1
    return PlanePool(lanes, initialize_plane, (metric_cache, field_table, options, layout_builder,
                                              module.__name__, children, rss_limit_bytes))
