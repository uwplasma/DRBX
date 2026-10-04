"""Pinned scientific campaign controller. Existing Q08 banks stay immutable."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# File execution and multiprocessing spawn must resolve this campaign by its
# package name, independently of the older controllers loaded below.
sys.path.insert(0, str(REPO))
from scripts import q08_rhs_mms_global as _package
if Path(_package.__file__).resolve() != HERE / '__init__.py':
    raise ValueError('wrong scientific campaign package import')
POLY = HERE.with_name('q08_polynomial_global')
BASE_ID = 'ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722'
POLY_ID = '1a953c60d7352b2a74e6a737d7c2be7dcca9ee194808db3154e5f840d4a9b087'
NS = (32, 48, 64)
# A remote wrapper appends its exit/time after final validation. Scientific
# provenance remains content-hashed; this one operational stream cannot be.
MUTABLE_PROVENANCE = ('provenance/complete_runtime.txt',)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(p)


def check_source():
    m = read(HERE/'manifest.json')
    if m['baseline_identity'] != BASE_ID or m['implementation_identity'] != POLY_ID:
        raise ValueError('wrong scientific input contract')
    for rel, h in m['files'].items():
        if sha(HERE/rel) != h:
            raise ValueError('frozen science source changed: '+rel)
    if sha(POLY/'manifest.json') != POLY_ID:
        raise ValueError('implementation manifest changed')
    for rel, h in read(POLY/'manifest.json')['files'].items():
        if sha(POLY/rel) != h:
            raise ValueError('implementation source changed: '+rel)
    return sha(HERE/'manifest.json'), m


def load(source, run, gpu=False):
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(POLY), str(REPO)]
    from scripts.q08_polynomial_global import campaign as poly
    baseline = poly.load_baseline(Path(source).resolve(), Path(run), gpu=gpu)
    return baseline, poly


def bind(args, identity):
    run, old, source, implementation = [Path(getattr(args, k)).resolve() for k in
                                      ('run', 'baseline_run', 'baseline_source', 'implementation_run')]
    for p in (old, implementation):
        if run == p or run.is_relative_to(p) or p.is_relative_to(run):
            raise ValueError('new separate scientific output folder required')
    value = dict(identity=identity, baseline_identity=BASE_ID, implementation_identity=POLY_ID,
                 baseline_run=str(old), baseline_source=str(source), implementation_run=str(implementation))
    if (run/'binding.json').exists() and read(run/'binding.json') != value:
        raise ValueError('run binding changed')
    run.mkdir(parents=True, exist_ok=True)
    write(run/'binding.json', value)
    for name in ('logs', 'provenance', 'tmp', 'cache'):
        (run/name).mkdir(exist_ok=True)
    return run, old, source, implementation


def verify(run, old, implementation, baseline, poly, identity):
    # Reuse the complete content-based baseline input validator without any
    # row preparation, CPU action replay, tracing or old-RUN mutations.
    result = poly.verify(run, old, baseline, identity, read(POLY/'manifest.json'))
    for rel in MUTABLE_PROVENANCE:
        result['records'].pop(rel, None)
    result['mutable_operational_logs'] = list(MUTABLE_PROVENANCE)
    completed = read(implementation/'completion.json')
    if not completed.get('passed') or completed.get('identity') != POLY_ID:
        raise ValueError('completed GPU implementation gate missing')
    for rel, h in completed['files'].items():
        p = (implementation/rel).resolve()
        if not p.is_relative_to(implementation) or sha(p) != h:
            raise ValueError('implementation completion content changed: '+rel)
    shutil.copy2(implementation/'completion.json', run/'provenance/implementation_completion.json')
    for name in (*read(HERE/'manifest.json')['files'], 'manifest.json'):
        dest = run/'science_source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE/name, dest)
    result['implementation_completion'] = sha(implementation/'completion.json')
    result['records'].update({str(p.relative_to(run)): sha(p) for p in (run/'science_source').rglob('*') if p.is_file()})
    result['records']['provenance/implementation_completion.json'] = sha(run/'provenance/implementation_completion.json')
    write(run/'verification.json', result)
    return result


def require(run, name, identity):
    run = Path(run).resolve()
    r = read(run/(name+'.json'))
    if r.get('identity') != identity or r.get('passed') is not True:
        raise ValueError('stage prerequisite failed: '+name)
    for rel, h in r.get('files', {}).items():
        p = (run/rel).resolve()
        if not p.is_relative_to(run) or sha(p) != h:
            raise ValueError('stage content changed: '+rel)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('verify', 'preflight', 'preflight-gpu', 'pilot',
                                    'references', 'gpu', 'analyze', 'validate-completion'))
    for k in ('run', 'baseline-run', 'baseline-source', 'implementation-run'):
        p.add_argument('--'+k, required=True, type=Path)
    p.add_argument('--n', type=int, choices=NS)
    p.add_argument('--workers', type=int)
    p.add_argument('--host-gib', type=float)
    a = p.parse_args()
    if a.stage in ('pilot', 'references') and (a.workers is None or a.host_gib is None):
        p.error('reference stages require explicit --workers and --host-gib')
    if a.stage in ('references', 'gpu') and a.n is None:
        p.error('stage requires --n')
    if a.stage == 'gpu' and a.host_gib is None:
        p.error('gpu requires --host-gib')
    identity, _ = check_source()
    run, old, source, implementation = bind(a, identity)
    with (run/'stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(run).free < 12*2**30:
            raise RuntimeError('at least 12 GiB free disk required for references and merge scratch')
        baseline, poly = load(source, run, gpu=a.stage in ('gpu', 'preflight-gpu'))
        if a.stage == 'verify':
            result = verify(run, old, implementation, baseline, poly, identity)
        else:
            verified = poly.require_verified(run, old, identity)
            if sha(implementation/'completion.json') != verified['implementation_completion']:
                raise ValueError('implementation completion identity changed')
            if a.stage in ('preflight', 'preflight-gpu'):
                name = 'preflight_gpu' if a.stage == 'preflight-gpu' else 'preflight'
                if (run/(name+'.json')).exists():
                    result = require(run, name, identity)
                    if result.get('test_only') or result.get('gpu') != (a.stage == 'preflight-gpu'):
                        raise ValueError('preflight backend mismatch')
                else:
                    from scripts.q08_rhs_mms_global.preflight import bounded
                    result = bounded(run, old, identity, verified['canonical_root'], gpu=a.stage == 'preflight-gpu')
            elif a.stage in ('pilot', 'references'):
                require(run, 'preflight', identity); require(run, 'preflight_gpu', identity)
                baseline.check_inputs(old, Path(verified['canonical_root']))
                from scripts.q08_rhs_mms_global.references import run_references
                result = run_references(run, old, source, identity, verified['canonical_root'],
                                        a.n or 32, a.workers, a.host_gib, pilot=a.stage == 'pilot')
            elif a.stage == 'gpu':
                require(run, 'preflight_gpu', identity)
                data = read(old/f'data_N{a.n}.json')
                if data.get('campaign_identity') != BASE_ID or not data.get('passed'):
                    raise ValueError('owner input stage identity')
                for name, h in data['files'].items():
                    if sha(old/'data'/f'N{a.n}'/name) != h:
                        raise ValueError('owner input content changed: '+name)
                from scripts.q08_rhs_mms_global.gpu import run_grid
                result = run_grid(run, old, identity, a.n, a.host_gib)
            else:
                from scripts.q08_rhs_mms_global.analyze import analyze, completion
                result = (analyze if a.stage == 'analyze' else completion)(run, identity)
        if check_source()[0] != identity or baseline.digest(baseline.design()) != BASE_ID:
            raise ValueError('source changed during stage')
        print(json.dumps(dict(stage=a.stage, passed=result['passed'], identity=identity)), flush=True)


if __name__ == '__main__':
    main()
