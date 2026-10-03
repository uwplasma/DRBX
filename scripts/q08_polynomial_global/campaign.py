"""Resume Q08 GPU qualification in a new namespace with read-only CPU inputs."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
BASE_ID = 'ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722'
NS = (32, 48, 64)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def check_source():
    manifest = read(HERE / 'manifest.json')
    for rel, expected in manifest['files'].items():
        if sha(HERE / rel) != expected:
            raise ValueError('frozen candidate source changed: ' + rel)
    if manifest['baseline_identity'] != BASE_ID:
        raise ValueError('wrong baseline identity')
    return sha(HERE / 'manifest.json'), manifest


def bind_paths(run, baseline_run, baseline_source, identity):
    run, baseline_run, baseline_source = map(lambda p: Path(p).resolve(),
                                           (run, baseline_run, baseline_source))
    if run == baseline_run or run.is_relative_to(baseline_run) or baseline_run.is_relative_to(run):
        raise ValueError('use a separate output folder; old RUN is immutable')
    binding = dict(identity=identity, baseline_identity=BASE_ID,
                   baseline_run=str(baseline_run), baseline_source=str(baseline_source))
    if (run / 'binding.json').exists() and read(run / 'binding.json') != binding:
        raise ValueError('output folder source/input identity mismatch')
    run.mkdir(parents=True, exist_ok=True)
    write(run / 'binding.json', binding)
    return run, baseline_run, baseline_source


def load_baseline(baseline_source, run, *, gpu):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(baseline_source.parents[1]))
    from scripts.q08_extraction_global import bootstrap, campaign as baseline
    if baseline.HERE.resolve() != baseline_source:
        raise ValueError('wrong frozen baseline import')
    bootstrap.configure(gpu=gpu, output=run)
    if baseline.digest(baseline.design()) != BASE_ID:
        raise ValueError('baseline source/input identity changed')
    import drbx.native
    drbx.native.__path__.insert(0, str(HERE / 'overlay'))
    for name in ('q_plan', 'q_sharding', 'q_parallel_material',
                 'q_parallel_characteristic', 'q_characteristic_polynomial'):
        module = importlib.import_module('drbx.native.' + name)
        if Path(module.__file__).resolve() != HERE / 'overlay' / (name + '.py'):
            raise ValueError('wrong candidate overlay import: ' + name)
    return baseline


def verify(run, baseline_run, baseline, identity, manifest):
    from cpu_validation import validate_cpu_readonly
    old = baseline.require(baseline_run, 'verification', BASE_ID)
    baseline.require(baseline_run, 'preflight', BASE_ID)
    baseline.check_inputs(baseline_run, Path(old['canonical_root']))
    grids = {}
    for n in NS:
        grids[str(n)] = validate_cpu_readonly(baseline_run, n, BASE_ID)
        for prefix in ('cpu', 'data'):
            path = baseline_run / f'{prefix}_N{n}.json'
            shutil.copy2(path, run / 'provenance' / path.name)
    for rel in ('design.json', 'input_manifest.json'):
        shutil.copy2(baseline.HERE / rel, run / 'provenance' / ('baseline_' + rel))
    for rel in (*manifest['files'], 'manifest.json'):
        target = run / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / rel, target)
    result = dict(passed=True, identity=identity, baseline_identity=BASE_ID,
                  baseline_run=str(baseline_run), cpu=grids,
                  canonical_root=old['canonical_root'], canonical=old['canonical'],
                  new_tracing=False, new_preparation=False, production_promoted=False,
                  records={str(p.relative_to(run)): sha(p)
                           for folder in ('provenance', 'source_snapshot')
                           for p in (run / folder).rglob('*') if p.is_file()})
    write(run / 'verification.json', result)
    return result


def require_verified(run, baseline_run, identity):
    result = read(run / 'verification.json')
    if (result.get('passed') is not True or result.get('identity') != identity or
            result.get('baseline_run') != str(baseline_run)):
        raise ValueError('verification identity/gate')
    for rel, expected in result['records'].items():
        if sha(run / rel) != expected:
            raise ValueError('verification receipt changed: ' + rel)
    return result


def collect(run, baseline_run, identity):
    from gpu_stage import validate_resolution
    verification = require_verified(run, baseline_run, identity)
    grids = [validate_resolution(run, n, identity, baseline_run=baseline_run,
                                 baseline_identity=BASE_ID) for n in NS]
    return dict(passed=True, identity=identity, baseline_identity=BASE_ID,
                cpu=verification['cpu'], gpu=grids,
                scientific_qualification=False, production_promoted=False)


def analyze(run, baseline_run, identity):
    result = collect(run, baseline_run, identity)
    write(run / 'analysis.json', result)
    lines = ['# Q08 polynomial full-grid implementation replay', '',
             f'Identity: `{identity}`', '',
             'All 22 states, four D/N patterns, both diffusion spans and one/four A100s.',
             'No new MMS convergence or production promotion is claimed.', '',
             '| N | GPU records | largest replay budget fraction | host peak GiB | BC producer calls |',
             '|---|---:|---:|---:|---:|']
    for grid in result['gpu']:
        records = [read(run / name) for name in grid['records']]
        lines.append(f"| {grid['n']} | {len(records)} | "
                     f"{max(r['metrics']['max_scaled_error'] for r in records):.6g} | "
                     f"{grid['host_peak_rss_gib']:.4f} | {grid['boundary_cache']['producer_calls']} |")
    lines += ['', 'Detailed setup, reference, compilation, transfer, warm timing and memory',
              'receipts remain in gpu/N*/records. Existing CPU banks are external immutable',
              'inputs with original identities; the old RUN was not modified.']
    (run / 'report.md').write_text('\n'.join(lines) + '\n')
    return result


def completion(run, baseline_run, identity):
    result = collect(run, baseline_run, identity)
    if read(run / 'analysis.json') != result:
        raise ValueError('analysis differs from independent completion reduction')
    files = {str(p.relative_to(run)): sha(p)
             for folder in ('gpu', 'source_snapshot', 'provenance')
             for p in (run / folder).rglob('*') if p.is_file() and p.name != 'stage.lock'}
    files.update({name: sha(run / name) for name in ('verification.json', 'binding.json',
                  'analysis.json', 'report.md', *(f'gpu_N{n}.json' for n in NS))})
    result = dict(passed=True, identity=identity, baseline_identity=BASE_ID, files=files,
                  scientific_qualification=False, production_promoted=False)
    write(run / 'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('verify', 'gpu', 'analyze', 'validate-completion'))
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--n', type=int, choices=NS)
    parser.add_argument('--host-gib', type=float)
    args = parser.parse_args()
    if args.stage == 'gpu' and (args.n is None or args.host_gib is None):
        parser.error('gpu requires --n and --host-gib')
    identity, manifest = check_source()
    run, baseline_run, baseline_source = bind_paths(args.run, args.baseline_run,
                                                   args.baseline_source, identity)
    for folder in ('provenance', 'tmp', 'logs'):
        (run / folder).mkdir(exist_ok=True)
    with (run / 'stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(run).free < 3 * 2**30:
            raise RuntimeError('free disk below 3 GiB')
        baseline = load_baseline(baseline_source, run, gpu=args.stage == 'gpu')
        if args.stage == 'verify':
            result = verify(run, baseline_run, baseline, identity, manifest)
        else:
            require_verified(run, baseline_run, identity)
            if args.stage == 'gpu':
                from cpu_validation import validate_cpu_readonly
                validate_cpu_readonly(baseline_run, args.n, BASE_ID)
                from gpu_stage import run_resolution
                result = run_resolution(run, args.n, identity, args.host_gib,
                                        baseline_run=baseline_run, baseline_identity=BASE_ID)
                from gpu_stage import validate_resolution
                validate_resolution(run, args.n, identity, baseline_run=baseline_run,
                                    baseline_identity=BASE_ID)
            elif args.stage == 'analyze':
                result = analyze(run, baseline_run, identity)
            else:
                result = completion(run, baseline_run, identity)
        if check_source()[0] != identity or baseline.digest(baseline.design()) != BASE_ID:
            raise ValueError('source changed during stage')
        print(json.dumps(dict(stage=args.stage, passed=result['passed'], identity=identity)), flush=True)


if __name__ == '__main__':
    main()
