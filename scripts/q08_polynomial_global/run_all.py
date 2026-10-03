"""One resumable controller; fresh processes prevent backend/cache leakage."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'baseline-run', 'baseline-source'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--host-gib', type=float, required=True)
    parser.add_argument('--through', choices=('verify', 'N32', 'N48', 'complete'), default='complete')
    args = parser.parse_args()
    from campaign import bind_paths, check_source, write
    identity, _ = check_source()
    run, old, source = bind_paths(args.run, args.baseline_run, args.baseline_source, identity)
    for name in ('logs', 'provenance', 'tmp'):
        (run / name).mkdir(exist_ok=True)
    with (run / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common = ['--run', str(run), '--baseline-run', str(old), '--baseline-source', str(source)]
        command = [sys.executable, str(HERE / 'campaign.py')]
        stages = [('tests', [sys.executable, str(HERE / 'verification/test_campaign.py')]),
                  ('verify', command + ['verify', *common])]
        for n in (32, 48, 64):
            if args.through == 'verify':
                break
            stages.append((f'gpu_N{n}', command + ['gpu', *common, '--n', str(n),
                                                  '--host-gib', str(args.host_gib)]))
            if args.through == f'N{n}':
                break
        if args.through == 'complete':
            stages += [(stage, command + [stage, *common])
                       for stage in ('analyze', 'validate-completion')]
        for stage, cmd in stages:
            tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + f'_{time.time_ns()}'
            log = run / 'logs' / f'{stage}_{tag}.log'
            env = dict(os.environ, TMPDIR=str(run / 'tmp'), JAX_ENABLE_X64='true', PYTHONDONTWRITEBYTECODE='1',
                       XLA_PYTHON_CLIENT_PREALLOCATE='false')
            if stage.startswith('gpu_'):
                env['JAX_PLATFORMS'] = 'cuda,cpu'
                # Preserve scheduler CUDA visibility from this parent process.
            else:
                env.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
            start = time.monotonic()
            print(json.dumps(dict(stage=stage, command=cmd, log=str(log))), flush=True)
            with log.open('w') as stream:
                child = subprocess.run(cmd, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
            receipt = dict(stage=stage, command=cmd, log=str(log.relative_to(run)),
                           exit_code=child.returncode, elapsed_seconds=time.monotonic() - start,
                           identity=identity)
            write(run / 'provenance' / f'exit_{stage}_{tag}.json', receipt)
            print(json.dumps(receipt), flush=True)
            if child.returncode:
                raise SystemExit(child.returncode)
        print(json.dumps(dict(passed=True, through=args.through, run=str(run))), flush=True)


if __name__ == '__main__':
    main()
