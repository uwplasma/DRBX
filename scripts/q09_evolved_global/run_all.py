"""Sequential, fail-fast subprocess controller; repeat unchanged to resume."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.q09_evolved_global import campaign as c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--science-run', type=Path, required=True)
    p.add_argument('--canonical-root', type=Path, required=True)
    p.add_argument('--host-gib', type=float, required=True)
    a = p.parse_args(); run = a.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    identity = c.check_source()
    binding = dict(identity=identity, **{k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(a).items()})
    with (run/'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = run/'controller_binding.json'
        if path.exists() and c.pilot.read(path) != binding: raise ValueError('controller RUN/source/inputs changed')
        c.pilot.write(path, binding)
        (run/'logs').mkdir(exist_ok=True); (run/'provenance').mkdir(exist_ok=True)
        stages = [('tests', []), ('verify', ['--baseline-run', str(a.baseline_run.resolve()),
            '--science-run', str(a.science_run.resolve()), '--canonical-root', str(a.canonical_root.resolve()),
            '--host-gib', str(a.host_gib)])]
        # Each process exits before the next grid is loaded, releasing dense banks/device memory.
        for n in c.GRIDS: stages += [('preflight', ['--n', str(n)]), ('run', ['--n', str(n)])]
        stages += [('validate', [])]
        for stage, extra in stages:
            tag = stage+('_N'+extra[-1] if stage in ('preflight','run') else '')
            argv = [sys.executable, str(HERE/'campaign.py'), stage, '--run', str(run), *extra]
            start = time.time(); attempt = time.time_ns()
            with (run/f'logs/{tag}_{attempt}.log').open('w') as log:
                result = subprocess.run(argv, cwd=c.REPO, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
            record = dict(stage=tag, command=argv, started_unix=start, seconds=time.time()-start,
                exit_code=result.returncode, identity=identity, log=str(log.name))
            with (run/'provenance/stages.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            if result.returncode: raise SystemExit(result.returncode)
        print('Complete: '+str(run/'completion.json'), flush=True)


if __name__ == '__main__': main()
