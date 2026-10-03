"""Cold interpreter / spawn regression for the three frozen campaign loaders."""
import argparse
import concurrent.futures
import importlib
import json
import multiprocessing
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parents[1]


def inspect(source, run):
    # Like campaign.py as a file (and spawn's __mp_main__), this does not
    # install the MMS controller in sys.modules under the bare name campaign.
    namespace = runpy.run_path(str(HERE / 'campaign.py'), run_name='__mp_main__')
    assert 'campaign' not in sys.modules
    baseline, poly = namespace['load'](source, run)
    # Reproduce the path ordering that selected the older controller remotely.
    collision = importlib.import_module('campaign')
    assert Path(collision.__file__).resolve() == Path(poly.__file__).resolve()
    assert not hasattr(collision, 'load')
    modules = {name: importlib.import_module('scripts.q08_rhs_mms_global.' + name)
               for name in ('campaign', 'references', 'preflight', 'gpu', 'analyze', 'run_all')}
    for name, module in modules.items():
        assert Path(module.__file__).resolve() == HERE / (name + '.py'), name
    controller = modules['campaign']
    assert modules['references'].load is controller.load
    assert modules['gpu'].require is controller.require
    assert modules['analyze'].require is controller.require
    assert modules['preflight'].write is controller.write
    assert modules['run_all'].bind is controller.bind
    # Exercise the worker's checkpoint reader in the cold process as well.
    path = Path(run) / ('spawn-checkpoint' if multiprocessing.parent_process() else 'parent-checkpoint')
    path.write_bytes(b'import-regression')
    controller.write(path.with_suffix('.json'), dict(passed=True, identity='test',
        input_receipt='test-input', sha256=controller.sha(path)))
    assert modules['references'].valid(path, 'test', 'test-input')['passed']
    return {name: str(Path(module.__file__).resolve()) for name, module in modules.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-source', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    a = p.parse_args(); a.run.mkdir(parents=True, exist_ok=True)
    parent = inspect(a.baseline_source, a.run)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        child = pool.submit(inspect, a.baseline_source, a.run).result(timeout=90)
    assert child == parent
    print(json.dumps(dict(passed=True, cold_parent=True, spawned_worker=True, modules=parent)))


if __name__ == '__main__':
    main()
