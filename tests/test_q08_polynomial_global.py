"""Campaign reuse, bounded boundary storage and candidate-only GPU mechanics."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / 'scripts/q08_polynomial_global'
sys.path[:0] = [str(HERE), str(REPO)]
from scripts.q08_polynomial_global import campaign, gpu_stage


def fixture_module():
    spec = importlib.util.spec_from_file_location('old_gpu_fixture', REPO / 'tests/test_q08_extraction_gpu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_overlay_byte_identical():
    _, manifest = campaign.check_source()
    for rel in manifest['files']:
        if rel.startswith('overlay/'):
            assert (HERE / rel).read_bytes() == (REPO / 'scripts/q08_polynomial_profile' / rel).read_bytes()


@pytest.mark.parametrize('target', ['cusolver_geev_ffi', 'lapack_dgetrf_ffi', 'xla_python_cpu_callback'])
def test_compiler_rejects_cpu_or_general_eig(target):
    with pytest.raises(ValueError):
        gpu_stage.compiler_guard(f'custom_call_target="{target}"')


def test_compiler_accepts_device_lu():
    assert gpu_stage.compiler_guard('custom_call_target="cusolver_getrf_ffi"') == ['cusolver_getrf_ffi']


def test_completion_refuses_old_or_test_identity(tmp_path):
    with pytest.raises(ValueError, match='configuration'):
        campaign.write(tmp_path / 'gpu_N32.json', dict(passed=True))
        gpu_stage.validate_resolution(tmp_path, 32, 'candidate', baseline_run=tmp_path / 'old', baseline_identity='old')


def test_real_gpu_subset_refused(tmp_path):
    old = fixture_module()
    with pytest.raises(ValueError, match='cannot restrict'):
        gpu_stage.run_resolution(tmp_path / 'new', 32, 'new', 8,
            baseline_run=tmp_path / 'old', baseline_identity='old', cases=(1,), common_api=old.api())


def test_original_cpu_receipt_reduced_without_writes(tmp_path):
    from scripts.q08_polynomial_global.cpu_validation import validate_cpu_readonly
    raw = [0, 2, 1]
    folder = tmp_path / 'cpu/N8/chunk_000000'
    folder.mkdir(parents=True)
    for name in ('bank.npz', 'geometry.npz'):
        (folder / name).write_bytes(b'content-checked-placeholder')
    data = tmp_path / 'data/N8'
    data.mkdir(parents=True)
    np.save(data / 'raw_to_owner.npy', np.array([0, 1, 0]))
    campaign.write(tmp_path / 'data_N8.json', dict(passed=True, campaign_identity='old',
        files={'raw_to_owner.npy':campaign.sha(data / 'raw_to_owner.npy')}))
    campaign.write(tmp_path / 'inputs/plan.json', {'8':dict(chunks=[[0, 1]], n_owner=2, n_raw=3)})
    from drbx.native.q_parallel_rhs import SixFieldAction
    from drbx.native.q_parallel_current_phi import CurrentPhiAction
    from scripts.q08_extraction_global.common import check_outputs
    current = CurrentPhiAction(*(np.zeros((22,3)) for _ in range(9)), np.ones((22,3),bool))
    action = SixFieldAction(*(np.zeros((22,2,6)) for _ in range(4)), current,
                            np.zeros((22,3)), np.ones((22,3),bool), np.ones((22,3),bool))
    leaves = check_outputs(action, action)['leaves']
    stats = dict(passed=True, campaign_identity='old', owners=[0,1], raw=raw,
        all_arrays_bitwise=True, max_scaled=0.,
        files={name:campaign.sha(folder/name) for name in ('bank.npz','geometry.npz')},
        components=[dict(span=span,kind=k,leaves=leaves) for span in (1/16,1/32) for k in range(4)])
    campaign.write(folder/'stats.json', stats)
    receipt = dict(passed=True,campaign_identity='old',n=8,owners=2,raw=3,chunks=1,
                   max_scaled=0., bytes=sum((folder/name).stat().st_size for name in stats['files']),
                   chunk_receipts={'0':campaign.sha(folder/'stats.json')})
    campaign.write(tmp_path/'cpu_N8.json',receipt)
    before = {str(p.relative_to(tmp_path)):campaign.sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert validate_cpu_readonly(tmp_path,8,'old') == receipt
    assert before == {str(p.relative_to(tmp_path)):campaign.sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    (folder/'bank.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='corrupt'):
        validate_cpu_readonly(tmp_path,8,'old')


def test_complete_four_host_device_pipeline_and_resume(tmp_path):
    old = fixture_module()
    source = tmp_path / 'old'
    old.fixture_chunks(source)
    env = dict(os.environ, JAX_ENABLE_X64='true', JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', JAX_COMPILATION_CACHE_DIR='/tmp/q08-global-tests-cache')
    env['XLA_FLAGS'] = env.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=4'
    child = subprocess.run([sys.executable, str(Path(__file__)), '--pipeline', str(tmp_path)],
                            env=env, capture_output=True, text=True, timeout=300)
    assert child.returncode == 0, child.stderr[-6000:] + child.stdout[-2000:]
    result = json.loads(child.stdout.strip().splitlines()[-1])
    assert result['passed'] and result['records'] == 16 and result['test_only']
    assert result['producer_calls'] == 3  # one full bank + two literal chunks
    assert result['resume_producer_calls'] == 0


if __name__ == '__main__' and '--pipeline' in sys.argv:
    # This test covers orchestration/BC/cache/resume, not spectral algebra.
    # Use a lightweight identical split on both sides; the actual polynomial
    # algebra and one/four-shard action are covered by the separate 24-test
    # characteristic suite and the returned A100 benchmark.
    import jax.numpy as jnp
    from drbx.native import q_parallel_characteristic as characteristic
    from drbx.native import q_characteristic_polynomial as polynomial
    calls = {'eig':0,'polynomial':0}
    def cheap_split(matrix, normal=1.):
        action = matrix*jnp.asarray(normal)[...,None,None]/2
        speed = jnp.ones(matrix.shape[:-2],dtype=matrix.dtype)
        return action,action,speed,speed,jnp.ones(matrix.shape[:-2],bool)
    def eig_split(matrix, normal=1.):
        calls['eig'] += 1
        return cheap_split(matrix,normal)
    def polynomial_split(state,tau,mu,normal=1.):
        calls['polynomial'] += 1
        return cheap_split(characteristic.parallel_matrix_from_state(state,tau,mu),normal)
    characteristic.parallel_characteristic_split = eig_split
    polynomial.polynomial_characteristic_split = polynomial_split
    root = Path(sys.argv[-1]); old = fixture_module()
    source, run = root / 'old', root / 'new'
    before = {str(p.relative_to(source)): campaign.sha(p) for p in source.rglob('*') if p.is_file()}
    kwargs = dict(baseline_run=source, baseline_identity=old.IDENTITY,
                  test_cpu=True, cases=(1,), common_api=old.api(), warm_repeats=1)
    result = gpu_stage.run_resolution(run, 32, 'polynomial-mechanics', 8, **kwargs)
    assert calls['eig'] > 0 and calls['polynomial'] > 0
    assert before == {str(p.relative_to(source)): campaign.sha(p) for p in source.rglob('*') if p.is_file()}
    records = {name: campaign.sha(run / name) for name in result['records']}
    again = gpu_stage.run_resolution(run, 32, 'polynomial-mechanics', 8, **kwargs)
    assert records == {name: campaign.sha(run / name) for name in again['records']}
    print(json.dumps(dict(passed=True, test_only=result['test_only'], records=len(records),
                          producer_calls=result['boundary_cache']['producer_calls'],
                          resume_producer_calls=again['boundary_cache']['producer_calls'])))
