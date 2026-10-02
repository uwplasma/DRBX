"""GPU orchestration mechanics; CPU emulation is explicitly test-only."""
from pathlib import Path
from types import SimpleNamespace
import json,os,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from scripts.q08_extraction_global import gpu
from scripts.q08_extraction_global import common
from drbx.stencils.q_bank import QBank
from drbx.stencils.q_artifact import save_q_bank
from drbx.stencils.q_parallel import sha256_array
from drbx.native.q_parallel import QBoundaryData

IDENTITY='gpu-mechanics-test-no-scientific-qualification'


def fixture_chunks(run):
    n=32;total=n**3;topology=np.arange(total,dtype=np.int64)
    coords=np.stack(np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)*2*np.pi/n,
                               (np.arange(n)+.5)*2*np.pi/n,indexing='ij'),axis=-1).reshape(-1,3)
    for index,(start,stop) in enumerate(((0,total//2),(total//2,total))):
        raw=np.arange(start,stop,dtype=np.int32);nr=len(raw);wall=np.flatnonzero(raw//n**2>=n-2).astype(np.int32);nw=len(wall)
        # Different legal dense donor widths exercise exact zero-padding/views.
        nd=index+1
        donor=np.zeros((nr,nd),np.int32);donor[:,0]=raw
        if nd==2:donor[:,1]=(raw//n)*n+(raw+1)%n
        mask=np.ones((nr,nd),bool)
        rows=np.zeros((nr,5,nd));rows[:,:,0]=1
        dw=np.zeros((2,nr,nd));dw[:,:,0]=np.array([.007,.011])[:,None]
        normal=rows[wall].copy();normal[:,:,0]=1.
        query=np.array([[1.,-0.,0.],[1.,0.,0.],[1.,0.,.125]])
        a=dict(owners=raw.copy(),raw=raw,raw_to_owner=np.arange(nr,dtype=np.int32),raw_weight=np.ones(nr),
            owner_raw=np.arange(nr,dtype=np.int32)[:,None],owner_weight=np.ones((nr,1)),donor=donor,
            mask=mask,row_count=np.full(nr,nd,np.int32),wall_index=wall,row_value_D=rows,
            row_value_N_wall=normal,boundary_value_D_trace=np.zeros((nw,5,35)),
            boundary_value_N_normal=np.full((nw,5,35),.001),diffusion_D=dw,
            diffusion_N_wall=dw[:,wall].copy()*1.2,boundary_D_node=np.full((2,nw,35),.0001),
            boundary_D_tangent=np.zeros((2,nw,3,2)),boundary_N_normal=np.full((2,nw,35),.0002),
            query_table=query,wall_node_query=np.tile(np.arange(35)%3,(nw,1)).astype(np.int32),
            wall_slot_query=np.tile(np.arange(5)%3,(nw,1)).astype(np.int32),
            boundary_wall_normal=np.tile([1.,0,0],(nw,35,1)),magnetic_b=np.tile([0.,0.,1.],(nr,5,1)),
            magnetic_L=np.tile([-1.,1.,.01],(2,nr,1)))
        fixed=dict(campaign_identity=IDENTITY,source_identity=f'chunk{index}',geometry_identity='algebra-only',
            topology_hash=sha256_array(topology)+':'+sha256_array(np.ones(total)),
            trace_hash=sha256_array(raw)+':'+sha256_array(np.zeros((nr,4,3))),support='synthetic exact-merge mechanics',
            n=n,n_owner=total,trace_steps=64,trace_method='RK4')
        meta=dict(fixed,schema='drbx.q-paired-bank.v1',spans=(1/16,1/32),
            slot_names=('outer_minus','inner_minus','center','inner_plus','outer_plus'),has_gradients=False,
            span_metadata=[dict(fixed,span=span) for span in (1/16,1/32)])
        points=np.repeat(coords[raw,None],5,axis=1)
        diag=dict(choice=np.full(nr,-1,np.int8),indicator=np.zeros((nr,3,2)),noise=np.zeros((nr,3)),slot_points=points)
        bank=QBank(meta,a,diag).validate()
        folder=run/'cpu/N32'/f'chunk_{index:06d}';folder.mkdir(parents=True)
        save_q_bank(bank,folder/'bank.npz')
        np.savez_compressed(folder/'geometry.npz',magnetic_L=np.tile([-1.,1.,.01],(nr,1)),
            b_eta=np.ones(nr),eta_step=np.array(.01),bmag=np.ones(nr))
        receipt=dict(passed=True,n=n,index=index,campaign_identity=IDENTITY,owners=raw.tolist(),
            raw=raw.tolist(),raw_hash=sha256_array(raw),files={name:gpu.sha256_file(folder/name) for name in ('bank.npz','geometry.npz')})
        gpu.write_json(folder/'stats.json',receipt)
    data=run/'data/N32';data.mkdir(parents=True)
    base=np.array([1.,1.1,.9,.13,.08,.2])
    state=np.broadcast_to(base[None,:,None],(22,6,total)).copy()
    state+=.001*np.sin(np.arange(total))[None,None,:]
    np.save(data/'state.npy',state);np.save(data/'phi.npy',.03*state[:,0]);np.save(data/'raw_to_owner.npy',topology)


def fake_boundaries(bank,case):
    nr=len(bank.raw);nw=bank.wall_index
    def bc(fields):
        arrays=[np.zeros((fields,nr,35)),np.zeros((fields,nr,3)),
                np.zeros((fields,nr,3,2)),np.zeros((fields,nr,35))]
        arrays[0][:,nw]=.003;arrays[1][:,nw]=.001;arrays[3][:,nw]=.002
        return QBoundaryData(*arrays)
    return bc(6),bc(6),QBoundaryData(*(a[0] for a in bc(1)))


def api():return SimpleNamespace(boundaries=fake_boundaries,KINDS=common.KINDS,COEFF=common.COEFF,
    TAU=common.TAU,MU=common.MU,check_outputs=common.check_outputs)


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    path=tmp_path_factory.mktemp('q08-gpu');fixture_chunks(path);return path


def test_exact_merge_preserves_every_literal_chunk_identity_and_signed_zero(run):
    merged,geometry,views,estimate=gpu.merge_chunks(run,32,IDENTITY,8)
    assert estimate['raw_count']==32**3 and len(merged.owners)==32**3
    assert merged.metadata['schema']==gpu.MERGED_SCHEMA
    assert not isinstance(merged,QBank)
    assert len(merged.query_table)==3 # Signed zero queries are distinct exact bits.
    assert merged.query_table[0].tobytes()!=merged.query_table[1].tobytes()
    for view in views:
        loaded=gpu.load_q_bank(view.path/'bank.npz')
        assert view.bank.identity==loaded.identity
        for k in loaded.arrays:
            assert gpu.sha256_array(view.bank.arrays[k])==gpu.sha256_array(loaded.arrays[k]),k
    np.testing.assert_array_equal(geometry['magnetic_L'],np.tile([-1.,1.,.01],(32**3,1)))


def test_memory_preflight_fails_before_dense_load(run,monkeypatch):
    monkeypatch.setattr(gpu,'load_q_bank',lambda *a,**kw:pytest.fail('dense bank load before resource gate'))
    with pytest.raises(MemoryError,match='host resource preflight'):
        gpu.merge_chunks(run,32,IDENTITY,.001)


def test_checked_chunk_rejects_stale_pass_receipts(run):
    with pytest.raises(ValueError,match='campaign/pass'):
        gpu.checked_chunks(run,32,'different-campaign')


def test_checkpoint_identity_and_pass_are_required(tmp_path):
    path=tmp_path/'record.json'
    assert not gpu._valid_record(path,'current')
    gpu.write_json(path,dict(identity='old',passed=True))
    assert not gpu._valid_record(path,'current')
    gpu.write_json(path,dict(identity='current',passed=False))
    assert not gpu._valid_record(path,'current')
    gpu.write_json(path,dict(identity='current',passed=True))
    assert gpu._valid_record(path,'current')


def test_gpu_memory_inventory_requires_observed_sufficient_available():
    inventory=dict(test_only_cpu_emulation=False,devices=[dict(id=0,local_hardware_id=0,memory_stats={})],
                   nvidia_smi=[dict(index=0,free_bytes=100)])
    with pytest.raises(MemoryError,match='GPU resource preflight'):
        gpu._device_guard(dict(estimated_gpu0_peak_upper_bytes=101),inventory)


def test_actual_gpu_matrix_cannot_be_restricted(tmp_path):
    with pytest.raises(ValueError,match='cannot restrict'):
        gpu.run_resolution(tmp_path,32,IDENTITY,8,cases=(0,),common_api=api())


def test_forced_four_cpu_pipeline_and_checked_resume(run):
    env=dict(os.environ,JAX_ENABLE_X64='true',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env['XLA_FLAGS']=env.get('XLA_FLAGS','')+' --xla_force_host_platform_device_count=4'
    completed=subprocess.run([sys.executable,str(Path(__file__)),'--test-pipeline',str(run)],
        env=env,capture_output=True,text=True,timeout=240)
    assert completed.returncode==0,completed.stderr[-5000:]
    result=json.loads(completed.stdout.strip().splitlines()[-1])
    assert result['passed'] and result['test_only'] and result['units']==16
    assert result['max_scaled']<=1


def test_completion_validator_rejects_test_only_receipt(tmp_path):
    gpu.write_json(tmp_path/'gpu_N32.json',dict(identity=IDENTITY,campaign_identity=IDENTITY,n=32,passed=True,test_only=True))
    with pytest.raises(ValueError,match='backend mismatch'):
        gpu.validate_resolution(tmp_path,32,IDENTITY)


def test_completion_validator_requires_full_matrix(tmp_path):
    gpu.write_json(tmp_path/'gpu_N32.json',dict(identity=IDENTITY,campaign_identity=IDENTITY,n=32,passed=True,test_only=False,matrix_units=16,records=[]))
    with pytest.raises(ValueError,match='352-unit'):
        gpu.validate_resolution(tmp_path,32,IDENTITY)


def test_metrics_receipts_require_all_leaves_shapes_and_nonnegative_errors(tmp_path):
    from drbx.native.q_parallel_rhs import SixFieldAction
    from drbx.native.q_parallel_current_phi import CurrentPhiAction
    current=CurrentPhiAction(*(np.zeros(3) for _ in range(9)),np.ones(3,bool))
    result=SixFieldAction(*(np.zeros((2,6)) for _ in range(4)),current,np.zeros(3),np.ones(3,bool),np.ones(3,bool))
    metrics=common.check_outputs(result,result)
    path=tmp_path/'unit.json'
    gpu.write_json(path,dict(identity=IDENTITY,passed=True,metrics=metrics))
    assert gpu._valid_record(path,IDENTITY,2,3)
    metrics['max_scaled_error']=-1
    gpu.write_json(path,dict(identity=IDENTITY,passed=True,metrics=metrics))
    assert not gpu._valid_record(path,IDENTITY,2,3)
    metrics['max_scaled_error']=0;metrics['leaves'].pop('raw_current.inputs_valid')
    gpu.write_json(path,dict(identity=IDENTITY,passed=True,metrics=metrics))
    assert not gpu._valid_record(path,IDENTITY,2,3)


if __name__=='__main__' and '--test-pipeline' in sys.argv:
    path=Path(sys.argv[-1]);result=gpu.run_resolution(path,32,IDENTITY,8,test_cpu=True,cases=(0,),common_api=api(),warm_repeats=1)
    records=[json.loads((path/name).read_text()) for name in result['records']]
    # Second pass exercises the same content-checked checkpoint path without
    # recomputing CPU references or overwriting completed timing records.
    before={name:gpu.sha256_file(path/name) for name in result['records']}
    again=gpu.run_resolution(path,32,IDENTITY,8,test_cpu=True,cases=(0,),common_api=api(),warm_repeats=1)
    assert before=={name:gpu.sha256_file(path/name) for name in again['records']}
    print(json.dumps(dict(passed=again['passed'],test_only=again['test_only'],units=len(records),
        max_scaled=max(r['metrics']['max_scaled_error'] for r in records))))
