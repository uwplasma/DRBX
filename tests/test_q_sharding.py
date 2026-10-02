"""Q owner localization mechanics and bounded portable actual-HSX replay.

Synthetic all-plane rows test collective mechanics only. Portable saved HSX
queries retain their original geometry identity; no trace is recomputed.
"""
from dataclasses import replace
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh,PartitionSpec as P

sys.path.insert(0,str(Path(__file__).parent))
from test_q_parallel_hsx_portable import patch
from test_q_parallel_rhs import assembled
from drbx.stencils.q_bank import QBank,build_q_bank
from drbx.stencils.q_plan import lower_q_plan
from drbx.stencils.q_parallel import sha256_array
from drbx.native.q_plan import apply_q_plan,reconstruct_q_state
from drbx.native.q_parallel import QBoundaryData
from drbx.native.owner_plane_layout import (owner_layout,plane_major_permutation,
                                          exchange_plane_halo,to_plane_major)
from drbx.native.q_sharding import (q_donor_reach,q_shard_layouts,localize_q_bank,
    localize_q_plan,local_owner_values,local_boundary_data,shard_q_plan,sharded_q_rhs)
from drbx.stencils import q_parallel_rhs as rhs_prep


def synthetic_bank():
    """All eta planes; no new magnetic/trace or scientific coverage claim."""
    n=32;nr=3*n;nd=5
    raw=np.array([i*n*n+k for i in (0,11,31) for k in range(n)],np.int32)
    topology=np.arange(n**3,dtype=np.int32)
    donor=np.array([i*n*n+(k+np.arange(-2,3))%n for i in (0,11,31) for k in range(n)],np.int32)
    wall=np.arange(2*n,3*n,dtype=np.int32);nw=len(wall)
    rng=np.random.default_rng(721)
    rows=rng.uniform(.1,1,(nr,5,nd));rows/=rows.sum(-1,keepdims=True)
    nodes=np.column_stack((np.ones(nw*35),np.zeros(nw*35),np.repeat(np.arange(n),35)*2*np.pi/n))
    slots=np.column_stack((np.ones(nw*5),np.ones(nw*5),np.repeat(np.arange(n),5)*2*np.pi/n))
    queries=np.concatenate((nodes,slots))
    a=dict(owners=raw,raw=raw,raw_to_owner=np.arange(nr,dtype=np.int32),
           raw_weight=np.ones(nr),owner_raw=np.arange(nr,dtype=np.int32)[:,None],
           owner_weight=np.ones((nr,1)),donor=donor,mask=np.ones((nr,nd),bool),
           row_count=np.full(nr,nd,np.int32),wall_index=wall,row_value_D=rows,
           row_value_N_wall=rows[wall].copy(),
           boundary_value_D_trace=rng.normal(size=(nw,5,35))*.001,
           boundary_value_N_normal=rng.normal(size=(nw,5,35))*.001,
           diffusion_D=rng.normal(size=(2,nr,nd)),
           diffusion_N_wall=rng.normal(size=(2,nw,nd)),
           boundary_D_node=rng.normal(size=(2,nw,35))*.01,
           boundary_D_tangent=rng.normal(size=(2,nw,3,2))*.01,
           boundary_N_normal=rng.normal(size=(2,nw,35))*.01,
           query_table=queries,wall_node_query=np.arange(nw*35,dtype=np.int32).reshape(nw,35),
           wall_slot_query=(nw*35+np.arange(nw*5,dtype=np.int32)).reshape(nw,5),
           boundary_wall_normal=np.tile([1.,0,0],(nw,35,1)),
           magnetic_b=np.tile([0.,0,1.],(nr,5,1)),magnetic_L=np.tile([-1.,1.,.01],(2,nr,1)))
    identity=dict(campaign_identity='mechanics',source_identity='synthetic-saved-shape',
        geometry_identity='algebra-only',topology_hash=sha256_array(topology.astype(np.int64))+':'+sha256_array(np.ones(n**3)),trace_hash=sha256_array(raw)+':'+sha256_array(np.zeros((nr,4,3))),
        support='synthetic-only',n=n,n_owner=n**3,trace_steps=64,trace_method='RK4')
    metadata=dict(identity,schema='drbx.q-paired-bank.v1',spans=(1/16,1/32),
        slot_names=('outer_minus','inner_minus','center','inner_plus','outer_plus'),
        span_metadata=[dict(identity,span=span) for span in (1/16,1/32)],has_gradients=False)
    diagnostics=dict(choice=np.full(nr,-1,np.int8),indicator=np.zeros((nr,3,2)),
                     noise=np.zeros((nr,3)),slot_points=np.zeros((nr,5,3)))
    return QBank(metadata,a,diagnostics).validate(),topology


def synthetic_case(span=1/32):
    bank,topology=synthetic_bank();nr=len(bank.raw)
    plan=lower_q_plan(bank,diffusion_span=span,magnetic_L=np.tile([-1.,1.,.01],(nr,1)),
                     b_eta=np.ones(nr),eta_step=np.array(.01),bmag=np.ones(nr))
    nowner=bank.metadata['n_owner'];axis=np.arange(nowner)
    state=np.array([1.,1.1,.9,.13,.08,.2])[:,None]+.001*np.sin(axis)[None,:]
    phi=.03*state[0]
    rng=np.random.default_rng(913)
    bc=QBoundaryData(*(rng.normal(size=shape)*.001 for shape in
                      ((6,nr,35),(6,nr,3),(6,nr,3,2),(6,nr,35))))
    pb=QBoundaryData(*(x[0] for x in bc))
    return bank,topology,plan,state,bc,phi,pb


def action(plan,state,bc,phi,pb):
    return apply_q_plan(plan,state,bc,bc,phi,pb,np.arange(1,7)*.01,
        kinds=('D','N','D','N','D','N'),phi_kind='N',tau=1.,mu=1836.)


@pytest.mark.parametrize('shards',[1,2,4])
def test_complete_owners_queries_reach_and_safe_padding(shards):
    bank,topology=synthetic_bank()
    nz,offsets,reach=q_donor_reach(bank,topology)
    assert reach==(-2,2)
    locals=localize_q_bank(bank,topology,shards)
    np.testing.assert_array_equal(np.sort(np.concatenate([b.raw for b in locals])),bank.raw)
    np.testing.assert_array_equal(np.sort(np.concatenate([b.owners for b in locals])),bank.owners)
    assert all(not isinstance(b,QBank) for b in locals)
    for local in locals:
        sh=local.shard;r=sh.raw_selector;w=sh.wall_selector
        np.testing.assert_array_equal(local.query_table,bank.query_table[sh.query_ids])
        np.testing.assert_array_equal(local.query_table[local.wall_node_query],
                                      bank.query_table[bank.wall_node_query[w]])
        np.testing.assert_array_equal(local.query_table[local.wall_slot_query],
                                      bank.query_table[bank.wall_slot_query[w]])
        np.testing.assert_array_equal(sh.extended_owner_ids[local.donor][sh.nonzero],bank.donor[r][sh.nonzero])
        assert sh.reach==(-2,2)
        assert len(local.query_table)==len(w)*40
        assert not np.any(local.owner_raw[local.owner_weight!=0]<0)
        np.testing.assert_array_equal(local.raw,np.asarray(bank.raw)[r])


def test_reject_missing_members_wrong_topology_and_insufficient_halo():
    bank,topology=synthetic_bank()
    with pytest.raises(ValueError,match='exceeds declared halo'):
        q_shard_layouts(bank,topology,4,halo=1)
    with pytest.raises(ValueError,match='divide'):
        q_shard_layouts(bank,topology,3)
    with pytest.raises(ValueError,match='at least'):
        q_shard_layouts(bank,topology,32,halo=2)
    other=topology.copy();other[32]=0
    _,other=np.unique(other,return_inverse=True)
    with pytest.raises(ValueError,match='topology identity|equally many|owner count'):
        q_shard_layouts(bank,other,2)
    bad=topology.copy();bad[32]=0 # same ring/plane, but owner0 now has another raw member
    bad[0]=32;bad[32]=32 # owner0 missing is explicitly rejected by shared authority
    with pytest.raises(ValueError,match='topology identity|no raw cell'):
        q_donor_reach(bank,bad)


def test_zero_weight_donors_are_redirected_to_trash():
    bank,topology=synthetic_bank()
    arrays={k:a.copy() for k,a in bank.arrays.items()}
    arrays['row_value_D'][:,:,4]=0;arrays['diffusion_D'][:,:,4]=0
    arrays['row_value_N_wall'][:,:,4]=0;arrays['diffusion_N_wall'][:,:,4]=0
    arrays['donor'][:,4]=12345
    sparse=QBank(bank.metadata,arrays,bank.diagnostics).validate()
    for sh in q_shard_layouts(sparse,topology,4):
        assert np.all(sh.donor[:,4]==sh.trash)
        assert np.all(~sh.nonzero[:,4])


@pytest.mark.parametrize('span',[1/16,1/32])
@pytest.mark.parametrize('shards',[1,2,4])
def test_local_full_six_field_action_matches_global_synthetic(span,shards):
    bank,topology,plan,x,bc,phi,pb=synthetic_case(span)
    ref=jax.jit(action)(plan,x,bc,phi,pb)
    for local in localize_q_plan(plan,bank,topology,shards):
        sh=local.shard
        got=jax.jit(action)(local.plan,local_owner_values(x,sh),local_boundary_data(bc,sh),
                           local_owner_values(phi,sh),local_boundary_data(pb,sh))
        for name in ('centered','correction','diffusion','combined'):
            np.testing.assert_allclose(getattr(got,name),np.asarray(getattr(ref,name))[sh.owner_selector],
                                      rtol=1e-12,atol=1e-10)
        np.testing.assert_array_equal(got.inputs_valid,np.asarray(ref.inputs_valid)[sh.raw_selector])


@pytest.fixture
def actual_case(assembled,patch):
    inner,outer,_,x,bi,bo,phi,pb=assembled
    runtime,_=rhs_prep.prepare_six_field_rhs(inner,outer,None,diffusion_span=1/32)
    bank=build_q_bank(outer,inner)
    plan=lower_q_plan(bank,diffusion_span=1/32,material_runtime=runtime)
    a=patch[0]
    from drbx.stencils.q_parallel import topology_from_arrays
    t=topology_from_arrays(tuple(a[f'centers_{i}'] for i in range(3)),a['active'],a['aggregate'],a['volume'])
    return bank,t.ro,plan,x,bi,bo,phi,pb


@pytest.mark.parametrize('shards',[1,2,4])
def test_portable_actual_hsx_local_action_and_empty_shards(actual_case,shards):
    bank,topology,plan,x,bi,bo,phi,pb=actual_case
    kinds=('D','N','D','N','D','N')
    def apply(plan,x,bi,bo,p,pb):
        return apply_q_plan(plan,x,bi,bo,p,pb,np.arange(1,7)*.01,
            kinds=kinds,phi_kind='N',tau=1.,mu=1836.)
    ref=apply(plan,x,bi,bo,phi,pb)
    raw_ref=reconstruct_q_state(plan,x,bi,bo,kinds=kinds)
    for local in localize_q_plan(plan,bank,topology,shards):
        sh=local.shard;xs=local_owner_values(x,sh)
        bis,bos=local_boundary_data(bi,sh),local_boundary_data(bo,sh)
        scalar=reconstruct_q_state(local.plan,xs,bis,bos,kinds=kinds)
        np.testing.assert_allclose(scalar.value,np.asarray(raw_ref.value)[:,sh.raw_selector],rtol=0,atol=2e-12)
        if not len(sh.raw_ids):continue # Bounded fixture covers eta plane0 only.
        got=apply(local.plan,xs,bis,bos,local_owner_values(phi,sh),local_boundary_data(pb,sh))
        for name in ('centered','correction','diffusion','combined'):
            np.testing.assert_allclose(getattr(got,name),np.asarray(getattr(ref,name))[sh.owner_selector],
                                      atol=1e-10,rtol=1e-12)


def test_neutral_exports_preserve_p_identity():
    from drbx.native import owner_plane_layout as neutral
    from drbx.native import fci_perpendicular_sharding as p
    from drbx.native import fci_perpendicular_plane_preconditioner as solver
    assert solver.owner_layout is neutral.owner_layout
    for name in ('PlaneLayout','plane_major_permutation','to_plane_major','from_plane_major','exchange_plane_halo'):
        assert getattr(p,name) is getattr(neutral,name)


def test_collective_halo_and_local_source_action_forced_cpu_devices():
    env=dict(os.environ,JAX_ENABLE_X64='true',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env['XLA_FLAGS']=env.get('XLA_FLAGS','')+' --xla_force_host_platform_device_count=4'
    run=subprocess.run([sys.executable,str(Path(__file__)),'--device-case'],env=env,capture_output=True,text=True,timeout=240)
    assert run.returncode==0,run.stderr[-4000:]
    receipt=json.loads(run.stdout.strip().splitlines()[-1])
    assert receipt['devices']==4
    # Existing Q action replay budget for a changed compiled batch shape.
    # Preserve the separately declared per-localizer budget unchanged.
    assert all(v<1e-10 for k,v in receipt.items() if k.startswith('shards'))
    assert all(v<=1e-8 for k,v in receipt.items() if k.startswith(('stacked','empty_')))



def bounded_synthetic_case(span=1/32):
    bank,topology,plan,x,bc,phi,pb=synthetic_case(span)
    keep=np.array([0,32,64]);wall=np.array([0])
    arrays={k:v.copy() for k,v in bank.arrays.items()}
    for k in ('owners','raw','raw_weight','mask','row_count','row_value_D','magnetic_b','donor'):
        arrays[k]=arrays[k][keep]
    for k in ('row_value_N_wall','boundary_value_D_trace','boundary_value_N_normal',
              'boundary_wall_normal','wall_node_query','wall_slot_query'):
        arrays[k]=arrays[k][wall]
    for k in ('diffusion_D','magnetic_L'):arrays[k]=arrays[k][:,keep]
    for k in ('diffusion_N_wall','boundary_D_node','boundary_D_tangent','boundary_N_normal'):
        arrays[k]=arrays[k][:,wall]
    arrays.update(raw_to_owner=np.arange(3,dtype=np.int32),owner_raw=np.arange(3,dtype=np.int32)[:,None],
                  owner_weight=np.ones((3,1)),wall_index=np.array([2],np.int32))
    metadata=dict(bank.metadata,trace_hash=sha256_array(arrays['raw'])+':'+sha256_array(np.zeros((3,4,3))))
    metadata['span_metadata']=[dict(m,trace_hash=metadata['trace_hash']) for m in metadata['span_metadata']]
    small=QBank(metadata,arrays,{k:v[keep] for k,v in bank.diagnostics.items()}).validate()
    q=lower_q_plan(small,diffusion_span=span,magnetic_L=plan.magnetic_L[keep],b_eta=plan.b_eta[keep],
                   eta_step=plan.eta_step,bmag=plan.bmag[keep])
    bc=QBoundaryData(*(a[:,keep] for a in bc));pb=QBoundaryData(*(a[keep] for a in pb))
    return small,topology,q,x,bc,phi,pb

def device_case():
    bank,topology,plan,x,bc,phi,pb=synthetic_case()
    perm,inverse,m=plane_major_permutation(topology,32)
    fields=np.concatenate((x,phi[None]),axis=0).T
    pm=to_plane_major(fields,inverse).reshape(32,m,7)
    ref=jax.jit(action)(plan,x,bc,phi,pb)
    out={'devices':len(jax.devices())}
    for sz in (1,2,4):
        mesh=Mesh(np.asarray(jax.devices()[:sz],object),('z',));p=32//sz
        def body(owned):return exchange_plane_halo(owned,2,'z',sz)
        exchanged=jax.shard_map(body,mesh=mesh,in_specs=P('z'),out_specs=P('z'),check_vma=False)(jnp.asarray(pm))
        received=np.asarray(exchanged).reshape(sz,p+4,m,7)
        worst=0.
        for local in localize_q_plan(plan,bank,topology,sz):
            sh=local.shard
            ext=np.concatenate((received[sh.shard].reshape(-1,7),np.zeros((1,7))))
            np.testing.assert_array_equal(ext[:-1],fields[sh.extended_owner_ids[:-1]])
            device=jax.devices()[sh.shard]
            args=(local.plan,ext[:,:6].T,local_boundary_data(bc,sh),ext[:,6],local_boundary_data(pb,sh))
            with jax.default_device(device):
                placed=jax.tree.map(lambda a:jax.device_put(a,device),args)
                got=jax.jit(action)(*placed)
            assert next(iter(got.combined.devices()))==device
            worst=max(worst,float(np.max(np.abs(np.asarray(got.combined)-np.asarray(ref.combined)[sh.owner_selector]))))
        out[f'shards{sz}']=worst
        stacked=shard_q_plan(plan,bank,topology,sz)
        compiled=jax.jit(lambda qp,xx,bb,pp,ppb,cc:sharded_q_rhs(qp,xx,bb,bb,pp,ppb,cc,
            kinds=('D','N','D','N','D','N'),phi_kind='N',tau=1.,mu=1836.,mesh=mesh))
        coefficients=np.arange(1,7)*.01
        mapped=compiled(stacked,x,bc,phi,pb,coefficients)
        cache=compiled._cache_size()
        changed=replace(stacked,plan=replace(stacked.plan,bmag=stacked.plan.bmag*1.001))
        compiled(changed,x,bc,phi,pb,coefficients*1.01).combined.block_until_ready()
        assert compiled._cache_size()==cache
        difference=0.
        for a,b in zip(jax.tree.leaves(mapped),jax.tree.leaves(ref),strict=True):
            if a.dtype.kind=='b':assert np.array_equal(a,b)
            else:difference=max(difference,float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
        out[f'stacked{sz}']=difference
        # A live BC failure on one populated wall shard invalidates every raw
        # output under the legacy global combined/diffusion finite contract.
        altered=[np.asarray(a).copy() for a in bc];altered[0][0,64,0]=np.nan
        badbc=QBoundaryData(*altered)
        expected_bad=action(plan,x,badbc,phi,pb)
        bad=compiled(stacked,x,badbc,phi,pb,coefficients)
        np.testing.assert_array_equal(bad.inputs_valid,expected_bad.inputs_valid)
        assert not np.any(bad.inputs_valid)
    # Sparse source coverage leaves whole shards empty. Dummy rows use a
    # private guard, and all dummy/raw/output diagnostics are trimmed.
    for span in (1/16,1/32):
        small,topology,q,sx,sbc,sp,spb=bounded_synthetic_case(span)
        expected=jax.jit(action)(q,sx,sbc,sp,spb)
        for sz in (1,2,4):
            stacked=shard_q_plan(q,small,topology,sz)
            assert sum(bool(len(sh.raw_ids)) for sh in stacked.shards)==1
            call=jax.jit(lambda qp,xx,bb,pp,ppb:sharded_q_rhs(qp,xx,bb,bb,pp,ppb,np.arange(1,7)*.01,
                kinds=('D','N','D','N','D','N'),phi_kind='N',tau=1.,mu=1836.))
            mapped=call(stacked,sx,sbc,sp,spb)
            assert mapped.combined.shape==(3,6) and mapped.inputs_valid.shape==(3,)
            difference=0.
            for a,b in zip(jax.tree.leaves(mapped),jax.tree.leaves(expected),strict=True):
                if a.dtype.kind=='b':assert np.array_equal(a,b)
                else:difference=max(difference,float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
            out[f'empty_span{int(1/span)}_shards{sz}']=difference
            # Outside all bounded donor closures: still globally invalid.
            invalid=sx.copy();invalid[0,20]=np.nan
            original=action(q,invalid,sbc,sp,spb)
            failed=call(stacked,invalid,sbc,sp,spb)
            np.testing.assert_array_equal(failed.inputs_valid,original.inputs_valid)
            assert not np.any(failed.inputs_valid)
    print(json.dumps(out))


if __name__=='__main__' and '--device-case' in sys.argv:device_case()


@pytest.mark.parametrize('name',['diffusion_D','diffusion_N_wall','boundary_D_node',
    'boundary_D_tangent','boundary_N_normal','boundary_value_D_trace','boundary_value_N_normal'])
def test_localizer_rejects_changed_plan_action_authority(name):
    bank,topology,plan,*_=synthetic_case()
    value=np.asarray(getattr(plan,name)).copy();value.flat[0]+=.001
    with pytest.raises(ValueError,match=name):
        localize_q_plan(replace(plan,**{name:value}),bank,topology,2)


@pytest.mark.parametrize('shards,halo',[(2.1,2),(2,2.1),(True,2),(2,False)])
def test_localizer_rejects_noninteger_layout(shards,halo):
    bank,topology=synthetic_bank()
    with pytest.raises(ValueError,match='integer'):
        q_shard_layouts(bank,topology,shards,halo=halo)


def test_complete_owner_check_rejects_omitted_raw_member():
    bank,topology=synthetic_bank()
    # Swap two identities so every owner remains nonempty and equal plane counts
    # are retained, while bank source rows no longer match their owner IDs.
    changed=topology.copy();changed[0],changed[32]=changed[32],changed[0]
    with pytest.raises(ValueError,match='topology identity|every raw member'):
        q_donor_reach(bank,changed)



def test_local_boundary_rejects_incomplete_global_data():
    bank,topology,plan,x,bc,*_=synthetic_case()
    shard=q_shard_layouts(bank,topology,4)[0]
    truncated=QBoundaryData(*(a[:,:-1] for a in bc))
    with pytest.raises(ValueError,match='global Q boundary raw shape'):
        local_boundary_data(truncated,shard)


def test_stacked_step_dtype_and_dynamic_pytree():
    bank,topology,plan,*_=synthetic_case()
    plan=replace(plan,eta_step=np.array(.01,np.float32))
    stacked=shard_q_plan(plan,bank,topology,4)
    assert stacked.plan.eta_step.dtype==plan.eta_step.dtype
    transformed=jax.tree.map(lambda a:a,stacked)
    assert transformed.shards==()
    assert transformed.n_shards==4
    np.testing.assert_array_equal(transformed.inverse,stacked.inverse)


def test_unobserved_topology_changes_are_stale_identity():
    bank,topology=synthetic_bank()
    changed=topology.copy();changed[12345],changed[12377]=changed[12377],changed[12345]
    with pytest.raises(ValueError,match='frozen topology identity'):
        q_donor_reach(bank,changed)
