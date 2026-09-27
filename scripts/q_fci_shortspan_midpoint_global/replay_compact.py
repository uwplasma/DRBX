"""Independent compact-row replay of immutable Q preflight maps."""
import argparse,hashlib,json,os,resource,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.q_fci_return_campaign import numerics as qnum
from .compact import compose,apply,normal_contravariant,raw_owner_states
from .fields import callable_field

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(x):return np.asarray(x['real'])+1j*np.asarray(x['imag'])
def compare(a,b):return float(np.max(abs(a-b)))
def replay(preflight,input_root,output):
    t=time.process_time();preflight=Path(preflight);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sample=json.loads((preflight/'sample.json').read_text());roles={(r['N'],r['owner']):r for r in sample['roles']}
    catalog=json.loads((Path(__file__).parent/'catalogue.json').read_text());names=catalog['fields'];pairs=catalog['boundary_pairs']
    records=[];owner_calc=defaultdict(list);worst=[];perN=[]
    for n in (32,48,64):
        start=time.process_time();ctx=qnum.context(n,input_root,json.loads((Path(__file__).parent.parent/'q_fci_projected_campaign/configuration.json').read_text()))
        state=raw_owner_states(ctx,names,callable_field);context_cpu=time.process_time()-start
        nrows=0;compact_cpu=apply_cpu=0.;maxraw=maxE=0.
        for stage in ('pilot','full'):
            prep=json.loads((preflight/f'{stage}_prepare.json').read_text());score=json.loads((preflight/f'{stage}_score.json').read_text())
            lookup={(r['N'],r['raw_id'],r['BC'],r['field']):r for r in score['raw']}
            with np.load(preflight/f'{stage}_traces.npz') as z:seeds=z['seeds'].copy();ends=z['endpoints'].copy()
            for rec in prep['records']:
                if rec['N']!=n:continue
                assert rec['selected'] and rec['map_ref']
                p=Path(rec['map_ref']['path']);assert sha(p)==rec['map_ref']['sha256']
                with np.load(p) as z:data={k:z[k].copy() for k in z.files}
                role=roles[n,rec['owner']];member=next(x for x in role['members'] if x['raw_id']==rec['raw_id']);q=np.asarray(member['midpoint']);seed=seeds[rec['trace_index']];endpoint=ends[rec['trace_index']]
                st=time.process_time();compact=compose(ctx,seed,q,data,rec['kind'],rec['owner']);compact_cpu+=time.process_time()-st
                for name in names:
                    kind=rec['kind'];allowed=pairs.get(name,pairs['waves'])
                    if kind!='interior' and kind not in allowed:continue
                    saved=lookup[n,rec['raw_id'],kind,name];field=callable_field(name);donor=state[name][compact['donors']];f0=state[name][rec['owner']]
                    st=time.process_time()
                    if kind=='interior':bc=np.empty(0)
                    else:
                        v,g=field(compact['wall_nodes']);bc=v if kind=='D' else np.einsum('ni,ni->n',normal_contravariant(ctx,compact['wall_nodes']),g)
                    numerical=apply(compact,donor,f0,bc,kind)
                    exact=(compact['K']@field(endpoint.reshape(-1,3))[0]).reshape(2,3)
                    apply_cpu+=time.process_time()-st
                    nd=compare(numerical,unpack(saved['N_action']));ed=compare(exact,unpack(saved['E_action']));maxraw=max(maxraw,nd);maxE=max(maxE,ed)
                    scale=max(1.,float(np.max(abs(unpack(saved['N_action'])))))
                    worst.append(dict(N=n,raw_id=rec['raw_id'],kind=kind,field=name,N_abs=nd,N_scaled=nd/scale,E_abs=ed))
                    owner_calc[n,rec['owner'],name,kind].append((member['projection_weight'],numerical,exact))
                    nrows+=1
        perN.append(dict(N=n,raw_field_cases=nrows,context_and_state_cpu_s=context_cpu,compact_cpu_s=compact_cpu,apply_cpu_s=apply_cpu,max_raw_N_abs=maxraw,max_raw_E_abs=maxE,maxrss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3))
    savedowner={}
    for stage in ('pilot','full'):
        score=json.loads((preflight/f'{stage}_score.json').read_text())
        for row in score['owners']:savedowner[row['N'],row['owner'],row['field'],row['BC']]=row
    ownermax=ownerE=0.;ownerrows=0
    for key,vals in owner_calc.items():
        row=savedowner[key];assert len(vals)==row['raw_count']
        nn=sum(w*N for w,N,E in vals);ee=sum(w*E for w,N,E in vals)
        ownermax=max(ownermax,compare(nn,unpack(row['N_action'])));ownerE=max(ownerE,compare(ee,unpack(row['E_action'])));ownerrows+=1
    assert ownerrows==len(savedowner)
    assert max(x['max_raw_N_abs'] for x in perN)<2e-8 and max(x['max_raw_E_abs'] for x in perN)<2e-8 and ownermax<2e-8 and ownerE<2e-8
    out=dict(schema='q-compact-preflight-replay-v1',input_preflight=str(preflight.resolve()),input_root=str(Path(input_root).resolve()),preflight_sample_sha256=sha(preflight/'sample.json'),raw_field_cases=len(worst),complete_owner_field_cases=ownerrows,
             per_resolution=perN,max_owner_N_abs=ownermax,max_owner_E_abs=ownerE,worst=sorted(worst,key=lambda x:x['N_abs'],reverse=True)[:20],total_cpu_s=time.process_time()-t)
    (output/'compact_replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='worst'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',required=True);p.add_argument('--input-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();replay(a.preflight,a.input_root,a.output)
