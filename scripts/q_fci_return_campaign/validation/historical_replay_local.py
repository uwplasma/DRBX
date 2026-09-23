from pathlib import Path
import sys,json,time,numpy as np
W=Path(__file__).resolve().parents[4];R=W/'DRBX';OUT=W/'work/q_fci_handoff_audit_20260923'
sys.path[:0]=[str(W/'work/q_fci_handoff_preflight_v2_20260923/software/src'),str(R)]
from scripts.q_fci_return_campaign import numerics as m
from scripts.q_fci_return_campaign.campaign import sha,write
old=W/'work/parallel_fci_local_return_refinement_20260923';parent=W/'work/parallel_fci_local_return_20260923'
result={};config=json.loads((R/'scripts/q_fci_return_campaign/configuration.json').read_text())
for n in (32,48,64):
 t=time.monotonic();ctx=m.context(n,W,config);keys=ctx['keys'];labels=ctx['topology']['compact_raw_owner'];index=keys[:,1:].copy()
 low=index.copy();low[np.arange(len(keys)),keys[:,0]]-=1;low%=n
 up=index%n
 expectedlow=labels[tuple(low.T)];expectedup=labels[tuple(up.T)];wall=(keys[:,0]==0)&(keys[:,1]==n);expectedup[wall]=-1
 assert np.array_equal(expectedlow,ctx['lower']) and np.array_equal(expectedup,ctx['upper'])
 assert np.all(ctx['lower']!=ctx['upper']) and not np.any((keys[:,0]==0)&(keys[:,1]==0))
 groups=json.loads((parent/'selection.json' if n==32 else old/f'selection_N{n}.json').read_text())['groups']
 rows={reg:[] for reg in ('ordinary','agglomerated','transition')}
 for group in groups:
  path=parent/'rows'/f"{group['id']}.npz" if n==32 else old/'rows'/f'N{n}'/f"{group['id']}.npz"
  with np.load(path) as z:
   for j,ij in enumerate(z['ij']):
    key=[int(ij[0]),int(ij[1]),group['k'],group['d']]
    rows[group['region']].append({'id':int(m.row_ids(n,[key])[0]),'source':z['source_points'][j],'endpoint':z['endpoint_points'][j],'F':z['source_F'][j],'Z':z['Z'][j],'g_sec':z['g_sec'][j],'numerical':z['g_num'][j]})
 size=2*n**3;data={'valid':np.zeros(size,bool)}
 for key,shape in {'source':(4,3),'endpoint':(4,3),'F':(4,),'Z':(),'g_sec':(4,),'numerical':(4,)}.items():data[key]=np.zeros((size,)+shape)
 for rr in rows.values():
  for row in rr:
   data['valid'][row['id']]=True
   for key in data:
    if key!='valid':data[key][row['id']]=row[key]
 s=json.loads((old/f'summary_N{n}.json').read_text());actual={'numerical':[],'g_sec':[]};defects=[]
 for d in s['support_diagnostics']['geometry_96']:
  rr=rows[d['region']];ids=np.array([rr[i]['id'] for i in d['row_indices']]);fc,scale=m.face_center_scale(ctx,d['face'])
  np.testing.assert_allclose(fc,d['face_center'],atol=2e-15);np.testing.assert_allclose(scale,d['scale'],atol=2e-15)
  f,g,_,weights,meta=m.return_map(ctx,d['face'],data,frozen_ids=ids)
  actual['numerical'].append(f);actual['g_sec'].append(g);defects.append(meta['target_relative_defect'])
 checks={}
 with np.load(old/f'parent_endpoint_N{n}.npz') as ref:
  for kind in actual:
   actual[kind]=np.array(actual[kind]);checks[kind]=float(abs(actual[kind]-ref[kind+'_flux']).max())
   np.testing.assert_allclose(actual[kind],ref[kind+'_flux'],rtol=2e-9,atol=3e-16)
  # Independent raw owner ledger and saved historical flux reduction.
  ownerids=s['selection']['owners'];lut={tuple(k):i for i,k in enumerate(ref['face_keys'])};inc=np.zeros_like(ref['incidence'])
  for oi,owner in enumerate(ownerids):
   for raw in s['selection']['raw_by_owner'][str(owner)]:
    cell=np.array(np.unravel_index(raw,(n,n,n)))
    for axis in range(3):
     for side,sign in ((0,-1),(1,1)):
      q=cell.copy();q[axis]+=side
      if axis:q[axis]%=n
      inc[oi,lut[(axis,*q)]]+=sign
  assert np.array_equal(inc,ref['incidence'])
  checks['action_max']=float(abs(inc@actual['numerical']/ref['volume'][:,None]-ref['numerical_action']).max())
 # Fresh two-row endpoint/trace check against archived actual-HSX data.
 rr=rows['ordinary'][:2];fresh=m.trace_rows(ctx,[row['id'] for row in rr]);checks['fresh_g_sec_max']=float(abs(fresh['g_sec']-np.array([r['g_sec'] for r in rr])).max());checks['fresh_numerical_max']=float(abs(fresh['numerical']-np.array([r['numerical'] for r in rr])).max())
 assert checks['fresh_g_sec_max']<2e-8 and checks['fresh_numerical_max']<2e-8
 checks.update(owners=len(ctx['volume']),faces=len(keys),wall_faces=int(wall.sum()),target_defect_max=max(defects),seconds=time.monotonic()-t)
 result[str(n)]=checks;write(OUT/'audit.json',result);print(n,checks,flush=True)
 del ctx,data,rows
write(OUT/'source_identity.json',{p.name:sha(p) for p in (R/'scripts/q_fci_return_campaign').glob('*.py')})
