"""Independent reload, raw-cell incidence, and cost audit of completed preflight."""
from pathlib import Path
import sys,json,time,math,numpy as np
W=Path(__file__).resolve().parents[4];R=W/'DRBX';RUN=W/'work/q_fci_handoff_preflight_v2_20260923';OUT=W/'work/q_fci_handoff_audit_20260923'
sys.path[:0]=[str(RUN/'software/src'),str(R)]
from scripts.q_fci_return_campaign import numerics as m
from scripts.q_fci_return_campaign.campaign import write,read,sha
report={}
for n in (32,48,64):
 root=RUN/'preflight'/f'N{n}'
 if not (root/'completion.json').exists():continue
 ctx=m.context(n,W,read(R/'scripts/q_fci_return_campaign/configuration.json'));sel=read(root/'selection.json');faces=np.array(sel['face_ids']);owners=np.array(sel['owners']);keys=ctx['keys'][faces]
 lut={tuple(k):i for i,k in enumerate(keys)};inc=np.zeros((len(owners),len(faces)));ownermap={owner:i for i,owner in enumerate(owners)}
 for raw in sel['raw_ids']:
  cell=np.array(np.unravel_index(raw,(n,n,n)));owner=ctx['topology']['compact_raw_owner'][tuple(cell)]
  for axis in range(3):
   for side,sign in ((0,-1),(1,1)):
    q=cell.copy();q[axis]+=side
    if axis:q[axis]%=n
    key=(axis,*q)
    if key in lut:inc[ownermap[owner],lut[key]]+=sign
 rows={p.stem:np.load(p,mmap_mode='r') for p in (root/'rows').glob('*.npy')};flux=np.zeros((len(faces),4));replay=0.;constant=0.;wall=0
 for path in sorted((root/'face_chunks').glob('*.npz')):
  with np.load(path) as z:
   for j,fid in enumerate(z['ids']):
    pos=int(np.searchsorted(faces,fid));ids=z['row_ids'][j];keep=ids>=0
    if keep.any():
     rebuilt=z['weights'][j,keep]@rows['numerical'][ids[keep]]
     replay=max(replay,float(abs(rebuilt-z['numerical'][j]).max()))
     # Changed field from the reloaded observations, sharing the saved map.
     mixture=np.array([.3,-.7,.2,1.1]);mix=z['weights'][j,keep]@(rows['numerical'][ids[keep]]@mixture)
     np.testing.assert_allclose(mix,z['numerical'][j]@mixture,rtol=1e-10,atol=1e-13)
    else:
     wall+=1;np.testing.assert_array_equal(z['numerical'][j],np.zeros(4))
    flux[pos]=z['numerical'][j]
 with np.load(root/'actions.npz') as z:
  action=inc@flux/z['volume'][:,None];action_error=float(abs(action-z['numerical_action']).max());np.testing.assert_allclose(action,z['numerical_action'],atol=2e-12,rtol=1e-10)
  old=np.load(W/'work/parallel_fci_local_return_refinement_20260923'/f'parent_endpoint_N{n}.npz')
  oldsummary=read(W/'work/parallel_fci_local_return_refinement_20260923'/f'summary_N{n}.json')
  comp=[]
  for name in ('ordinary','agglomerated','transition'):
   track=next(x for x in sel['tracks'] if x['name']==name);oi=int(np.flatnonzero(owners==track['owner'])[0])
   comp.append({'track':name,'owner':track['owner'],'numerical_action':z['numerical_action'][oi].tolist(),'reference':z['reference'][oi].tolist()})
 timing=read(root/'rows/manifest.json')['timings'];summary=read(root/'summary.json');nf=len(ctx['keys']);nr=2*n**3;nv=n**3
 seconds_per_row=sum(x['seconds'] for x in timing)/sum(x['rows'] for x in timing)
 seconds_per_face=sum(x['seconds'] for x in summary['face_timings'])/sum(x['faces'] for x in summary['face_timings'])
 seconds_per_cell=sum(x['seconds'] for x in summary['volume_timings'])/sum(x['cells'] for x in summary['volume_timings'])
 trace_bytes=sum(p.stat().st_size for p in (root/'trace_chunks').glob('*'))/sum(x['rows'] for x in timing)
 face_bytes=sum(p.stat().st_size for p in (root/'face_chunks').glob('*'))/len(faces)
 peak=max([x['peak_rss_gib'] for x in timing+summary['face_timings']+summary['volume_timings']])
 report[str(n)]={'reload_flux_max':replay,'independent_raw_incidence_action_max':action_error,'wall_faces':wall,'constant_max':summary['constant_max'],'sampled_rows':sum(x['rows'] for x in timing),'excluded_rows':sum(x['invalid'] for x in timing),'peak_worker_gib':peak,'canonical_original_tracks':comp,'global_counts':{'rows':nr,'faces':nf,'raw_cells':nv,'owners':len(ctx['volume']),'trace_units':2*n*math.ceil(n*n/24),'face_units':math.ceil(nf/64),'volume_units':math.ceil(nv/32)},'measured_worker_seconds_per_unit':{'row':seconds_per_row,'face':seconds_per_face,'cell':seconds_per_cell},'projected_serial_worker_hours':(nr*seconds_per_row+nf*seconds_per_face+nv*seconds_per_cell)/3600,'projected_trace_face_checkpoint_gib':(nr*trace_bytes+nf*face_bytes)/2**30,'uncertainty':'Planning projection only; allow factor 2-3 variability. Bounded sample overrepresents axis/wall and misses unsampled global exceptions; allocation scaling and context/I/O overhead are not included.'}
 write(OUT/'postflight.json',report);print(n,report[str(n)],flush=True)
