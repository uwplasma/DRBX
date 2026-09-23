from pathlib import Path
import argparse,sys,json,time,numpy as np
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
W=a.workspace;R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from scripts.q_fci_return_campaign import campaign as c
OUT=a.output;c.unpack(OUT)
c.initialize(32,str(W),str(OUT),trace_capacity=96);m=c.NUM;ctx=c.CTX
old=W/'work/q_fci_handoff_preflight_v2_20260923/preflight/N32';rows=c.load_catalogue(old/'rows');plan=c.read(old/'trace_plan.json')['units']
units=[plan[0],next(u for u in plan if any(16<=x//2//1024<=20 and x//2%32==13 for x in u)),plan[-1]]
report=[]
for step in (64,128):
 ctx['config']['policy']['trace_substeps']=step
 for j,ids in enumerate(units):
  t=time.monotonic();out=m.trace_rows(ctx,ids);dt=time.monotonic()-t;valid=out['valid']&rows['valid'][ids]
  error=m.regular(out['endpoint'][valid].reshape(-1,3),0)-m.regular(rows['endpoint'][ids][valid].reshape(-1,3),0)
  rec={'step':step,'unit':j,'rows':len(ids),'seconds':dt,'valid_difference':int(np.sum(out['valid']!=rows['valid'][ids])),'endpoint_regular_max':float(abs(error).max()),'ell_max':float(abs(out['ell'][valid]-rows['ell'][ids][valid]).max()),'numerical_max':float(abs(out['numerical'][valid]-rows['numerical'][ids][valid]).max())}
  report.append(rec);c.save(OUT/f'smoke_{step}_{j}.npz',**out);c.write(OUT/'smoke.json',report);print(rec,flush=True)

assert all(r['valid_difference']==0 and r['endpoint_regular_max']<1e-6 for r in report)
