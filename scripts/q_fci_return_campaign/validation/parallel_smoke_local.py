from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np,time
W=Path(__file__).resolve().parents[4];sys.path.insert(0,str(W/'DRBX'))
from scripts.q_fci_return_campaign import campaign as c

def main():
    old=W/'work/q_fci_handoff_preflight_v2_20260923/preflight/N32';out=W/'work/q_fci_handoff_audit_20260923/parallel_smoke';c.unpack(out)
    args=SimpleNamespace(workers=2,input_root=W,output=out);selection=c.read(old/'selection.json');previous=c.read(old/'summary.json');jobs=[]
    with np.load(old/'actions.npz') as z:
        for name in ('ordinary','wall'):
            track=next(x for x in selection['tracks'] if x['name']==name);owner=track['owner'];pos=np.flatnonzero(z['owners']==owner)[0]
            jobs.append((name,owner,z['reference'][pos],z['numerical_action'][pos]))
    start=time.monotonic();results=list(c.pool_jobs(args,32,jobs,c.do_strong))
    for result in results:
        expected=next(x for x in previous['strong_volume_checks'] if x['track']==result['track'])
        for key in ('volume_q9','reference_q9_volume','face_q11_minus_strong','numerical_minus_strong'):
            np.testing.assert_allclose(result[key],expected[key],atol=1e-13,rtol=1e-11)
    pids=sorted({x['worker_pid'] for x in results});assert len(pids)==2
    receipt={'passed':True,'worker_pids':pids,'seconds_including_spawn':time.monotonic()-start,'results':results,'source':c.source_identity(),'scope':'Two existing N32 strong-volume controls only; no N48/N64 preflight restart.'}
    c.write(out/'receipt.json',receipt);print('Parallel N32 strong-volume checks reproduced stored values; worker PIDs:',pids,flush=True)
if __name__=='__main__':main()
