import sys,json,multiprocessing as mp,time
from pathlib import Path
sys.path.insert(0,'/Users/yxie/Desktop/HSX drbx/DRBX/scripts/q07_balanced_global')
import campaign as c
import numpy as np
ROOT=Path('/Users/yxie/Desktop/HSX drbx/work')
if __name__=='__main__':
 start=time.perf_counter()
 with mp.get_context('spawn').Pool(2,initializer=c.worker_init,initargs=(64,)) as pool:receipts=pool.map(c.task,[256,511])
 differences={}
 for r in receipts:
  b=r['block'];p=f'chunks/N64/block_{b:06d}.npz'
  with np.load(c.OUT/p) as z:
   for label,folder,cols,oldcols in [('balanced','q07-balanced-material-3f618689-20261001T020648Z/results_hash_recovery_v1',[0,1,2,5,6,7,10,11,12],list(range(9,18))),('velocity','q07_material_global_20260930',[3,4,8,9,13,14],[3,4,8,9,13,14])]:
    with np.load(ROOT/folder/p) as old:
     np.testing.assert_array_equal(z['owners'],old['owners']);np.testing.assert_array_equal(z['count'],old['count'])
     for key in ('sum2','maximum','signed_sum'):
      a=np.take(z[key][...,:18,:],cols,axis=-1);bb=np.take(old[key],oldcols,axis=-1);vol=z['volume'][:,None,None,None,None];mask=z['volume']>0
      if key=='sum2':a=np.sqrt(a[mask]/vol[mask]);bb=np.sqrt(bb[mask]/vol[mask])
      if key=='signed_sum':a=a[mask]/vol[mask];bb=bb[mask]/vol[mask]
      err=float(abs(a-bb).max());differences[f'{b}_{label}_{key}']=err
      if err>1e-8:raise ValueError(f'full chunk baseline {b} {label} {key} {err}')
  # Check completed checkpoint path really reuses the same payload.
  checked=c.checked_saved(64,b,c.plans(64)['global_'][b]);assert checked==r
 c.write_json(c.OUT/'full_chunk_smoke.json',dict(passed=True,identity=c.digest(c.design()),blocks=[256,511],workers=2,seconds=time.perf_counter()-start,baseline_difference=differences,receipts=receipts))
 print('PASS',json.dumps(differences))
