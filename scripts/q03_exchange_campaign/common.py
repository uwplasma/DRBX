import hashlib,json,os
from pathlib import Path
import numpy as np

SCHEMA='q03-whole-support-remote-v1'
POLICY={'degree':3,'max_exchanges':64,'candidate':'whole_support','quadrature':9,'observations':['exact','G3'],'boundary':'frozen_prescribed_continuum_flux','pool':'frozen_local_five_plane_nearest128','coverage':'initial_sector_and_direction_per_plane','objective':'weighted_cubic_information'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def source_identity():
 return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
def load(root,N):
 return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (Path(root)/f'N{N}').glob('*.npy')}
def args_for(d,face,N):
 rows={k:d['rows.'+k] for k in ('source','target','length','direction','source_plane')}
 original=d['original.indices'][d['original.indptr'][face]:d['original.indptr'][face+1]]
 seed=d['seed.indices'][d['seed.indptr'][face]:d['seed.indptr'][face+1]]
 pools={}
 for j,delta in enumerate((-2,-1,0,1,2)):
  plane=(int(d['face.plane'][face])+delta)%N;r=d['pool.indices'][face,j];r=r[r>=0];r=np.setdiff1d(r,original)
  pools[plane]={'rows':r}
 return rows,original,seed,pools,d['face.center'][face],d['face.scale'][face],d['target'][face],True,64
