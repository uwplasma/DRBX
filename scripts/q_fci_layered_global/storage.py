"""Identity, atomic checkpoints and a single coordinator lock."""
from pathlib import Path
from contextlib import contextmanager
import fcntl,hashlib,json,os
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n');os.replace(temp,p)
def arrays(p,**data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**data)
    os.replace(temp,p)
def sources():return {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts and 'validation' not in p.parts}
def identity(root):
    design=read(Path(root)/'design.json')
    if design['sources']!=sources():raise RuntimeError('runtime source identity changed')
    return sha(Path(root)/'design.json'),design
@contextmanager
def lock(root):
    p=Path(root)/'coordinator.lock';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('campaign already has an active writer')
        yield

def complete(path,ident):
    path=Path(path);receipt=path.with_suffix('.json')
    if not path.exists() and not receipt.exists():return False
    if not receipt.exists():return False # interrupted atomic output before receipt
    z=read(receipt)
    if z['identity']!=ident or not path.exists() or sha(path)!=z['sha256']:raise RuntimeError(f'corrupt/incompatible checkpoint: {path}')
    return True

def record(path,ident,**meta):write(Path(path).with_suffix('.json'),dict(identity=ident,sha256=sha(path),**meta))
