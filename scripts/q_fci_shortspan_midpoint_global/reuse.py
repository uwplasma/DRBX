"""Checksummed, relocatable import of the original exact P/R chunks."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path

EXPECTED={32:(25376,32768),48:(86016,110592),64:(202304,262144)}
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def freeze_reuse(screen,path):
    screen=Path(screen);design=screen/'design.json';entries=[]
    design_sha=sha(design)
    for n,(owners,raw) in EXPECTED.items():
        lo=got_raw=0
        chunks=sorted((screen/f'N{n}').glob('chunk_*.npz'))
        if len(chunks)!=(owners+255)//256:raise RuntimeError(f'N{n} exact chunk count differs')
        for chunk in chunks:
            receipt=chunk.with_suffix('.json');r=json.loads(receipt.read_text())
            if r['N']!=n or r['lo']!=lo or r['hi']!=min(lo+256,owners) or r['design_sha256']!=design_sha:raise RuntimeError(f'exact chunk coverage differs: {chunk}')
            digest=sha(chunk)
            if r['sha256']!=digest:raise RuntimeError(f'exact chunk hash differs: {chunk}')
            entries.append(dict(npz=str(chunk.relative_to(screen)),sha256=digest,bytes=chunk.stat().st_size,receipt=str(receipt.relative_to(screen)),receipt_sha256=sha(receipt),lo=r['lo'],hi=r['hi'],raw_count=r['raw_count']))
            lo=r['hi'];got_raw+=r['raw_count']
        if (lo,got_raw)!=(owners,raw):raise RuntimeError(f'N{n} exact coverage differs')
    result=dict(schema='q-exact-reuse-v1',original_design_sha256=design_sha,original_design=json.loads(design.read_text()),chunks=entries,expected={str(n):dict(owners=a,raw=b) for n,(a,b) in EXPECTED.items()})
    write(path,result)
    return dict(chunks=len(entries),design_sha256=design_sha)
def validate_reuse(screen,path):
    screen=Path(screen);m=json.loads(Path(path).read_text())
    if m['schema']!='q-exact-reuse-v1' or sha(screen/'design.json')!=m['original_design_sha256'] or json.loads((screen/'design.json').read_text())!=m['original_design']:raise RuntimeError('original exact design differs')
    if len(m['chunks'])!=1227:raise RuntimeError('exact chunk inventory differs')
    for e in m['chunks']:
        npz=screen/e['npz'];receipt=screen/e['receipt']
        if not npz.is_file() or npz.stat().st_size!=e['bytes'] or sha(npz)!=e['sha256'] or sha(receipt)!=e['receipt_sha256']:raise RuntimeError(f'exact reuse file differs: {npz}')
        r=json.loads(receipt.read_text())
        if (r['lo'],r['hi'],r['raw_count'],r['sha256'])!=(e['lo'],e['hi'],e['raw_count'],e['sha256']):raise RuntimeError(f'exact receipt differs: {receipt}')
    return dict(chunks=len(m['chunks']),original_design_sha256=m['original_design_sha256'])
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','validate'));p.add_argument('--screen',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
    print(json.dumps(freeze_reuse(a.screen,a.manifest) if a.command=='freeze' else validate_reuse(a.screen,a.manifest)))
if __name__=='__main__':main()
