#!/usr/bin/env python3
"""Maintainer-only generation of source and immutable-input identities."""
import argparse,json,sys
from pathlib import Path
from campaign import HERE,REPO,sha,write

def record(root,path):
 p=root/path
 return {'path':path,'bytes':p.stat().st_size,'sha256':sha(p)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-root',required=True,type=Path);args=ap.parse_args();root=args.input_root.resolve()
 cfg=json.loads((HERE/'configuration.json').read_text());inp=cfg['inputs']
 paths=[inp[k] for k in ('reference_sidecar','metric_cache','makegrid')]
 paths += [f"{inp['geometry']}/{n}x{n}x{n}/{f}" for n in (32,48,64) for f in ('base_geometry.npz','rlp_topology.npz')]
 write(HERE/'input_manifest.json',{'schema':'drbx.p07-inputs-v1','files':[record(root,p) for p in sorted(paths)]})
 # Exercise lazy evaluator imports before recording the complete loaded source closure.
 import numerics
 side=json.loads((root/inp['reference_sidecar']).read_text())
 for k in ('metric_cache','makegrid'):side[k]['path']=str(root/inp[k])
 import tempfile
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'sidecar.json';write(p,side);ref=numerics.reference(p,verify_hashes=False)
 paths=set()
 for module in list(sys.modules.values()):
  p=getattr(module,'__file__',None)
  if not p:continue
  p=Path(p).resolve()
  if p.suffix=='.py' and p.is_relative_to(REPO):paths.add(str(p.relative_to(REPO)))
 paths.update(str(p.relative_to(REPO)) for p in HERE.glob('*.py'))
 write(HERE/'source_manifest.json',{'schema':'drbx.p07-sources-v1','scope':'loaded repository Python dependency closure plus campaign entry points','files':[record(REPO,p) for p in sorted(paths)]})
if __name__=='__main__':main()
