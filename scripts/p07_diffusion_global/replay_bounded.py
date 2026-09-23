#!/usr/bin/env python3
"""Local extraction check against the archived three-owner P07 preflight."""
import argparse,time,json
from pathlib import Path
import numpy as np
import campaign as c

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-root',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--archive',required=True,type=Path);args=ap.parse_args()
 args.output=args.output.resolve();args.input_root=args.input_root.resolve();args.output.mkdir(parents=True,exist_ok=True)
 cfg,ident=c.verify(args);report={}
 for size in (32,48,64):
  start=time.monotonic();c.initialize(args.input_root,args.output,size,cfg,ident);s=c.STATE;num=s['numerics']
  d=json.loads((args.archive/f'N{size}.json').read_text());z=np.load(args.archive/f'N{size}.evidence.npz');keys=z['face_keys'];ids=num.face_indices(size,keys)
  faces=num.face_chunk(s,ids);raw=np.concatenate([np.ravel_multi_index(np.array(t['raw_members']).T,(size,)*3) for t in d['tracks']]);ref=num.cell_chunk(s,raw,7)
  from drbx.geometry.fci_perpendicular_bracket import build_face_block
  pts,w=num.quadrature(s['faces'],keys,3,face=True)
  block=build_face_block(s['geometry'],ids,keys,pts,w,lambda q:np.zeros((len(q),3)),block_size=len(ids))
  for row in range(len(ids)):
   count=int(faces['donor_count'][row]);assert count==block.donor_count[row]
   np.testing.assert_array_equal(faces['donors'][row,:count],block.donors[row,:count])
  fd={}
  for j,f in enumerate(num.FIELDS):
   action=z['incidence'].T@faces['flux'][:,j];truth=[];offset=0
   for t in d['tracks']:
    count=len(t['raw_members']);truth.append(sum(ref['numerator'][offset:offset+count,j])/sum(ref['continuous_volume'][offset:offset+count]));offset+=count
   ae=float(max(abs(action-z[f+'.action'])));fe=float(max(abs(faces['flux'][:,j]-z[f+'.flux'])));re=float(max(abs(np.array(truth)-z[f+'.continuum'])))
   np.testing.assert_allclose(action,z[f+'.action'],rtol=1e-10,atol=1e-9)
   np.testing.assert_allclose(truth,z[f+'.continuum'],rtol=1e-10,atol=1e-9)
   np.testing.assert_allclose(faces['flux'][:,j],z[f+'.flux'],rtol=1e-10,atol=1e-12)
   fd[f]={'action_max_abs_difference':ae,'face_flux_max_abs_difference':fe,'q7_reference_max_abs_difference':re}
  report[str(size)]={'fields':fd,'donor_identity_pass':True,'seconds':time.monotonic()-start,'archive_json_sha256':c.sha(args.archive/f'N{size}.json'),'archive_arrays_sha256':c.sha(args.archive/f'N{size}.evidence.npz')}
 c.write(args.output/'replay.json',{'identity':ident,'resolutions':report,'status':'passed'})
 print(json.dumps(report))
if __name__=='__main__':main()
