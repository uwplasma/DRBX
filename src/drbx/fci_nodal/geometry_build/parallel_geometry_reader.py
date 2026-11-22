from pathlib import Path
import base64,hashlib,json,struct
import numpy as np

PARALLEL_GEOMETRY_READER_SCHEMA = 'drbx.parallel.geometry_reader.v2'

FRAME_PREFIX = 'geometry_frame:'
FRAME_READER_SENTINEL = '_indexed_frame_reader_installed'

MAGIC=b'GUNIT13\0'
DONE=b'GCOMMIT1'
HEADER=struct.Struct('<8sQQ')

def encode(frame,root):
 f=dict(frame);f['path']=str((Path(root)/frame['path']).resolve());return FRAME_PREFIX+base64.urlsafe_b64encode(json.dumps(f,separators=(',',':')).encode()).decode()

def runs(ids):
 if not len(ids):return
 start=0
 for i in range(1,len(ids)):
  if int(ids[i])!=int(ids[i-1])+1:yield start,i;start=i
 yield start,len(ids)

def frame_info(ref,verify=False):
 r=json.loads(base64.urlsafe_b64decode(ref[len(FRAME_PREFIX):]));p=Path(r['path'])
 with p.open('rb') as f:
  f.seek(r['offset']);head=f.read(HEADER.size)
  if len(head)!=HEADER.size:raise RuntimeError('Torn geometry frame')
  magic,nmeta,size=HEADER.unpack(head)
  if magic!=MAGIC or nmeta>16*1024**2:raise RuntimeError('Geometry frame header differs')
  body=f.read(nmeta);meta=json.loads(body);data=((r['offset']+HEADER.size+nmeta+63)//64)*64;f.seek(data+size);tail=f.read(40)
  if len(tail)!=40 or tail[32:]!=DONE or tail[:32].hex()!=r['sha256'] or data+size+40-r['offset']!=r['bytes']:raise RuntimeError('Geometry frame is not committed')
  if verify:
   f.seek(r['offset']);h=hashlib.sha256();left=r['bytes']-40
   while left:
    b=f.read(min(left,1024**2))
    if not b:raise RuntimeError('Frame truncated')
    h.update(b);left-=len(b)
   if h.hexdigest()!=r['sha256']:raise RuntimeError('Geometry frame checksum differs')
 return r,dict(key=meta['record']['identity'],arrays=meta['record']['arrays']),data

def get(ref,keys=None,verify=False):
 r,meta,data=frame_info(ref,verify);out={}
 for key,x in meta['arrays'].items():
  if keys is not None and key not in keys:continue
  out[key]=np.memmap(r['path'],dtype=np.dtype(x['dtype']),mode='r',offset=data+x['offset'],shape=tuple(x['shape'])) if x['bytes'] else np.empty(tuple(x['shape']),np.dtype(x['dtype']))
 return out

def install_parallel_geometry_reader(indexed_stage):
 """Q adopts one I/O dispatch; its IndexedGeometryArray/View and all kernels are unchanged."""
 if getattr(indexed_stage,FRAME_READER_SENTINEL,False):return
 old_get=indexed_stage.get;old_info=indexed_stage.info
 indexed_stage.get=lambda ref,keys=None,verify=False:get(ref,keys,verify) if ref.startswith(FRAME_PREFIX) else old_get(ref,keys,verify)
 indexed_stage.info=lambda ref,verify=False:frame_info(ref,verify) if ref.startswith(FRAME_PREFIX) else old_info(ref,verify)
 setattr(indexed_stage,FRAME_READER_SENTINEL,True)
