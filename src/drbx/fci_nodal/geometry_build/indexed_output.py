"""Write the geometry and Q indexes from one local stream of owner leaves."""
from contextlib import closing
import os
from pathlib import Path
import sqlite3

import numpy as np

from .parallel_geometry_reader import PARALLEL_GEOMETRY_READER_SCHEMA, encode, runs
from .reader import GEOMETRY_SCHEMA
from .storage import FrameWriter, connect_ro, dump, load
from .._hashing import sha256_file as sha


class Writer:
 def __init__(self,root,m,plan,published_root=None):
  self.root=Path(root);self.m=m;self.plan=plan;self.storage_bins=plan['storage_bins'];self.spec={};self.frames=FrameWriter(root)
  self.root.mkdir(parents=True,exist_ok=True)
  self.reference_root=self.root if published_root is None else published_root
  self.index=self.root/'index.sqlite';self.q_index=self.root/'parallel_geometry_spans.sqlite'
  self.db=sqlite3.connect(self.index)
  self.q=sqlite3.connect(self.q_index)
  self.db.executescript('''
   create table chunks(name text,lo integer,hi integer,owner integer,path text,key text,ids_key text,count integer,sequence integer,primary key(name,path));
   create index ranges on chunks(name,lo,hi);
   create index destinations on chunks(name,owner,sequence);
   create table leaf_hashes(path text primary key,sha256 text);
   create table fit_keys(owner integer,key integer,primary key(owner,key));
  ''')
  self.q.executescript('''
   create table spans(array text,lo integer,hi integer,ref text,key text,local_lo integer,primary key(array,lo));
   create index coverage on spans(array,lo,hi);
  ''')
  self.sequence=0;self.shift=0;self.last_sequence=0;self.part=None

 def part_start(self,part):
  if self.part is not None:self.shift+=self.last_sequence+1
  self.part=part;self.sequence=0;self.last_sequence=0

 def owner(self,plane):
  return min(int(plane)*self.storage_bins//self.m['N'],self.storage_bins-1)

 def emit(self,label,data,ids,owners,extra=None):
  ids=np.asarray(ids,np.int64);owners=np.broadcast_to(owners,(len(ids),));extra=extra or {}
  for owner in np.unique(owners):
   take=owners==owner;rel=f'builders/{self.part}/owners/o{owner:03d}/{label}';vals={k:np.asarray(v)[take] for k,v in data.items()};vals.update({k:np.asarray(v)[take] for k,v in extra.items()});vals['logical_ids']=ids[take]
   for name in ['faces','wall']:
    if name in vals:
     z=vals[name];keys=z[:,:2].ravel().astype(np.int64) if name=='faces' else z[:,0].astype(np.int64)
     keys=keys[keys<self.plan['NN']]
     if self.m['symmetry']=='nfp':keys=keys%(self.plan['K']*self.plan['plane_nodes'])
     self.db.executemany('insert or ignore into fit_keys values(?,?)',((int(owner),int(k)) for k in np.unique(keys)))
   j=ids[take];record=self.frames.write(rel,vals,dict(owner=int(owner),count=int(take.sum())))
   ref=encode(record['frame'],self.reference_root)
   self.q.executemany('insert into spans values(?,?,?,?,?,?)',((name,int(j[a]),int(j[b-1])+1,ref,name,a) for a,b in runs(j) for name in data))
   self.db.execute('insert into leaf_hashes values(?,?)',(rel,sha(self.root/rel/'complete.json')))
   for key,a in data.items():
    self.spec.setdefault(key,dict(shape=[0,*np.shape(a)[1:]],dtype=np.asarray(a).dtype.str));self.spec[key]['shape'][0]=max(self.spec[key]['shape'][0],int(ids[-1])+1 if len(ids) else 0)
    self.db.execute('insert into chunks values(?,?,?,?,?,?,?,?,?)',(key,int(j[0]),int(j[-1])+1,int(owner),rel,key,'logical_ids',len(j),self.shift+self.sequence))
   self.last_sequence=self.sequence
   self.sequence+=1

 def keys(self):
  return [key for key, in self.db.execute('select distinct key from fit_keys order by key')]

 def collect_part(self, root, part, published_root):
  """Attach one private worker index in canonical part order (sharded geometry scheduling and canonical collection owner_build.finish)."""
  self.part_start(part)
  done=load(root/'job_complete.json');versions=done['products']
  for name, expected in versions.items():
   if sha(root/name)!=expected:raise RuntimeError(f'Owner part changed: {root/name}')
  shard=Path('builders')/part/'geometry.gds'
  refs={}
  with closing(connect_ro(root/'index.sqlite')) as child:
   leaves=child.execute('select path,sha256 from leaf_hashes order by path')
   for rel,expected in leaves:
    leaf=self.root/rel; source=root/rel
    if sha(source/'complete.json')!=expected:raise RuntimeError(f'Owner leaf changed: {source}')
    record=load(source/'complete.json');old=encode(record['frame'],root)
    record['frame']['path']=str(shard);refs[old]=encode(record['frame'],published_root)
    record['containers']['data.bin']['path']=os.path.relpath(self.root/shard,leaf)
    dump(leaf/'complete.json',record)
    self.db.execute('insert into leaf_hashes values(?,?)',(rel,sha(leaf/'complete.json')))
   if refs:os.link(root/'geometry.gds',self.root/shard)
   for row in child.execute('select * from chunks order by sequence,name'):
    self.db.execute('insert into chunks values(?,?,?,?,?,?,?,?,?)',(*row[:-1],row[-1]+self.shift))
    self.last_sequence=row[-1]
   self.db.executemany('insert or ignore into fit_keys values(?,?)',child.execute('select owner,key from fit_keys'))
  with closing(connect_ro(root/'parallel_geometry_spans.sqlite')) as child:
   self.q.executemany('insert into spans values(?,?,?,?,?,?)',
      ((name,lo,hi,refs[ref],key,local) for name,lo,hi,ref,key,local in child.execute('select * from spans')))
  for name,spec in done['result'].items():
   before=self.spec.setdefault(name,spec)
   if before['dtype']!=spec['dtype'] or before['shape'][1:]!=spec['shape'][1:]:
    raise RuntimeError(f'Owner array schema changed: {name}')
   before['shape'][0]=max(before['shape'][0],spec['shape'][0])

 def close(self):
  self.db.close()
  self.q.close()

 def finish(self, published_root=None):
  r=self.plan;m=self.m
  for name,spec in self.spec.items():
   at=0
   for lo,hi in self.q.execute('select lo,hi from spans where array=? order by lo',(name,)):
    if lo!=at or hi<=lo:raise RuntimeError(('Q input coverage/overlap',name,at,lo,hi))
    at=hi
   if at!=spec['shape'][0]:raise RuntimeError(('Q input incomplete',name,at,spec['shape'][0]))
  self.db.commit();self.q.commit()
  index_hash=sha(self.index);spans=self.q.execute('select count(*) from spans').fetchone()[0]
  geometry=dict(schema=GEOMETRY_SCHEMA,configuration=m['fingerprint'],symmetry=m['symmetry'],
                field_periods=m['field_periods'],field_filter=m['filter_spec'].as_dict(),
                replication_periods=r['p'],source_planes=r['K'],
                N=r['N'],E=r['E'],storage_bins=r['storage_bins'],arrays=self.spec,index_sha256=index_hash,
                plane_nodes=r['plane_nodes'],physical_nodes=r['NN'],
                polygon_prefix=r['polygon_prefix'],coverage=r['coverage'])
  index_path=self.q_index.resolve() if published_root is None else published_root/self.q_index.name
  entries={name:dict(format='drbx.parallel.owner_geometry_containers.v1',index=str(index_path),array=name,
                     shape=spec['shape'],dtype=spec['dtype']) for name,spec in self.spec.items()}
  reader=dict(schema=PARALLEL_GEOMETRY_READER_SCHEMA,geometry=entries,index_sha256=sha(self.q_index),
              spans=spans,captured_inline=True,geometry_index_sha256=index_hash,N=r['N'],E=r['E'],
              P=r['plane_nodes'],NN=r['NN'],source_shard_rows=32000,
              inner_reset_rows=8192,reference_protocol='geometry_frame: indexed immutable frame',payload_copies=0)
  dump(self.root/'geometry.json',geometry)
  dump(self.root/'canonical_prefixes.json',dict(ad_per_period=r['ad_per_period'],
       physical_wall_rows=r['physical_wall_rows'],polygon_prefix=r['polygon_prefix'],p=r['p'],source_planes=r['K']))
  self.reader=reader
  return geometry

 def publish_reader(self):
  dump(self.root/'parallel_geometry_reader.json',self.reader)


def publish_fits(root,m,geometry,keys,fit):
 root=Path(root);database=root/'plane_fit_keys.sqlite';maximum=0
 with closing(sqlite3.connect(database)) as db:
  db.executescript('''
   create table keys(key integer primary key,ordinal integer unique,path text,row integer,ncol integer);
   create table leaves(path text primary key,sha256 text);
  ''')
  frames=FrameWriter(root)
  for lo in range(0,len(keys),64):
   path=f'fits/leaf_{lo:09d}';z=fit(m,keys[lo:lo+64])
   frames.write(path,z,dict(stage='fits'))
   db.execute('insert into leaves values(?,?)',(path,sha(root/path/'complete.json')))
   db.executemany('insert into keys values(?,?,?,?,?)',((int(k),lo+i,path,i,int(n)) for i,(k,n) in enumerate(zip(z['keys'],z['ncol']))))
   maximum=max(maximum,max(map(int,z['ncol']),default=0))
  db.commit()
 dump(root/'fit_manifest.json',dict(configuration=m['fingerprint'],geometry_index_sha256=geometry['index_sha256'],
      keys=len(keys),key_database_sha256=sha(database),global_maximum_width=maximum,mode=m['symmetry'],
      key_formula='plane%K,local' if m['symmetry']=='nfp' else 'actual plane,local',
      padding='ragged live coefficients; uses recorded global fit width',
      array_catalogue='immutable fits/leaf_*/complete.json'))
