"""Compact immutable observations; copy values, never reorder candidates."""
import numpy as np
from .clip_index import CandidateIndex

class LazyQuads:
 def __init__(self,vertices,ptr):
  self.vertices=np.asarray(vertices);self.ptr=np.asarray(ptr);self.counts=np.diff(self.ptr).astype(np.int64,copy=False)
  if (self.counts>4).any():raise ArithmeticError('Original polygon quad capacity exceeded')
  self.padded=np.zeros((len(self.counts),4,2),dtype=self.vertices.dtype)
  rows=np.repeat(np.arange(len(self.counts)),self.counts);cols=np.arange(len(self.vertices))-np.repeat(self.ptr[:-1],self.counts)
  self.padded[rows,cols]=self.vertices
 def __getitem__(self,i):return self.vertices[self.ptr[i]:self.ptr[i+1]]
 def __len__(self):return len(self.counts)

def entry(d,compiled):
 data={k:v for k,v in d.items() if k not in ['edges','inside','im','ia','quad_vertices','quad_indptr']}
 data['quads']=LazyQuads(d['quad_vertices'],d['quad_indptr'])
 E=np.array(d['edges'],copy=True);X=np.array(d['inside'],copy=True);M=np.array(d['im'],copy=True);A=np.array(d['ia'],copy=True)
 data['boundary']={tuple(e):(x,mm,a) for e,x,mm,a in zip(E.tolist(),X,M,A)}
 data.update(clip_topology_capture=True,candidate_mode='indexed',compiled_polygons=compiled)
 data['candidate_index']=CandidateIndex(data['bb0'],data['bb1']);return data
