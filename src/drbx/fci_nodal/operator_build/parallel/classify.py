"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import PLANE_WALL_STEM
import numpy as np


def bind(context):
    TAU = context.TAU
    HOP = context.HOP
    BISECT = context.BISECT
    WALL_BISECT = context.wall_bisections
    OUT = context.out
    gu = context.gu
    book = context.book
    check_projection_budget = context.check_projection_budget
    trace_cached = context.trace_cached
    trace_sides = context.trace_sides
    dump = context.dump
    sample = context.sample
    seeds = context.seeds
    edges = context.edges
    transform = context.transform

    def midpoint_observe(ev,ref,k,label):
     q=midpoint_seeds(ref,k);zz=trace_sides(ev,q,[-HOP,HOP],[label+' '+str(s) for s in [-1,1]]);times=[np.where(z['hit'],abs((z['wall'][:,2]-q[:,2]+np.pi)%TAU-np.pi)/HOP,1.) for z in zz];reach=[t>=.5 for t in times];cl=np.where(reach[0],np.where(reach[1],0,1),np.where(reach[1],2,3));pp=[sample(z,.5) for z in zz]
     return dict(classes=cl,maps=np.stack([p[:,:2] for p in pp],1),arcs=np.stack([p[:,2] for p in pp],1),wallhit=np.stack([z['wall'] for z in zz],1),hitell=np.stack([z['wall_length'] for z in zz],1))


    def clip_class(ev,p,k,c):
     xy=p['xy'];tri=p['tri'];good=p['classes']==c;ee=edges(tri);ee=ee[good[ee[:,0]]!=good[ee[:,1]]];ii=np.where(good[ee[:,0]],ee[:,0],ee[:,1]);oo=np.where(good[ee[:,0]],ee[:,1],ee[:,0]);inside=xy[ii].copy();outside=xy[oo].copy();fields=['maps','arcs','wallhit','hitell'];values={f:p[f][ii].copy() for f in fields}
     for it in range(BISECT):
      if not len(ee):break
      mid=(inside+outside)/2;obs=midpoint_observe(ev,mid,k,f'base polygon and wall classification midpoint{k} class{c} bisect{it}');valid=obs['classes']==c;inside[valid]=mid[valid];outside[~valid]=mid[~valid]
      for f in fields:values[f][valid]=obs[f][valid]
     border={tuple(e):i for i,e in enumerate(ee)};out={f:[] for f in ['xy']+fields};parents=[]
     for ti,ids in enumerate(tri):
      poly={f:[] for f in out}
      for a,b in zip(ids,np.roll(ids,-1)):
       if good[a]:
        poly['xy'].append(xy[a])
        for f in fields:poly[f].append(p[f][a])
       if good[a]!=good[b]:
        j=border[tuple(sorted([a,b]))];poly['xy'].append(inside[j])
        for f in fields:poly[f].append(values[f][j])
      for j in range(1,len(poly['xy'])-1):
       take=[0,j,j+1];v=np.array(poly['xy'])[take]
       if gu.cross(v[1]-v[0],v[2]-v[0])<=2e-18:continue
       for f in out:out[f].append(np.array(poly[f])[take])
       parents.append(ti)
     shapes={'xy':(3,2),'maps':(3,2,2),'arcs':(3,2),'wallhit':(3,2,3),'hitell':(3,2)}
     return {**{f:np.asarray(v).reshape((-1,)+shapes[f]) for f,v in out.items()},'parent':np.array(parents),'boundary_edges':len(ee)}


    def clip_plane_seeded_wall(ev,mesh,k,sg,p):
     xy=mesh['plane_xy'];tri=mesh['plane_tri'];ow=mesh['plane_owner'];good=p['hit'];ee=edges(tri);ee=ee[good[ee[:,0]]!=good[ee[:,1]]];ii=np.where(good[ee[:,0]],ee[:,0],ee[:,1]);oo=np.where(good[ee[:,0]],ee[:,1],ee[:,0]);inside=xy[ii].copy();outside=xy[oo].copy();wh=p['wall'][ii].copy();ell=p['wall_length'][ii].copy()
     budget=dict(plane=k,side=sg,boundary_edges=len(ee),maximum_additional_bisection_rays=WALL_BISECT*len(ee),cap=book()['cap']);dump(OUT/f'checkpoints/{PLANE_WALL_STEM}_projection_{k}_{sg}.json',budget)
     check_projection_budget(budget['maximum_additional_bisection_rays'])
     for it in range(WALL_BISECT):
      if not len(ee):break
      mid=(inside+outside)/2;actual,_=transform(mid,'plane');z=trace_cached(ev,seeds(actual,(k+.5)*HOP),sg*.5*HOP,f'base polygon and wall classification plane_seeded_wall_rows plane{k} side{sg} bisect{it}');hit=z['hit'];inside[hit]=mid[hit];outside[~hit]=mid[~hit];wh[hit]=z['wall'][hit];ell[hit]=z['wall_length'][hit]
     border={tuple(e):i for i,e in enumerate(ee)};fans=[];walls=[];ells=[];rows=[];owners=[];parents=[]
     for ti,ids in enumerate(tri):
      pts=[];ww=[];ll=[]
      for a,b in zip(ids,np.roll(ids,-1)):
       if good[a]:pts.append(xy[a]);ww.append(p['wall'][a]);ll.append(p['wall_length'][a])
       if good[a]!=good[b]:j=border[tuple(sorted([a,b]))];pts.append(inside[j]);ww.append(wh[j]);ll.append(ell[j])
      if len(pts)<3:continue
      row=len(owners);owners.append(ow[ti]);parents.append(ti)
      for j in range(1,len(pts)-1):
       take=[0,j,j+1];fans.append(np.asarray(pts)[take]);walls.append(np.asarray(ww)[take]);ells.append(np.asarray(ll)[take]);rows.append(row)
     return dict(fans=np.asarray(fans).reshape(-1,3,2),wallhit=np.asarray(walls).reshape(-1,3,3),hitell=np.asarray(ells).reshape(-1,3),fanrow=np.array(rows,dtype=int),owner=np.array(owners,dtype=int),parent=np.array(parents,dtype=int),boundary_edges=len(ee))


    def midpoint_seeds(ref,k):
     xy,_=transform(ref,'half');return seeds(xy,(k+1)*HOP)

    def plane_seeds(mesh,k):
     xy,_=transform(mesh['xy'],'plane');return seeds(xy,(k+.5)*HOP)

    return {'midpoint_observe': midpoint_observe, 'clip_class': clip_class, 'clip_plane_seeded_wall': clip_plane_seeded_wall, 'midpoint_seeds': midpoint_seeds, 'plane_seeds': plane_seeds}
