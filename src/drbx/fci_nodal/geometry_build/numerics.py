import numpy as np
from ..operators.parallel.geometry_io import Layout
from ..operators.parallel.weights import PointCubic, basis
from .clipping import disk, tiles

def source_plan(ns,k,order,cuts=None):
 q,w,owner=ns['source_rule'](ns['build']()[1],order)
 if cuts is not None:
  select=np.isin(owner,cuts);q=q[select];w=w[select];owner=owner[select]
 return dict(seed=np.column_stack([q,np.full(len(q),(k+.5)*ns['HOP'])]),weight=w,owner=owner)

def integrate(ns,ev,rays,plan,sign,order,bounds):
 z=plan;lo,hi=bounds;seed=np.asarray(z['seed'][lo:hi]);ww=z['weight'][lo:hi];owner=z['owner'][lo:hi];hop=ns['HOP'];trace=rays.read(seed,sign*hop,'source '+str(order))
 extent=np.where(trace['hit'],abs((trace['wall'][:,2]-seed[:,2]+np.pi)%(2*np.pi)-np.pi)/hop,.5)
 mm=ns['metric'](ev,seed,False);ans=ns['integrate'](ev,trace,extent,ww*mm['J']*abs(mm['B'][:,2]),8)
 x=seed[:,0]*np.cos(seed[:,1]);y=seed[:,0]*np.sin(seed[:,1]);mon=np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y])
 out=dict(seed=seed,source_weight=ww,owner=owner,extent=extent,hit=trace['hit'],wall=trace['wall'],wall_length=trace['wall_length'],label_moments=ans['volume'][:,None]*mon,**ans)
 return out

def observe(ns,rays,k,angular_subdivisions,clip_bisections):
 N=ns['N'];hop=ns['HOP'];xy,_,tris=disk(N,angular_subdivisions*N);quads,owner,bb0,bb1=tiles(ns['build']()[1],N,angular_subdivisions)
 good,maps,arcs=rays.observe(xy,k,hop);edges=np.unique(np.sort(np.concatenate([tris[:,[0,1]],tris[:,[1,2]],tris[:,[2,0]]]),axis=1),axis=0);edges=edges[good[edges[:,0]]!=good[edges[:,1]]];inside=np.where(good[edges[:,0],None],xy[edges[:,0]],xy[edges[:,1]]);outside=np.where(good[edges[:,0],None],xy[edges[:,1]],xy[edges[:,0]]);ii=np.where(good[edges[:,0]],edges[:,0],edges[:,1]);im=maps[ii].copy();ia=arcs[ii].copy()
 for it in range(clip_bisections):
  mid=(inside+outside)/2;valid,mm,aa=rays.observe(mid,k,hop);inside[valid]=mid[valid];im[valid]=mm[valid];ia[valid]=aa[valid];outside[~valid]=mid[~valid]
 return dict(xy=xy,tris=tris,good=good,maps=maps,arcs=arcs,edges=edges,inside=inside,im=im,ia=ia,quad_vertices=np.concatenate(quads),quad_indptr=np.r_[0,np.cumsum([len(q) for q in quads])],owner=owner,bb0=bb0,bb1=bb1)

def metric_task(ns,ev,k,z):
 r=z['records'];q=np.c_[np.linalg.norm(r[:,2:4],axis=1),np.arctan2(r[:,3],r[:,2])%(2*np.pi),np.full(len(r),(k+1)*ns['HOP'])]
 if len(q):
  B,mag,A,pos=ev(q)
  mu=z['areas']*np.linalg.det(A)/q[:,0]*abs(B[:,2]/mag)
  # Only configuration-dependent cells are retained; evaluator call shapes,
  # point arithmetic, locator predicates and canonical leaves are unchanged.
  cells=ns.get('_metric_cells')
  if cells is None:
   cells=ns['build']()[1];cells.setflags(write=False);ns['_metric_cells']=cells
  sourcecell=ns['locate'](q,cells)
 else:mu=np.empty(0);sourcecell=np.empty(0,np.int64)
 out=dict(owners=r[:,:2].astype(int),point=q,foot_left=r[:,4:6],foot_right=r[:,6:8],ell=r[:,8],area=z['areas'],mu=mu,source_triangle=z['source_triangle'],source_cell=sourcecell,source_quad_pair=z['record_key'],polygon_topology=z['polygon_topology'],clip_counts=z['clip_counts'])
 return out

def fit_task(m,keys):
 from ..operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout
 n=m['N'];p=m['replication_periods'];l=build_zernike_core_ring_layout(n,n_eta=n);I=l.P;layout=Layout(n,n,n*I,np.arange(n+1,dtype=np.int64)*I,n//p,np.asarray(l.node_u),np.asarray(l.node_theta),np.asarray(l.node_ring))
 rec=PointCubic(layout,np.ones((n//p,I),bool),m['support_widths'],m['expand_widths']);rec.rule=m['config'].get('reconstruction_rule','expand_fit_support');keys=np.asarray(keys,np.int64);cols=[];coeff=[];vt=[];centers=[];nc=[]
 for key in keys:
  ids,C=rec.fit(int(key));cols.append(ids);coeff.append(C.ravel());center=rec.xy[int(key)%I];centers.append(center);vt.append(basis((rec.xy[ids[1:]]-center)*n).ravel());nc.append(len(ids))
 ptr=np.r_[0,np.cumsum(nc)];cp=np.r_[0,np.cumsum([(x-1)*9 for x in nc])]
 return dict(keys=keys,ncol=np.asarray(nc,np.int64),indptr=ptr,ids=np.concatenate(cols),coefficient_ptr=cp,C=np.concatenate(coeff),Vt=np.concatenate(vt),center=np.asarray(centers))
