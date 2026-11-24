"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import json
from scipy.spatial import ConvexHull


def bind(context, remainder):
    N = context.N
    TAU = context.TAU
    HOP = context.HOP
    BISECT = context.BISECT
    ROOT = context.root
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    Lookup = context.Lookup
    book = context.book
    trace_cached = context.trace_cached
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    relative = context.relative
    key = context.key
    moment = context.moment
    BITS = context.BITS
    FIELDS = context.FIELDS
    RAY_LIMIT = context.supplemental_ray_budget
    PRIOR_LEDGERS = context.prior_ledgers

    NAMES=REACH_CLASSES


    LEGACY_REASON='Two source gaps below partition audit absolute 1e-10 area threshold, found by final relative audit. Shared boundary uses frozen saved G-side bracket coordinates; stopping geometry is the cached one-sided non-G endpoint, reaching geometry uses the cached endpoints. No G replacement or new gap_centroid_trace_audit rays.'


    RAY_PREFIX=f'supplemental polygon tiny-gap tracing N{N} '


    def supplemental_charged():
     ledgers=[book(),*PRIOR_LEDGERS]
     return sum(c['rays'] for ledger in ledgers for c in ledger['calls'] if c['label'].startswith(RAY_PREFIX))


    class SupplementalCache:
     def __init__(self):
      self.cache=Lookup();receipt=OUT/f'supplemental_tiny_gaps_N{N}.json'
      self.rays=json.loads(receipt.read_text()).get('rays',[]) if receipt.exists() else [];self.known={}
     def observation(self,ref,k,parent,it):
      q,z,tau=self.cache.observation(ref,k);z=list(z)
      for side,sg in enumerate([-1,1]):
       ky=key(q,sg*HOP)
       if z[side] is None:z[side]=self.known.get(ky)
       if z[side] is None:
        used=supplemental_charged()
        assert used<RAY_LIMIT,('supplemental tiny-gap ray cap',used,RAY_LIMIT)
        label=RAY_PREFIX+f'plane{k} parent{parent} iteration{it} side{sg} key{ky[:16]}'
        result=trace_cached(EVALUATOR,q[None],sg*HOP,label)
        z[side]=dict(hit=bool(result['hit'][0]),wall=result['wall'][0],wall_length=result['wall_length'][0]);self.known[ky]=z[side]
        calls=[c for c in book()['calls'] if c['label']==label];charged=sum(c['rays'] for c in calls)
        self.rays.append(dict(plane=k,parent=parent,iteration=it,side=sg,seed=q.tolist(),delta=sg*HOP,key=ky,label=label,new_rays=charged))
        # Durable even if later geometry validation fails. No synthetic observations.
        dump(OUT/f'supplemental_tiny_gaps_N{N}.json',dict(complete=False,N=N,new_rays=sum(r['new_rays'] for r in self.rays),rays=self.rays,ray_limit=RAY_LIMIT,ledger_count=book()['new_rays']))
      tau=[abs((a['wall'][2]-q[2]+np.pi)%TAU-np.pi)/HOP if a['hit'] else 1. for a in z]
      return q,z,tau


    def legacy_eligible(p,parent,pieces):
     cl=p['classes'][p['tri'][parent]];non=np.flatnonzero(cl!=0)
     eligible=len(non)==1 and int(cl[non[0]]) in [1,2] and len(pieces[0]['xy'])>0 and not any(len(z['xy']) for z in pieces[1:])
     if not eligible:return False
     ids=p['tri'][parent];a=p['xy'][ids[non[0]]];v=pieces[0]['xy'].reshape(-1,2)
     return all(np.min(np.linalg.norm(v-(a+(p['xy'][b]-a)/2**BISECT),axis=1))<2e-14 for b in ids[cl==0])


    def legacy_corner(p,parent,g,k,cache):
     # Arithmetic/order deliberately identical to the two executed N32 closures.
     ids=p['tri'][parent];cl=p['classes'][ids];non=np.flatnonzero(cl!=0);a=ids[non[0]];c=int(p['classes'][a]);stop=1 if c==1 else 0;reach=1-stop
     v=g['xy'].reshape(-1,2);vv={f:g[f].reshape((-1,)+g[f].shape[2:]) for f in FIELDS};gp=np.unique(v,axis=0);gp=gp[ConvexHull(gp).vertices];rr=remainder(p['xy'][ids],gp);assert len(rr)==1 and len(rr[0])==3
     bracket=[]
     for b in ids[cl==0]:
      ray_start=len(cache.rays)
      left=p['xy'][a].copy();right=p['xy'][b].copy()
      for it in range(BISECT):
       mid=(left+right)/2;q,z,tau=cache.observation(mid,k,parent,it);assert all(zz is not None for zz in z),('uncached supplemental bracket',k,parent,it);assert tau[reach]>=.5
       if tau[stop]<.5:left=mid
       else:right=mid
      assert np.array_equal(left,p['xy'][a]);assert np.min(np.linalg.norm(v-right,axis=1))<2e-14;bracket.append(dict(other_class_endpoint=left.tolist(),G_endpoint=right.tolist(),iterations=BISECT,all_samples_cached=len(cache.rays)==ray_start))
     poly=rr[0];maps=np.full((3,2,2),np.nan);arcs=np.full((3,2),np.nan);wh=np.full((3,2,3),np.nan);hl=np.full((3,2),np.nan)
     for j,x in enumerate(poly):
      if np.linalg.norm(x-p['xy'][a])<2e-14:maps[j,reach]=p['maps'][a,reach];arcs[j,reach]=p['arcs'][a,reach]
      else:
       i=np.argmin(np.linalg.norm(v-x,axis=1));assert np.linalg.norm(v[i]-x)<2e-14;maps[j,reach]=vv['maps'][i,reach];arcs[j,reach]=vv['arcs'][i,reach]
      wh[j,stop]=p['wallhit'][a,stop];hl[j,stop]=p['hitell'][a,stop]
     assert np.isfinite(wh[:,stop]).all();new=dict(xy=poly,maps=maps,arcs=arcs,wallhit=wh,hitell=hl,parent=np.array(parent),child=np.array(-2));area=moment(poly,np.eye(3),'half')[0]
     return [(c,new)],dict(plane=k,parent=parent,cls=NAMES[c],area=area,both_planes_reached_bitwise_unchanged=True,new_rays=0,refined=False,brackets=bracket)


    def signed_area(poly):
     return float(np.sum(gu.cross(poly[1:-1]-poly[0],poly[2:]-poly[0]))/2) if len(poly)>=3 else 0.


    def halfplane(poly,a,b,inside=True):
     if len(poly)<3:return np.empty((0,2))
     d=gu.cross(b-a,poly-a)*(1 if inside else -1);out=[]
     for x,y,fx,fy in zip(poly,np.roll(poly,-1,axis=0),d,np.roll(d,-1)):
      if fx>=0:out.append(x)
      if fx*fy<0:out.append(x+fx/(fx-fy)*(y-x))
     if len(out)<3:return np.empty((0,2))
     out=np.asarray(out);out=out[np.linalg.norm(out-np.roll(out,1,axis=0),axis=1)>1e-15]
     return out if len(out)>=3 and signed_area(out)>1e-16 else np.empty((0,2))


    def subtract(poly,tri):
     parts=[];inside=poly
     for a,b in zip(tri,np.roll(tri,-1,axis=0)):
      outside=halfplane(inside,a,b,False)
      if len(outside):parts.append(outside)
      inside=halfplane(inside,a,b,True)
      if not len(inside):break
     return parts


    def parent_local(xx):
     # Explicit 2x2 solve uses the widest available native scalar precision for
     # nearly parallel edges. macOS ARM longdouble is float64; saved outputs stay
     # float64 on every platform. No coordinates are snapped in this subtraction.
     xx=np.asarray(xx,dtype=np.longdouble);a=xx[1]-xx[0];b=xx[2]-xx[0];det=gu.cross(a,b)
     def local(x):
      d=np.asarray(x,dtype=np.longdouble)-xx[0]
      return np.stack([gu.cross(d,b)/det,gu.cross(a,d)/det],axis=-1)
     return local


    def local_remainders(xx,pieces):
     local=parent_local(xx)
     parts=[np.array([[0.,0.],[1.,0.],[0.,1.]],dtype=np.longdouble)]
     for z in pieces:
      for tri in z['xy']:
       clip=local(tri)
       if signed_area(clip)<0:clip=clip[::-1]
       parts=[p for old in parts for p in subtract(old,clip)]
     return parts,local


    def segment_distance(x,a,b):
     d=b-a;dd=float(d@d);t=np.clip(float((x-a)@d)/dd,0,1) if dd else 0.
     return float(np.linalg.norm(x-(a+t*d)))


    def touches(a,b):
     return any(segment_distance(x,c,d)<2e-11 for x in a for c,d in zip(b,np.roll(b,-1,axis=0))) or any(segment_distance(x,c,d)<2e-11 for x in b for c,d in zip(a,np.roll(a,-1,axis=0)))


    def inpoly(x,poly):
     return all(gu.cross(b-a,x-a)>=-2e-11*np.linalg.norm(b-a) for a,b in zip(poly,np.roll(poly,-1,axis=0)))


    def closest_bary(x,tri):
     mat=np.column_stack([tri[1]-tri[0],tri[2]-tri[0]])
     if np.linalg.det(mat)!=0:
      u=np.linalg.solve(mat,x-tri[0]);b=np.r_[1-u.sum(),u]
      if (b>=0).all():return b,0.
     best=None
     for a,b in [(0,1),(1,2),(2,0)]:
      d=tri[b]-tri[a];dd=float(d@d);t=np.clip(float((x-tri[a])@d)/dd,0,1) if dd else 0.;w=np.zeros(3);w[a]=1-t;w[b]=t;dist=float(np.linalg.norm(x-w@tri))
      if best is None or dist<best[1]:best=(w,dist)
     return best


    def side_value(x,s,bit,p,ids,pieces):
     candidates=[]
     for c,z in enumerate(pieces):
      if BITS[c,s]!=bit:continue
      for i,tri in enumerate(z['xy']):
       vals=np.c_[z['maps'][i,:,s],z['arcs'][i,:,s]] if bit else np.c_[z['wallhit'][i,:,s],z['hitell'][i,:,s]]
       assert np.isfinite(vals).all();w,dist=closest_bary(x,tri);candidates.append((dist,tri,vals,w))
     for a in ids:
      if BITS[p['classes'][a],s]!=bit:continue
      vals=np.r_[p['maps'][a,s],p['arcs'][a,s]] if bit else np.r_[p['wallhit'][a,s],p['hitell'][a,s]]
      assert np.isfinite(vals).all();candidates.append((float(np.linalg.norm(x-p['xy'][a])),p['xy'][a:a+1],vals[None],np.ones(1)))
     assert candidates,('no compatible one-sided source',s,bool(bit))
     _,tri,vals,w=min(candidates,key=lambda r:r[0]);vals=vals.copy()
     if not bit:vals[:,1:3]=vals[0,1:3]+(vals[:,1:3]-vals[0,1:3]+np.pi)%TAU-np.pi
     value=w@vals
     if not bit:value[1:3]%=TAU
     return value


    def component_polygons(xx,parts,group,pieces):
     # Reuse saved coordinates for roundoff-equivalent boundary vertices; this
     # avoids artificial strips from repeated subtraction of a shared diagonal.
     anchors=np.vstack([xx]+[z['xy'].reshape(-1,2) for z in pieces if len(z['xy'])]);polygons=[]
     for i in group:
      poly=np.asarray(xx[0]+parts[i]@np.array([xx[1]-xx[0],xx[2]-xx[0]],dtype=np.longdouble),dtype=xx.dtype)
      for j,x in enumerate(poly):
       d=np.linalg.norm(anchors-x,axis=1);nearest=np.argmin(d)
       if d[nearest]<2e-14:poly[j]=anchors[nearest]
      polygons.append(poly)
     vertices=np.unique(np.vstack(polygons),axis=0);local=parent_local(xx)
     if len(vertices)>=3:
      hull=vertices[ConvexHull(np.asarray(local(vertices),dtype=float)).vertices]
      difference=abs(signed_area(local(hull))-sum(signed_area(local(t)) for t in polygons))
      if difference<.5e-12:return [hull]
     return polygons


    def general_closure(p,parent,pieces,k):
     ids=p['tri'][parent];xx=p['xy'][ids];parts,local=local_remainders(xx,pieces);groups=[];unseen=set(range(len(parts)))
     while unseen:
      group=[min(unseen)];unseen.remove(group[0]);pending=group.copy()
      while pending:
       i=pending.pop()
       for j in sorted(unseen.copy()):
        if touches(parts[i],parts[j]):unseen.remove(j);group.append(j);pending.append(j)
      groups.append(sorted(group))
     appended=[];decisions=[];areas=np.zeros(4)
     for group in groups:
      candidates=[]
      for c in [1,2,3]:
       near_piece=any(touches(parts[i],local(t)) for i in group for t in pieces[c]['xy'])
       near_vertex=any(inpoly(local(p['xy'][a:a+1])[0],parts[i]) for i in group for a in ids if p['classes'][a]==c)
       if near_piece or near_vertex:candidates.append(c)
      assert candidates,('no adjacent non-G class in qualified remainder',k,parent,p['classes'][ids],group)
      c=min(candidates,key=lambda c:(int(BITS[c].sum()),c));component_area=0.
      for poly in component_polygons(xx,parts,group,pieces):
       for j in range(1,len(poly)-1):
        tri=poly[[0,j,j+1]]
        if gu.cross(tri[1]-tri[0],tri[2]-tri[0])<=0:continue
        values=dict(xy=tri,maps=np.full((3,2,2),np.nan),arcs=np.full((3,2),np.nan),wallhit=np.full((3,2,3),np.nan),hitell=np.full((3,2),np.nan),parent=np.array(parent),child=np.array(-2))
        for s,bit in enumerate(BITS[c]):
         v=np.array([side_value(x,s,bit,p,ids,pieces) for x in tri])
         if bit:values['maps'][:,s]=v[:,:2];values['arcs'][:,s]=v[:,2]
         else:values['wallhit'][:,s]=v[:,:3];values['hitell'][:,s]=v[:,3]
        a=moment(tri,np.eye(3),'half')[0];areas[c]+=a;component_area+=a;appended.append((c,values))
      decisions.append(dict(adjacent_classes=[NAMES[i] for i in candidates],assigned_class=NAMES[c],area=component_area,remainder_polygons=len(group)))
     assert appended,('no positive closing sliver',k,parent)
     row=dict(plane=k,parent=parent,classes=p['classes'][ids].tolist(),rule='All-piece subtraction; connected remainder takes adjacent class with most stopping directions, tie F before B; nearest compatible saved one-sided extension',sliver_area_per_class=dict(zip(NAMES,areas.tolist())),components=decisions,both_planes_reached_bitwise_unchanged=True,all_existing_pieces_bitwise=True,new_rays=0,refined=False,brackets=[])
     return appended,row


    def validate_parent(p,parent,pieces,appended):
     ids=p['tri'][parent];xx=p['xy'][ids];area=moment(xx,np.eye(3),'half')[0];alltri=[t for z in pieces for t in z['xy']]+[z['xy'] for c,z in appended]
     total=sum(moment(t,np.eye(3),'half')[0] for t in alltri);relative=abs(total-area)/area
     assert relative<=1e-12,('tiny-gap tiling',parent,relative)
     # Check actual intersections of closing pieces with each other and ALL saved
     # pieces. Existing-existing overlaps are checked too, not inferred from sums.
     to_local=parent_local(xx);local=[to_local(t) for t in alltri];overlap=0.
     for i,a in enumerate(local):
      for b in local[i+1:]:
       poly=a
       if signed_area(b)<0:b=b[::-1]
       for x,y in zip(b,np.roll(b,-1,axis=0)):poly=halfplane(poly,x,y)
       overlap+=max(signed_area(poly),0.)
     relative_overlap=overlap/.5
     assert relative_overlap<=1e-12,('tiny-gap overlap',parent,relative_overlap)
     return dict(parent_area=area,closed_area=total,relative_gap=relative,relative_overlap=relative_overlap)


    def main():
     rows=[];audit=json.loads((OUT/f'tiling_N{N}.json').read_text());remaining=audit['remaining']
     receipt=OUT/f'supplemental_tiny_gaps_N{N}.json'
     if not remaining and receipt.exists() and json.loads(receipt.read_text()).get('complete'):
      log(f'N{N} no remaining tiny gaps; retain completed supplemental parent/ray receipt for downstream geometry');return
     cache=SupplementalCache()
     assert all(not r['changed'] and 0<r['gap_area']<1e-10 for r in remaining),('Unqualified residual topology',remaining)
     assert len({(r['plane'],r['parent']) for r in remaining})==len(remaining)
     plans=[];checks=[]
     for r in remaining:
      k,parent=r['plane'],r['parent'];p=load(ROOT/f'base_polygon/checkpoints/midpoint_trace_vertices_{k}.npz');pieces=[]
      for name in NAMES:
       z=load(OUT/f'arrays/{REACH_OUTPUT_STEMS[name][1]}_N{N}_{k}.npz');pieces.append({f:v[z['parent']==parent] for f,v in z.items()})
      if legacy_eligible(p,parent,pieces):appended,row=legacy_corner(p,parent,pieces[0],k,cache)
      else:appended,row=general_closure(p,parent,pieces,k)
      row['new_rays']=sum(r['new_rays'] for r in cache.rays if r['plane']==k and r['parent']==parent)
      checks.append(dict(plane=k,parent=parent,**validate_parent(p,parent,pieces,appended)));plans.append((k,parent,appended));rows.append(row)
     # Validate every plan before changing any class file. All both_planes_reached files stay exact.
     for k in dict.fromkeys(k for k,parent,appended in plans):
      for c in [1,2,3]:
       additions=[z for kk,parent,aa in plans if kk==k for cc,z in aa if cc==c]
       if not additions:continue
       path=OUT/f'arrays/{REACH_OUTPUT_STEMS[NAMES[c]][1]}_N{N}_{k}.npz';old=load(path);z={f:np.concatenate([v,np.array([a[f] for a in additions])]) for f,v in old.items()}
       assert all(z[f].dtype==v.dtype and np.array_equal(z[f][:len(v)],v,equal_nan=True) for f,v in old.items())
       save(path,**z)
     new_rays=sum(r['new_rays'] for r in cache.rays)
     legacy=(N==32 and [(r['plane'],r['parent']) for r in rows]==[(1,2775),(5,2599)] and all('cls' in r for r in rows) and new_rays==0)
     result=dict(complete=True,N=N,parents=rows,new_rays=new_rays,reason=LEGACY_REASON if legacy else 'Qualified sub-1e-10 source remainders appended using frozen G/all-piece geometry and deterministic adjacent-class one-sided extension. Legacy one-F/B-corner brackets are cache-observed; missing observations use the charged ledger, at most2000 new rays per N.')
     if not legacy:result.update(tiling_checks=checks,rays=cache.rays,ray_limit=RAY_LIMIT,supplemental_ledger_charges=supplemental_charged(),N64_difference='None: same large-N paths and BISECT10; NP and HOP follow N.')
     dump(OUT/f'supplemental_tiny_gaps_N{N}.json',result)
     log(f'N{N} {len(rows)} qualified tiny gaps closed; every saved piece/G bitwise, supplemental new rays{new_rays}/{RAY_LIMIT}')


    return {'supplemental_charged': supplemental_charged, 'SupplementalCache': SupplementalCache, 'legacy_eligible': legacy_eligible, 'legacy_corner': legacy_corner, 'signed_area': signed_area, 'halfplane': halfplane, 'subtract': subtract, 'parent_local': parent_local, 'local_remainders': local_remainders, 'segment_distance': segment_distance, 'touches': touches, 'inpoly': inpoly, 'closest_bary': closest_bary, 'side_value': side_value, 'component_polygons': component_polygons, 'general_closure': general_closure, 'validate_parent': validate_parent, 'main': main}
