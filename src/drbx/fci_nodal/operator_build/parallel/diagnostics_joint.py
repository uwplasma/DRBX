"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import json


def bind(context, repaired_tiling_audit_stats):
    N = context.N
    NP = context.NP
    COPIES = context.copies
    ROOT = context.root
    OUT = context.out
    gu = context.gu
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log

    def in_triangles(points,tri):
     points=np.asarray(points).reshape(-1,2);out=np.zeros(len(points),bool)
     for t in tri:
      det=gu.cross(t[1]-t[0],t[2]-t[0])
      if abs(det)<=1e-18:continue
      if det<0:t=t[[0,2,1]]
      ii=np.flatnonzero(~out&np.all(points>=t.min(0)-2e-12,axis=1)&np.all(points<=t.max(0)+2e-12,axis=1))
      if len(ii):out[ii]=np.all(np.stack([gu.cross(b-a,points[ii]-a)>=-2e-13 for a,b in zip(t,np.roll(t,-1,axis=0))]),axis=0)
     return out


    def main():
     source=ROOT/('base_polygon');base=load(source/'arrays/polygon_flux_residuals.npz');phi=base['phi'][:NP];scope=base['scope'][:NP];cells=gu.build()[1];P=gu.P;geo=load(OUT/f'arrays/polygon_geometry_N{N}.npz');a=load(OUT/f'arrays/wall_rows_N{N}.npz');fp=load(OUT/f'arrays/face_flux_N{N}.npz')['psi'];face=np.zeros((N,P,2));midpoint_wall_flux=np.zeros_like(face);plane_seeded_wall_flux=np.zeros_like(face)
     for s,j in [(1,0),(0,1)]:np.add.at(face.reshape(-1,2),(geo['faces'][:,j].astype(int),s),fp)
     for kind,out in [(1,midpoint_wall_flux),(2,plane_seeded_wall_flux)]:
      mask=a['kind']==kind;np.add.at(out.reshape(-1,2),(a['wall'][mask,0].astype(int),((a['side'][mask]+1)//2).astype(int)),a['psi'][mask])
     face=face[:NP];midpoint_wall_flux=midpoint_wall_flux[:NP];plane_seeded_wall_flux=plane_seeded_wall_flux[:NP];assert np.allclose(plane_seeded_wall_flux,base['plane_seeded_wall_rows'][:NP],atol=1e-15,rtol=1e-12);comp=phi[:,:,None]-face-midpoint_wall_flux;delta=comp-plane_seeded_wall_flux;rd=delta/phi[:,:,None];has_plane_seeded_wall=plane_seeded_wall_flux>0;negative=scope&(comp<0);floor=np.minimum(phi[:,:,None]-face,0)/phi[:,:,None]
     groups=dict(global_=np.ones(P,bool),core=cells[:,0]<gu.build()[0].blocks[0].R_c-1e-12,interface=(cells[:,0]>=gu.build()[0].blocks[0].R_c-1e-12)&(cells[:,0]<gu.build()[0].blocks[0].R_c+3/N-1e-12),interior=(cells[:,0]>=gu.build()[0].blocks[0].R_c+3/N-1e-12)&(cells[:,0]<(N-4)/N-1e-12),outer_three=(cells[:,0]>=(N-4)/N-1e-12)&(cells[:,1]<1-1e-12),wall_ring=cells[:,1]==1);groups['global']=groups.pop('global_');rows=[]
     for region,reg in groups.items():
      for s,label in enumerate(['backward','forward']):
       for group in ['all','with_plane_seeded_wall_rows','without_plane_seeded_wall_rows']:
        mask=scope[:,:,s]&reg[None]
        if group=='with_plane_seeded_wall_rows':mask&=has_plane_seeded_wall[:,:,s]
        elif group=='without_plane_seeded_wall_rows':mask&=~has_plane_seeded_wall[:,:,s]
        rows.append(dict(N=N,eta_planes=N,BC='no PDE BC',measurement='static geometry',region=region,side=label,group=group,cells_full=COPIES*int(mask.sum()),residual=repaired_tiling_audit_stats(rd[:,:,s][mask]),prior=repaired_tiling_audit_stats(base['delta_over_phi'][:NP,:,s][mask]),face_only_floor=repaired_tiling_audit_stats(floor[:,:,s][mask])))
     save(OUT/f'arrays/delta_N{N}.npz',phi=phi,face=face,midpoint_wall_rows=midpoint_wall_flux,plane_seeded_wall_rows=plane_seeded_wall_flux,delta=delta,delta_over_phi=rd,scope=scope,with_plane_seeded_wall_rows=has_plane_seeded_wall,negative_complement=negative,complement=comp,face_overcoverage_floor=floor,**{'G_'+k:v for k,v in groups.items()})
     dump(OUT/f'balance_N{N}.json',dict(complete=True,N=N,metric='Unweighted cell RMS of signed delta/Phi; full-torus counts, original complete geometric scope retained; merge preserves every owner-side wall flux.',rows=rows,negative_complements_full=COPIES*int(negative.sum()),negative_rows=[dict(plane=int(k),cell=int(j),side=int(2*s-1),complement=float(comp[k,j,s]),delta_over_phi=float(rd[k,j,s])) for k,j,s in np.argwhere(negative)],face_floor_definition='min(Phi-G,0)/Phi',new_rays=0))
     cov=json.loads((source/'coverage.json').read_text());total=float(a['psi'][a['kind']<3].sum());gap=cov['reference']-total;account={};support=[]
     refsum=0.
     for k in range(NP):
      z=load(ROOT/f'base_polygon/arrays/wall_reference_{k}.npz');old=load(ROOT/f'base_polygon/arrays/reference_support_{k}.npz');ids=np.flatnonzero(~z['hit']);q=z['endpoint'][ids];xy=q[:,0,None]*np.c_[np.cos(q[:,1]),np.sin(q[:,1])];cat=old['category'].astype('<U64');eligible=old['plane_seeded_wall_rows_eligible']
      for sg,name,s in [(1,REACH_CLASSES[1],0),(-1,REACH_CLASSES[2],1)]:
       choose=np.flatnonzero((-z['sign'][ids]==sg)&~eligible&(cat!='outside_complete_scope'));p=load(OUT/f'arrays/{REACH_OUTPUT_STEMS[name][1]}_N{N}_{k}.npz');covered=in_triangles(xy[choose],p['maps'][:,:,s]);cat[choose]='supported';cat[choose[~covered]]='midpoint_wall_rows_mapping_support_miss'
      sf=z['source_flux'][ids];refsum+=COPIES*float(sf.sum())
      for c in np.unique(cat):account[c]=account.get(c,0)+COPIES*float(sf[cat==c].sum())
      save(OUT/f'arrays/support_N{N}_{k}.npz',category=cat,flux=sf)
     supported=account.pop('supported',0);account['supported_reference_minus_polygon']=supported-total;account['surface_minus_slivers_minus_reaching_reference']=cov['reference']-refsum
     assert abs(sum(account.values())-gap)<1e-12,(account,gap)
     dump(OUT/f'coverage_N{N}.json',dict(complete=True,N=N,eta_planes=N,measurement='static magnetic-flux geometry; no PDE BC',reference=cov['reference'],physical=total,gap=gap,gap_fraction=gap/cov['reference'],remaining_accounting=account,limitation='Disjoint cached-ray support accounting, not causal attribution. Signed supported remainder retains quadrature and source-to-image flux differences. Original wall reference reused unchanged.',new_rays=0));log(f'N{N} diagnostics: coverage gap {gap:.8g} ({gap/cov["reference"]:.3%}); negative complements {COPIES*int(negative.sum())}; no rays')


    return {'in_triangles': in_triangles, 'main': main}
