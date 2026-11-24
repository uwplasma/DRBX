"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import MIDPOINT_WALL_CLASSES, PLANE_WALL_STEM, REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import json
import collections
import time


def bind(context):
    N = context.N
    NP = context.NP
    TAU = context.TAU
    HOP = context.HOP
    RINGS = context.RINGS
    COPIES = context.copies
    PERIOD_ANGLE = context.period_angle
    DISK_ANGLES = context.disk_angles
    OUT = context.out
    gu = context.gu
    book = context.book
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    sectors = context.sectors
    cut = context.cut
    point_in_segments = context.point_in_segments
    segment_in_cell = context.segment_in_cell
    in_triangles = context.in_triangles
    stats = context.stats

    def inverse(xy,kind):
     r=np.linalg.norm(xy,axis=1);th=np.arctan2(xy[:,1],xy[:,0])%TAU;aa=sectors(kind);i=np.clip(np.searchsorted(aa,th,side='right')-1,0,len(aa)-2);a=aa[i];b=aa[i+1];c=np.cos((b-a)/2)/np.cos(th-(a+b)/2);A=N/c-(N-1);rp=(r-(N-1)*(c-1))/A;rp[r<(N-1)/N*c]=r[r<(N-1)/N*c];return xy*(rp/r)[:,None]


    def locate(xy):
     cells=gu.build()[1];r=np.linalg.norm(xy,axis=1);th=np.arctan2(xy[:,1],xy[:,0]);owner=np.full(len(r),-1,int)
     for j,(r0,r1,a,b) in enumerate(cells):
      dt=(th-a)%TAU;mask=(owner<0)&(r>=r0-2e-12)&(r<=r1+2e-12)&(dt<=b-a+2e-12);owner[mask]=j
     assert np.all(owner>=0),('outside cells',xy[owner<0]);return owner


    def main():
     start=time.time();cells=gu.build()[1];P=gu.P;phi=np.array([load(OUT/f'arrays/phi_{k}.npz')['phi'] for k in range(NP)]);face=np.zeros((NP,P,2));midpoint_wall_flux=np.zeros_like(face);plane_seeded_wall_flux=np.zeros_like(face);hasface=np.zeros_like(face,bool);walls=[];faces=[];wflux=[];fflux=[];sides=[];kinds=[];inter=[];units=[]
     for k in range(NP):
      a=load(OUT/f'arrays/geometry_{k}.npz');f=a['faces'];fw=a['face_flux'];w=a['wall'];wp=a['wall_flux'];s=a['wall_side'];kind=a['wall_kind'];walls.append(w);faces.append(f);wflux.append(wp);fflux.append(fw);sides.append(s);kinds.append(kind);inter.extend(json.loads((OUT/f'checkpoints/interpolation_{k}.json').read_text()));units.append(json.loads((OUT/f'checkpoints/quadrature_{k}.json').read_text()))
      for si,j in [(1,0),(0,1)]:
       ow=f[:,j].astype(int)%(NP*P);np.add.at(face.reshape(-1,2),(ow,si),fw);hasface.reshape(-1,2)[ow,si]=True
      for typ,out in [(1,midpoint_wall_flux),(2,plane_seeded_wall_flux)]:
       take=kind==typ;np.add.at(out.reshape(-1,2),(w[take,0].astype(int)%(NP*P),((s[take]+1)//2).astype(int)),wp[take])
     wall=np.vstack(walls);wp=np.concatenate(wflux);side=np.concatenate(sides);kind=np.concatenate(kinds);f=np.vstack(faces);fp=np.concatenate(fflux)
     # Geometric completeness against the artificial inner boundary, plus the boundary one ring inward of it.
     certificates={};scopes={};required=np.zeros_like(hasface);cut=(midpoint_wall_flux+plane_seeded_wall_flux)>0;required|=cut
     refs=[]
     for k in range(NP):
      z=load(OUT/f'arrays/wall_reference_{k}.npz');ids=np.flatnonzero(~z['hit']);q=z['endpoint'][ids];xy=q[:,0,None]*np.c_[np.cos(q[:,1]),np.sin(q[:,1])];ow=locate(xy);plane=(k+(z['sign'][ids]>0).astype(int))%NP;si=(z['sign'][ids]<0).astype(int);required[plane,ow,si]=True;refs.append((k,z,ids,ow,plane,si,xy))
     required&=hasface.any(2)[:,:,None]
     for rings in [RINGS-1,RINGS]:
      scope=scope_candidates(hasface);allgood=True
      level=RINGS-rings
      for k in range(NP):
       z=load(OUT/f'checkpoints/midpoint_trace_vertices_{k}.npz');ids=level*DISK_ANGLES+np.arange(DISK_ANGLES);allgood&=bool(np.all(z['classes'][ids]==0));edge=np.c_[ids,np.roll(ids,-1)]
       for reach in [0,1]:
        plane=(k+reach)%NP;si=1-reach;image=z['maps'][edge,reach]
        aa=image[:,0];dd=image[:,1]-aa;tt=np.clip(-np.sum(aa*dd,axis=1)/np.sum(dd*dd,axis=1),0,1);emin=np.linalg.norm(aa+tt[:,None]*dd,axis=1);emax=np.linalg.norm(image,axis=2).max(1)
        for j,c in enumerate(cells):
         if not scope[plane,j,si]:continue
         r=(c[0]+c[1])/2;t=(c[2]+c[3])/2;x=r*np.array([np.cos(t),np.sin(t)])
         scope[plane,j,si]=not point_in_segments(x,image) and not any(segment_in_cell(e,c) for e in image[(emax>=c[0]-2e-12)&(emin<=c[1]+2e-12)])
      missing=required&~scope;certificates[str(rings)]=dict(rings=rings,artificial_boundary_all_G=allgood,required_cut_cellsides=int(required.sum()),incomplete_required=int(missing.sum()),missing_indices=np.argwhere(missing).tolist(),scope_cellsides=int(scope.sum()));scopes[rings]=scope
     assert certificates[str(RINGS)]['artificial_boundary_all_G'] and certificates[str(RINGS)]['incomplete_required']==0,certificates
     assert certificates[str(RINGS-1)]['artificial_boundary_all_G'] and certificates[str(RINGS-1)]['incomplete_required']==0,('one-ring margin not certified',certificates)
     scope=scopes[RINGS];comp=phi[:,:,None]-face-midpoint_wall_flux;delta=comp-plane_seeded_wall_flux;rd=delta/phi[:,:,None];has_plane_seeded_wall=plane_seeded_wall_flux>0;scale=np.full_like(plane_seeded_wall_flux,np.nan);scale[has_plane_seeded_wall]=comp[has_plane_seeded_wall]/plane_seeded_wall_flux[has_plane_seeded_wall];negative=scope&(comp<0);unresolved=scope&~has_plane_seeded_wall&(rd>.01);floor=np.minimum(phi[:,:,None]-face,0)/phi[:,:,None]
     groups=dict(global_=np.ones(P,bool),core=cells[:,0]<gu.build()[0].blocks[0].R_c-1e-12,interface=(cells[:,0]>=gu.build()[0].blocks[0].R_c-1e-12)&(cells[:,0]<gu.build()[0].blocks[0].R_c+3/N-1e-12),interior=(cells[:,0]>=gu.build()[0].blocks[0].R_c+3/N-1e-12)&(cells[:,0]<(N-4)/N-1e-12),outer_three=(cells[:,0]>=(N-4)/N-1e-12)&(cells[:,1]<1-1e-12),wall_ring=cells[:,1]==1);groups['global']=groups.pop('global_');rows=[];scalerows=[]
     for region,reg in groups.items():
      for si,label in enumerate(['backward','forward']):
       for group in ['all','with_plane_seeded_wall_rows','without_plane_seeded_wall_rows']:
        mask=scope[:,:,si]&reg[None]
        if group=='with_plane_seeded_wall_rows':mask&=has_plane_seeded_wall[:,:,si]
        elif group=='without_plane_seeded_wall_rows':mask&=~has_plane_seeded_wall[:,:,si]
        rows.append(dict(region=region,side=label,group=group,cells_full=COPIES*int(mask.sum()),N32=stats(rd[:,:,si][mask]),face_only_floor=stats(floor[:,:,si][mask]),face_only_no_wall=stats(((phi[:,:,None]-face)/phi[:,:,None])[:,:,si][mask])))
       scalerows.append(dict(region=region,side=label,s=stats(scale[:,:,si][scope[:,:,si]&reg[None]&has_plane_seeded_wall[:,:,si]])))
     save(OUT/'arrays/polygon_flux_residuals.npz',phi=phi,face=face,midpoint_wall_rows=midpoint_wall_flux,plane_seeded_wall_rows=plane_seeded_wall_flux,delta=delta,delta_over_phi=rd,scope=scope,with_plane_seeded_wall_rows=has_plane_seeded_wall,scale=scale,negative_complement=negative,unresolved=unresolved,face_overcoverage_floor=floor,**{'G_'+k:v for k,v in groups.items()})
     save(OUT/'arrays/geometry_base_period.npz',faces=f,face_flux=fp,wall=wall,wall_flux=wp,wall_side=side,wall_kind=kind,cells=cells)
     candidate=scope_candidates(hasface)
     excluded=candidate&~scope
     dump(OUT/'scope.json',dict(complete=True,band_rings=RINGS,certified_without_margin=RINGS-1,margin_rings=1,certificates=certificates,excluded_inner_edge_first_period=int(excluded.sum()),excluded_inner_edge_full_torus=COPIES*int(excluded.sum()),excluded_inner_edge_indices=np.argwhere(excluded).tolist(),index_columns=['plane_in_first_period','cell','side_0_backward_1_forward'],no_face_cells_outside_flux_box=int((~hasface.any(2)).sum()),limitation='Finite affine mesh and observed wall quadrature; unsampled sub-grid hit islands are not certified.'))
     dump(OUT/'balance.json',dict(complete=True,rows=rows,scales=scalerows,negative_complement=COPIES*int(negative.sum()),no_plane_wall_unresolved_above_1percent=COPIES*int(unresolved.sum()),unresolved_first_period=[dict(plane=int(k),cell=int(j),side=int(2*s-1),residual=float(rd[k,j,s])) for k,j,s in np.argwhere(unresolved)],scope_full=COPIES*int(scope.sum())))
     # Disjoint support accounting from fresh plane-reaching surface rays. These retain dense histories.
     accounting=collections.defaultdict(float);reference_flux=0.;surface_rows=[]
     for k,z,ids,ow,plane,si,xy in refs:
      age=abs(z['delta'][ids]);eligible=age<=.5*HOP+1e-13;covered=np.zeros(len(ids),bool)
      for sgn in [-1,1]:
       choose=np.flatnonzero((-z['sign'][ids]==sgn)&eligible)
       for j in np.unique(ow[choose]):
        take=choose[ow[choose]==j];pk=int(plane[take[0]]);p=load(OUT/f'arrays/{PLANE_WALL_STEM}_{pk}_{sgn}.npz');covered[take]=in_triangles(inverse(xy[take],'plane'),p['fans'][p['owner'][p['fanrow']]==j])
       choose=np.flatnonzero((-z['sign'][ids]==sgn)&~eligible)
       # A backward-traced wall entry reaches the left plane (F); forward reaches right (B).
       name=REACH_CLASSES[1 if sgn==1 else 2];p=load(OUT/f'arrays/{REACH_OUTPUT_STEMS[name][0]}_{k}.npz');reach=0 if sgn==1 else 1;covered[choose]=in_triangles(xy[choose],p['maps'][:,:,reach])
      cat=np.full(len(ids),'supported',dtype='<U40');cat[~covered&eligible]='plane_seeded_wall_rows_support_miss';cat[~covered&~eligible]='midpoint_wall_rows_mapping_support_miss';cat[~scope[plane,ow,si]]='outside_complete_scope';sf=z['source_flux'][ids];reference_flux+=float(sf.sum())
      for c in np.unique(cat):accounting[c]+=COPIES*float(sf[cat==c].sum())
      save(OUT/f'arrays/reference_support_{k}.npz',category=cat,flux=sf,plane=plane,owner=ow,side=2*si-1,plane_seeded_wall_rows_eligible=eligible)
      u=units[k];diff=float(phi[(k+1)%NP].sum()-phi[k].sum());u=dict(u,plane_difference=diff,slab_residual=diff-u['signed_inward_flux'],slab_residual_over_plane_flux=(diff-u['signed_inward_flux'])/float(phi[k].sum()));surface_rows.append(u)
     ref=COPIES*sum(r['physical_reference'] for r in units);physical=COPIES*float(wp.sum());gap=ref-physical;supported=accounting.pop('supported',0.);accounting['supported_reference_minus_new']=supported-physical;accounting['surface_minus_slivers_minus_reaching_reference']=ref-COPIES*reference_flux;assert abs(sum(accounting.values())-gap)<1e-12
     dump(OUT/'coverage.json',dict(complete=True,reference=ref,new=physical,gap=gap,gap_fraction=gap/ref,remaining_accounting=dict(accounting),rows=surface_rows,slab_relative_rms=float(np.sqrt(np.mean([r['slab_residual_over_plane_flux']**2 for r in surface_rows]))),limitation='Base wall q3 quadrature; remote legacy gathered geometry products anchor completion absent. Support accounting is disjoint, not a causal split. Source-vs-transported landing flux differences remain in the signed remainder.'))
     groups=[]
     for cls in (*MIDPOINT_WALL_CLASSES.values(), PLANE_WALL_STEM):
      for ring in ['all']+sorted({r['ring'] for r in inter if r['cls']==cls}):
       rr=[r for r in inter if r['cls']==cls and (ring=='all' or r['ring']==ring)];v=[r for r in rr if r['barycentre_hit']];groups.append(dict(cls=cls,ring=ring,rows=len(rr),flux=sum(r['flux'] for r in rr),validation_rows=len(v),grazing_count=sum(r['grazing_flag'] for r in rr),grazing_flux=sum(r['flux'] for r in rr if r['grazing_flag']),traced_override_count=sum(r['used_traced_barycentre'] for r in rr),**{q:stats([r[q] for r in v]) for q in ['wall_distance_cells','relative_ell_error']}))
     dump(OUT/'wall_interpolation.json',dict(complete=True,groups=groups,rows=inter,new_barycentre_rays=0,sg_side_mismatch_count_full=COPIES*sum(r['sg_side_mismatch'] for r in inter)))
     cl={name:COPIES*sum(json.loads((OUT/f'checkpoints/geometry_{k}.json').read_text())['classes'][name]['flux'] for k in range(NP)) for name in REACH_CLASSES};cl.update(midpoint_wall_rows=COPIES*float(wp[kind==1].sum()),plane_seeded_wall_rows=COPIES*float(wp[kind==2].sum()));dump(OUT/'flux_shares.json',dict(full_torus_flux=cl,midpoint_wall_rows_fraction_physical=cl['midpoint_wall_rows']/physical,plane_seeded_wall_rows_fraction_physical=cl['plane_seeded_wall_rows']/physical,neither_plane_reached_over_surface=cl[REACH_CLASSES[3]]/(COPIES*sum(r['surface_absolute_flux'] for r in units)),note='neither_plane_reached is half-plane magnetic flux; not additive to physical midpoint_wall_rows+plane_seeded_wall_rows surface legs.'))
     # Explicit full-torus geometry export, without masses, legacy gathered geometry products, operators or static assembly.
     ff=[];ww=[]
     for period in range(COPIES):
      a=f.copy();a[:,:2]=((a[:,:2].astype(int)//P+NP*period)%N)*P+a[:,:2].astype(int)%P;a[:,4]=(a[:,4]+period*PERIOD_ANGLE)%TAU;ff.append(a)
      a=wall.copy();a[:,0]=((a[:,0].astype(int)//P+NP*period)%N)*P+a[:,0].astype(int)%P;a[:,3]=(a[:,3]+period*PERIOD_ANGLE)%TAU;ww.append(a)
     fullwall=np.vstack(ww)
     save(OUT/'arrays/polygon_geometry.npz',faces=np.vstack(ff),face_flux=np.tile(fp,COPIES),wall=fullwall,wall_flux=np.tile(wp,COPIES),wall_side=np.tile(side,COPIES),wall_kind=np.tile(kind,COPIES),grazing_flag=np.tile(np.array([r['grazing_flag'] for r in inter]),COPIES),used_traced_barycentre=np.tile(np.array([r['used_traced_barycentre'] for r in inter]),COPIES),phi=np.tile(phi,(COPIES,1)),scope=np.tile(scope,(COPIES,1,1)),cells=cells,N=N,P=P)
     (OUT/'arrays/FORMAT.txt').write_text(f'polygon_geometry.npz is the full N{N} torus, {N} eta planes. faces columns: left_owner,right_owner,rho,theta,eta,left_x,left_y,right_x,right_y,ell,mu. wall columns: owner,rho,theta,eta,foot_x,foot_y,mu,evaluated_Brho_sign,ell,aux_flag. wall_side retains intended trace side. wall_kind: 1=midpoint_wall_rows, 2=plane_seeded_wall_rows. Scalar flux arrays are magnetic flux Psi=mu*|B|. No auxiliary sliver rows, legacy gathered geometry products, H, reconstruction, or operators are fabricated. phi and scope have all {N} planes. The separate polygon_flux_residuals and geometry_base_period files contain {NP} independent planes/strips; their counts/fluxes multiply by the number of symmetry copies for the torus. Slivers enter only the independently rebuilt wall-reference diagnostic.\n')
     log(f'diagnostics saved; {time.time()-start:.1f}s; ledger {book()["new_rays"]}/{book()["cap"]}; unresolved {COPIES*int(unresolved.sum())}')


    def scope_candidates(hasface):
     cells=gu.build()[1]
     return hasface.any(2)[:,:,None]&np.ones_like(hasface)&(cells[:,0]>=(N-RINGS)/N-1e-12)[None,:,None]

    return {'inverse': inverse, 'locate': locate, 'main': main, 'scope_candidates': scope_candidates}
