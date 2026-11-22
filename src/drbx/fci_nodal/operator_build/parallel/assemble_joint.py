"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
import numpy as np


def bind(context):
    N = context.N
    NP = context.NP
    TAU = context.TAU
    COPIES = context.copies
    PERIOD_ANGLE = context.period_angle
    ROOT = context.root
    OUT = context.out
    gu = context.gu
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log

    def expand(a,face=False):
     out=[]
     for per in range(COPIES):
      b=a.copy()
      if face:b[:,:2]=((b[:,:2].astype(int)//gu.P+NP*per)%N)*gu.P+b[:,:2].astype(int)%gu.P;b[:,4]=(b[:,4]+per*PERIOD_ANGLE)%TAU
      else:b[:,0]=((b[:,0].astype(int)//gu.P+NP*per)%N)*gu.P+b[:,0].astype(int)%gu.P;b[:,3]=(b[:,3]+per*PERIOD_ANGLE)%TAU
      out.append(b)
     return np.vstack(out)


    def main():
     dest=OUT/f'geometry_N{N}.json'
     if dest.exists():return
     units=[load(OUT/f'arrays/geometry_N{N}_{k}.npz') for k in range(NP)];f=np.vstack([a['faces'] for a in units]);fp=np.concatenate([a['face_flux'] for a in units]);w=np.vstack([a['wall'] for a in units]);wp=np.concatenate([a['wall_flux'] for a in units]);ws=np.concatenate([a['wall_side'] for a in units]);wk=np.concatenate([a['wall_kind'] for a in units]);flags=np.concatenate([a['grazing_flag'] for a in units]);over=np.concatenate([a['used_traced_barycentre'] for a in units]);oldcounts=[len(load(ROOT/(f'base_polygon/arrays/geometry_{k}.npz'))['faces']) for k in range(NP)];mapping=[]
     for per in range(COPIES):
      for k,a in enumerate(units):
       ix=a['old_face_index'].copy();good=ix>=0;ix[good]+=per*sum(oldcounts)+sum(oldcounts[:k]);mapping.extend(ix)
     ff=expand(f,True);ww=expand(w);psi=np.tile(wp,COPIES);side=np.tile(ws,COPIES);kind=np.tile(wk,COPIES);fl=np.tile(flags,COPIES);ov=np.tile(over,COPIES);facepsi=np.tile(fp,COPIES);mapping=np.array(mapping,int)
     geo=dict(faces=ff,wall=ww);old=load(ROOT/'base_polygon/arrays/polygon_geometry.npz');good=mapping>=0;assert np.array_equal(ff[good],old['faces'][mapping[good]])
     save(OUT/f'arrays/polygon_geometry_N{N}.npz',**geo);save(OUT/f'arrays/wall_rows_N{N}.npz',wall=ww,psi=psi,side=side,kind=kind,grazing_flag=fl,used_traced_barycentre=ov);save(OUT/f'arrays/face_flux_N{N}.npz',psi=facepsi,prior_index=mapping)
     assert np.array_equal(ww[kind==2],old['wall'][old['wall_kind']==2]);assert np.array_equal(psi[kind==2],old['wall_flux'][old['wall_kind']==2])
     dump(dest,dict(complete=True,N=N,faces=len(ff),untouched_face_rows=int(good.sum()),untouched_faces_bitwise=True,plane_seeded_wall_rows_bitwise=True,midpoint_wall_row_count=int((kind==1).sum()),plane_seeded_wall_row_count=int((kind==2).sum()),physical_flux=float(psi.sum()),midpoint_wall_rows_flux=float(psi[kind==1].sum()),plane_seeded_wall_rows_flux=float(psi[kind==2].sum()),new_rays=0))
     log(f'N{N} geometry assembled; untouched both_planes_reached and plane_seeded_wall_rows bitwise; new rays 0')


    return {'expand': expand, 'main': main}
