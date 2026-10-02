"""Portable Q08 catalogue, live BC queries and component replay checks.

This module does not select a JAX backend or import a prior campaign driver.
The accepted omega is a linear combination of five primitive fields, not Te.
"""
from pathlib import Path
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import numpy as np

DESIGNS=[dict(name='constant'),dict(name='smooth')]
DESIGNS += [dict(name=f'wave_l{lam}_a{angle:g}_p{phase}',wavelength=lam,angle=angle,phase=phase*np.pi/2)
            for lam in (1.4,.7,.5,.35) for angle in (30.,75.) for phase in (0,1)]
DESIGNS += [dict(name=f'heldout_l{lam}_a{angle}',wavelength=lam,angle=angle,phase=np.pi/4)
            for lam in (.7,.35) for angle in (10.,110.)]
NF=len(DESIGNS)
BASE=np.array([1.,1.1,.9,.13,.08]);AMP=np.array([.10,.05,.05,.03,.04])
WC=np.array([.6,-.4,0.,.3,0.]);WOFF=.2-WC@BASE
COEFF=np.arange(1,7)*.01;TAU=1.;MU=1836.
FIVE_KINDS=((('D',)*5,'D'),(('N',)*5,'N'),
            (('D','N','D','N','D'),'N'),(('N','D','N','D','N'),'D'))
KINDS=tuple((five+(five[1],),phi) for five,phi in FIVE_KINDS)
SPAN_SLOTS=((0,4,2),(1,3,2))


def primitive_modes(points,mode):
    p=np.asarray(points);u,th,e=np.moveaxis(p,-1,0);x=u*np.cos(th);y=u*np.sin(th)
    dx=np.stack((np.cos(th),-y,np.zeros_like(u)),axis=-1)
    dy=np.stack((np.sin(th),x,np.zeros_like(u)),axis=-1)
    if mode=='sx':
        v=x*np.cos(e);g=dx*np.cos(e)[...,None];g[...,2]=-x*np.sin(e)
    elif mode=='sy':
        v=y*np.sin(2*e);g=dy*np.sin(2*e)[...,None];g[...,2]=2*y*np.cos(2*e)
    else:
        raise ValueError('Q08 catalogue uses only sx/sy primitive controls')
    return v,g


def fields(points):
    p=np.asarray(points);u,th,e=np.moveaxis(p,-1,0);x=u*np.cos(th);y=u*np.sin(th)
    dx=np.stack((np.cos(th),-y,np.zeros_like(u)),axis=-1)
    dy=np.stack((np.sin(th),x,np.zeros_like(u)),axis=-1)
    vals=[];grads=[]
    for d in DESIGNS:
        vv=[];gg=[]
        for j in range(5):
            if d['name']=='constant':f=np.zeros_like(u);g=np.zeros_like(p)
            elif d['name']=='smooth':f,g=primitive_modes(p,'sx' if j%2==0 else 'sy')
            else:
                a=np.deg2rad(d['angle']+13*j);k=2*np.pi/d['wavelength'];m=1+j%2
                ph=k*(np.cos(a)*x+np.sin(a)*y)+m*e+d['phase']+.3*j
                dp=k*(np.cos(a)*dx+np.sin(a)*dy);dp[...,2]+=m
                f=np.cos(ph);g=-np.sin(ph)[...,None]*dp
            vv.append(BASE[j]+AMP[j]*f);gg.append(AMP[j]*g)
        vals.append(np.stack(vv,axis=-1));grads.append(np.stack(gg,axis=-2))
    return np.stack(vals,axis=-2),np.stack(grads,axis=-3)


def six_fields(points):
    value,gradient=fields(points)
    omega=WOFF+value@WC
    domega=np.einsum('...fa,f->...a',gradient,WC)
    return (np.concatenate((value,omega[...,None]),axis=-1),
            np.concatenate((gradient,domega[...,None,:]),axis=-2))


def phi_fields(points):
    a,da=primitive_modes(points,'sx');b,db=primitive_modes(points,'sy')
    v=.07*a+.04*b;g=.07*da+.04*db
    return np.stack([v*0]+[v]*(NF-1),axis=-1),np.stack([g*0]+[g]*(NF-1),axis=-2)


def boundaries(bank,case=None):
    """Live inner/outer six-field and inner-phi BCs from exact bank query maps.

    ``case=None`` returns a batch of 22; an integer removes the case axis.
    Only the bank array interface is required, allowing trusted merged banks.
    Nonwall omega padding retains the accepted old ``six_bc`` affine offset.
    """
    from drbx.native.q_parallel import QBoundaryData
    if case is not None and (not isinstance(case,(int,np.integer)) or not 0<=case<NF):
        raise ValueError('invalid Q08 catalogue case')
    selection=slice(None) if case is None else slice(int(case),int(case)+1)
    nc=NF if case is None else 1
    nr=len(bank.raw)
    outer=[np.zeros((nc,5,nr,35)),np.zeros((nc,5,nr,3)),
           np.zeros((nc,5,nr,3,2)),np.zeros((nc,5,nr,35))]
    inner=[a.copy() for a in outer]
    phi=[np.zeros((nc,nr,35)),np.zeros((nc,nr,3)),
         np.zeros((nc,nr,3,2)),np.zeros((nc,nr,35))]
    for w,r in enumerate(np.asarray(bank.wall_index)):
        points=bank.query_table[bank.wall_node_query[w]]
        v,g=fields(points)
        v,g=v[:,selection],g[:,selection]
        normal=np.einsum('qa,qcfa->cfq',bank.boundary_wall_normal[w],g)
        for target in (outer,inner):
            target[0][:,:,r]=v.transpose(1,2,0);target[3][:,:,r]=normal
        for target,slots in ((outer,SPAN_SLOTS[0]),(inner,SPAN_SLOTS[1])):
            points=bank.query_table[bank.wall_slot_query[w,list(slots)]]
            v,g=fields(points)
            v,g=v[:,selection],g[:,selection]
            target[1][:,:,r]=v.transpose(1,2,0)
            target[2][:,:,r]=g[:,:,:,1:].transpose(1,2,0,3)
        points=bank.query_table[bank.wall_node_query[w]]
        v,g=phi_fields(points);v,g=v[:,selection],g[:,selection];phi[0][:,r]=v.T
        phi[3][:,r]=np.einsum('qa,qca->cq',bank.boundary_wall_normal[w],g)
        points=bank.query_table[bank.wall_slot_query[w,list(SPAN_SLOTS[1])]]
        v,g=phi_fields(points);v,g=v[:,selection],g[:,selection]
        phi[1][:,r]=v.T;phi[2][:,r]=g[:,:,1:].transpose(1,0,2)
    def six(target):
        result=[]
        for i,a in enumerate(target):
            extra=np.einsum('cf...,f->c...',a,WC)
            if i in (0,1):extra+=WOFF
            result.append(np.concatenate((a,extra[:,None]),axis=1))
        return result
    result=(six(inner),six(outer),phi)
    if case is not None:
        result=tuple([a[0] for a in target] for target in result)
    return tuple(QBoundaryData(*target) for target in result)


def check_outputs(actual,expected,*,atol=1e-8,rtol=1e-11):
    """Check every NamedTuple leaf; never compare only the combined action."""
    metrics={}
    def visit(a,b,path):
        if hasattr(a,'_fields') or hasattr(b,'_fields'):
            if getattr(a,'_fields',None)!=getattr(b,'_fields',None):
                raise ValueError('Q08 output structure mismatch '+path)
            for key in a._fields:visit(getattr(a,key),getattr(b,key),path+'.'+key if path else key)
            return
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or a.dtype.kind!=b.dtype.kind:
            raise ValueError('Q08 output shape/dtype mismatch '+path)
        if a.dtype.kind=='b':
            if not np.array_equal(a,b) or not a.all():
                raise ValueError('Q08 invalid/mismatched boolean output '+path)
            maximum=scaled=0.
        else:
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Q08 nonfinite output '+path)
            error=abs(a-b);budget=atol+rtol*abs(b)
            maximum=float(np.max(error,initial=0));scaled=float(np.max(error/budget,initial=0))
            if scaled>1:
                raise ValueError(f'Q08 output replay {path}: scaled {scaled} > 1')
        metrics[path]=dict(max_abs_error=maximum,max_scaled_error=scaled,shape=list(a.shape),dtype=str(a.dtype))
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol<=0 or rtol<0:
        raise ValueError('invalid Q08 replay budget')
    visit(actual,expected,'')
    return dict(max_scaled_error=max((v['max_scaled_error'] for v in metrics.values()),default=0.),
                max_abs_error=max((v['max_abs_error'] for v in metrics.values()),default=0.),leaves=metrics)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_geometry_model(inputdir):
    """Load only the copied C3 model package, without campaign driver imports."""
    path=Path(inputdir)/'geometry_model'
    name='q08_geometry_model'
    if name in sys.modules and Path(sys.modules[name].__file__).resolve()!= (path/'__init__.py').resolve():
        raise ValueError('Q08 geometry model already loaded from a different input directory')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
        package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    return importlib.import_module(name+'.model')
