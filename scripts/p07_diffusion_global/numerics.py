"""Portable P07 research kernels: frozen cubic reconstruction and shared fluxes."""
from pathlib import Path
import json,time,resource,sys
import numpy as np
REPO=Path(__file__).resolve().parents[2]
for p in (REPO,REPO/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from hsx_mms_continuum_reference import build_continuum_reference_from_sidecar
from drbx.geometry.fci_perpendicular_bracket import CubicOwnerGeometry,select_cubic_support_v3,_basis
from drbx.geometry.fci_boundary_functional_reconstruction import BoundaryRelation,prepare_boundary_reconstruction
FIELDS=('phi_mms','Ti_mms','regular_neumann','mixed_eta_neumann')
TIME=1e-6


def face_keys(n: int, indices: np.ndarray) -> np.ndarray:
    """Invert the producer's canonical stored-face numbering."""
    index = np.asarray(indices, dtype=np.int64)
    size0, size1 = (n + 1) * n * n, n * (n + 1) * n
    result = np.empty((len(index), 4), dtype=np.int64)
    radial = index < size0
    theta = (index >= size0) & (index < size0 + size1)
    eta = ~(radial | theta)
    q = index[radial]; result[radial, 0] = 0; result[radial, 1] = q // (n * n); result[radial, 2] = (q // n) % n; result[radial, 3] = q % n
    q = index[theta] - size0; result[theta, 0] = 1; result[theta, 1] = q // ((n + 1) * n); result[theta, 2] = (q // n) % (n + 1); result[theta, 3] = q % n
    q = index[eta] - size0 - size1; result[eta, 0] = 2; result[eta, 1] = q // (n * (n + 1)); result[eta, 2] = (q // (n + 1)) % n; result[eta, 3] = q % (n + 1)
    return result


def face_indices(n: int, keys: np.ndarray) -> np.ndarray:
    """Canonical bracket-builder indices; topology ``face_id`` is not this numbering."""
    key = np.asarray(keys, dtype=np.int64)
    size0, size1 = (n + 1) * n * n, n * (n + 1) * n
    result = np.empty(len(key), dtype=np.int64)
    for axis in range(3):
        rows = key[:, 0] == axis
        i, j, k = key[rows, 1:].T
        if axis == 0:
            result[rows] = i * n * n + j * n + k
        elif axis == 1:
            result[rows] = size0 + i * (n + 1) * n + j * n + k
        else:
            result[rows] = size0 + size1 + i * n * (n + 1) + j * (n + 1) + k
    return result


def quadrature(faces: tuple[np.ndarray, np.ndarray, np.ndarray], keys: np.ndarray, order: int, *, face: bool):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    if face:
        logical = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
        logical_weight = np.prod(np.stack(np.meshgrid(weights, weights, indexing="ij"), axis=-1), axis=-1).reshape(-1)
        points = np.empty((len(keys), order * order, 3)); result_weight = np.empty((len(keys), order * order))
        for axis in range(3):
            rows = np.flatnonzero(keys[:, 0] == axis)
            if not len(rows):
                continue
            local = keys[rows, 1:]
            points[rows, :, axis] = faces[axis][local[:, axis], None]
            other = [a for a in range(3) if a != axis]
            halves = []
            for slot, component in enumerate(other):
                lo, hi = faces[component][local[:, component]], faces[component][local[:, component] + 1]
                half = 0.5 * (hi - lo); halves.append(half)
                points[rows, :, component] = 0.5 * (lo + hi)[:, None] + half[:, None] * logical[None, :, slot]
            result_weight[rows] = halves[0][:, None] * halves[1][:, None] * logical_weight[None, :]
        return points, result_weight
    logical = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    logical_weight = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1), axis=-1).reshape(-1)
    points = np.empty((len(keys), order ** 3, 3)); result_weight = np.empty((len(keys), order ** 3))
    for axis in range(3):
        lo, hi = faces[axis][keys[:, axis]], faces[axis][keys[:, axis] + 1]
        half = 0.5 * (hi - lo)
        points[:, :, axis] = 0.5 * (lo + hi)[:, None] + half[:, None] * logical[None, :, axis]
        if axis == 0:
            product = half[:, None]
        else:
            product = product * half[:, None]
    result_weight[:] = product * logical_weight[None, :]
    return points, result_weight


def build_context(n: int, geometry_root: Path):
    directory = geometry_root / f"{n}x{n}x{n}"
    with np.load(directory / "base_geometry.npz", allow_pickle=False) as source:
        centers = tuple(np.asarray(source[f"grid.{axis}.centers"], dtype=np.float64) for axis in "xyz")
        faces = tuple(np.asarray(source[f"grid.{axis}.faces"], dtype=np.float64) for axis in "xyz")
    with np.load(directory / "rlp_topology.npz", allow_pickle=False) as source:
        active = np.asarray(source["is_active_owner"], dtype=bool)
        aggregate = np.asarray(source["aggregate_id"], dtype=np.int64).reshape(-1)
        raw_volume = np.asarray(source["raw_volume"], dtype=np.float64).reshape(-1)
    owner_flat = np.flatnonzero(active.reshape(-1)).astype(np.int64)
    compact = np.full(n ** 3, -1, dtype=np.int64); compact[owner_flat] = np.arange(len(owner_flat))
    raw_owner = compact[aggregate]
    if np.any(raw_owner < 0):
        raise ValueError("topology raw member without active complete owner")
    mesh = np.meshgrid(*centers, indexing="ij")
    raw_x = (mesh[0] * np.cos(mesh[1])).reshape(-1); raw_y = (mesh[0] * np.sin(mesh[1])).reshape(-1); raw_eta = mesh[2].reshape(-1)
    volume = np.bincount(raw_owner, weights=raw_volume, minlength=len(owner_flat))
    exponents = tuple((a, total-a) for total in range(4) for a in range(total + 1))
    moments = np.empty((len(owner_flat), len(exponents)))
    for column, (a, b) in enumerate(exponents):
        moments[:, column] = np.bincount(raw_owner, weights=raw_volume * raw_x**a * raw_y**b, minlength=len(owner_flat)) / volume
    owner_eta = np.bincount(raw_owner, weights=raw_volume * raw_eta, minlength=len(owner_flat)) / volume
    geometry = CubicOwnerGeometry(
        owner_flat_ids=owner_flat, owner_plane=(owner_flat % n).astype(np.int64),
        owner_angular_column=((owner_flat // n) % n).astype(np.int64),
        owner_centroid_xy=np.column_stack((moments[:, 2], moments[:, 1])), owner_eta=owner_eta,
        owner_moments_xy=moments, owner_volume=volume, raw_owner=raw_owner, raw_volume=raw_volume,
        resolution=n, dr=float(np.median(np.diff(faces[0]))), dtheta=float(np.median(np.diff(faces[1]))),
        deta=float(np.median(np.diff(faces[2]))), eta_period=float(faces[2][-1] - faces[2][0]),
        identity=f"p07-topology-midpoint-volume-n{n}",
    )
    return faces, centers, raw_owner, raw_volume, volume, geometry


def normal_row(reference, wall_point, center, scale, eta_period):
    value, derivative = _basis(np.asarray(wall_point)[None, :], center, scale, eta_period)
    metric = reference._metric(np.asarray(wall_point)[None, :])["gcontra"][0]
    normal = metric[0] / np.sqrt(metric[0, 0])
    return value[0], derivative[0] @ normal, normal

def _regular_field(points: np.ndarray, *, m: int, k: float, phase: float, amplitude: float = 0.08):
    q = np.asarray(points, dtype=np.float64)
    u, theta, eta = q[:, 0], q[:, 1], q[:, 2]
    one = 1.0 - u * u
    radial = u * u * one**4
    radial_d = 2.0 * u * one**4 - 8.0 * u**3 * one**3
    radial_dd = 2.0 * one**4 - 40.0 * u * u * one**3 + 48.0 * u**4 * one**2
    phase_value = m * theta + k * eta + phase
    c, s = np.cos(phase_value), np.sin(phase_value)
    value = amplitude * radial * c
    gradient = amplitude * np.stack((radial_d * c, -m * radial * s, -k * radial * s), axis=-1)
    hessian = np.zeros((len(q), 3, 3), dtype=np.float64)
    hessian[:, 0, 0] = amplitude * radial_dd * c
    hessian[:, 1, 1] = -amplitude * (m * m) * radial * c
    hessian[:, 2, 2] = -amplitude * (k * k) * radial * c
    hessian[:, 0, 1] = hessian[:, 1, 0] = -amplitude * m * radial_d * s
    hessian[:, 0, 2] = hessian[:, 2, 0] = -amplitude * k * radial_d * s
    hessian[:, 1, 2] = hessian[:, 2, 1] = -amplitude * m * k * radial * c
    return value, gradient, hessian

def fields(reference,points):
    raw=reference._fields_raw(np.asarray(points),TIME)
    out=[]
    for name in ('phi','Ti'):
        a=raw[name];out.append((a[0],np.stack(a[1:4],axis=-1),a[5]))
    out.extend(_regular_field(points,m=2,k=k,phase=.23) for k in (0.,-2*np.pi/reference.eta_period))
    return tuple(np.stack([v[i] for v in out],axis=1) for i in range(3))

def reference(sidecar, *, verify_hashes=True):
    ref=build_continuum_reference_from_sidecar(sidecar,verify_hashes=verify_hashes,tau=1.,mi_over_me=1836.,rho_star=1.,Ve_nu=0.,perp_diffusion=1e-5,enable_generalized_potential=True)
    ref.finite_difference_step=2e-4
    return ref

def owner_values(centers,raw_owner,raw_volume,volume,ref):
    # Bounded field evaluation; avoid retaining full-grid Hessians.
    n=len(centers[0]);values=np.zeros((len(volume),4))
    for lo in range(0,n**3,4096):
        ids=np.arange(lo,min(lo+4096,n**3));ijk=np.array(np.unravel_index(ids,(n,)*3)).T
        p=np.column_stack([centers[a][ijk[:,a]] for a in range(3)])
        v=fields(ref,p)[0]
        np.add.at(values,raw_owner[ids],raw_volume[ids,None]*v)
    return values/volume[:,None]

def face_count(n):return 3*(n+1)*n*n

def incidence_for_keys(n,keys):
    lower=np.full(len(keys),-1,dtype=np.int64);upper=lower.copy()
    for r,key in enumerate(keys):
        axis=int(key[0]);p=list(map(int,key[1:]))
        if p[axis]>0:
            a=p.copy();a[axis]-=1;lower[r]=np.ravel_multi_index(a,(n,)*3)
        if p[axis]<n:upper[r]=np.ravel_multi_index(p,(n,)*3)
    return lower,upper

def face_chunk(ctx,indices):
    started=time.monotonic();geom=ctx['geometry'];n=geom.resolution;ref=ctx['reference'];keys=face_keys(n,indices)
    pts,w=quadrature(ctx['faces'],keys,3,face=True);collapsed=(keys[:,0]==0)&(keys[:,1]==0)
    tensor=np.zeros((len(keys),9,3,3));active=~collapsed
    if active.any():tensor[active]=ref._perpendicular_flux_tensor(pts[active].reshape(-1,3)).reshape(-1,9,3,3)
    flux=np.zeros((len(keys),4));oracle=flux.copy();constant=np.zeros((len(keys),2))
    counts=np.zeros(len(keys),np.int32);levels=counts.copy();wallreach=np.zeros(len(keys),bool)
    donors_list=[];cond=np.zeros(len(keys))
    # Dynamic boundary fields are evaluated only at wall-reaching fits.
    for r,key in enumerate(keys):
        if collapsed[r]:donors_list.append(np.array([],np.int32));continue
        d,c,s,obs,disp,dist,level=select_cubic_support_v3(geom,pts[r,4],int(key[0]),int(key[3]))
        counts[r]=len(d);levels[r]=level;donors_list.append(np.asarray(d,np.int32));wallreach[r]=np.any(geom.owner_flat_ids[d]//n**2==n-1)
        wt=(1/(1+dist**2))**2
        if wallreach[r]:
            wall=pts[r,4].copy();wall[0]=1.
            vr,nr,normal=normal_row(ref,wall,c,s,geom.eta_period)
            # Preserve the point-policy constrained arithmetic and weights.
            cw=(1+dist**2)**-2
            pv=prepare_boundary_reconstruction(obs,cw,BoundaryRelation(vr[None],np.ones((1,1)),('wall-centre-value',),'value'))
            pn=prepare_boundary_reconstruction(obs,cw,BoundaryRelation(nr[None],np.ones((1,1)),('wall-centre-normal',),'normal_derivative'))
            bv,bg,_=fields(ref,wall[None]);values=ctx['values'][d]
            coefficient=np.empty((20,4));coefficient[:,0]=pv.owner_map@values[:,0]+pv.boundary_map[:,0]*bv[0,0]
            coefficient[:,1:]=pn.owner_map@values[:,1:]+pn.boundary_map*(bg[0,1:]@normal)[None,:]
            cconst=np.column_stack((pv.owner_map@np.ones(len(d))+pv.boundary_map[:,0],pn.owner_map@np.ones(len(d))))
            cond[r]=pn.observation_condition
        else:
            central=np.linalg.solve(obs.T@(wt[:,None]*obs),obs.T*wt[None,:])
            coefficient=central@ctx['values'][d];one=central@np.ones(len(d));cconst=np.column_stack((one,one))
        deriv=_basis(pts[r],c,s,geom.eta_period)[1]
        target=np.einsum('q,qj,qkj->k',w[r],tensor[r,:,key[0],:],deriv)
        flux[r]=target@coefficient;constant[r]=target@cconst
        grad=fields(ref,pts[r])[1]
        oracle[r]=np.einsum('q,qj,qfj->f',w[r],tensor[r,:,key[0],:],grad)
    lower,upper=incidence_for_keys(n,keys)
    width=max(1,max(map(len,donors_list)));donors=np.full((len(keys),width),-1,np.int32)
    for i,d in enumerate(donors_list):donors[i,:len(d)]=d
    return {'indices':np.asarray(indices),'keys':keys,'flux':flux,'oracle_flux':oracle,'constant_flux':constant,'donors':donors,'donor_count':counts,'expansion_level':levels,'wall_reaching':wallreach,'constraint_condition':cond,'lower_raw':lower,'upper_raw':upper,'seconds':np.array(time.monotonic()-started),'peak_rss_gib':np.array(rss())}

def cell_chunk(ctx,indices,order=3):
    started=time.monotonic();n=ctx['geometry'].resolution;ref=ctx['reference'];ij=np.array(np.unravel_index(indices,(n,)*3)).T
    pts,w=quadrature(ctx['faces'],ij,order,face=False);p=pts.reshape(-1,3)
    # Only the geometry this operator uses; share it across the four fields.
    J=ref._metric(p)['J'];tensor,div=ref._perpendicular_geometry(p)
    _,g,h=fields(ref,p)
    integrand=-(np.einsum('nj,nfj->nf',div,g)+np.einsum('nij,nfij->nf',tensor,h))
    num=np.sum((w.reshape(-1,1)*integrand).reshape(len(indices),-1,4),axis=1)
    volume=np.sum(w*J.reshape(len(indices),-1),axis=1)
    return {'indices':np.asarray(indices),'numerator':num,'continuous_volume':volume,'seconds':np.array(time.monotonic()-started),'peak_rss_gib':np.array(rss())}

def rss():
    v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(v/(2**30 if sys.platform=='darwin' else 2**20))

def masks(geometry):
    n=geometry.resolution;owner=geometry.owner_flat_ids;i=owner//n**2
    count=np.bincount(geometry.raw_owner,minlength=len(owner));rawcount=count[geometry.raw_owner].reshape((n,)*3)
    trans=np.zeros((n,)*3,bool);diff=rawcount[1:]!=rawcount[:-1];trans[1:]|=diff;trans[:-1]|=diff
    tm=np.bincount(geometry.raw_owner,weights=trans.reshape(-1),minlength=len(owner))>0
    axis=i==0;wall=i==n-1;transition=tm&~axis&~wall;agg=(count>1)&~axis&~wall&~transition
    return {'axis':axis,'boundary':wall,'transition':transition,'agglomerated':agg,'ordinary':~(axis|wall|transition|agg)}

def select_owners(geometry,centers):
    n=geometry.resolution;result=[];labels={};mm=masks(geometry)
    # Known tracks for replay plus representative additional actual-HSX coverage.
    seeds=[('wall',(n-1,0,0)),('agglomerated',(n//4,0,0)),('ordinary',(int(n*.64),n//2-1,n//2))]
    for name,ijk in seeds:
        oid=int(geometry.raw_owner[np.ravel_multi_index(ijk,(n,)*3)]);result.append(oid);labels.setdefault(oid,[]).append(name)
    for name in ('axis','transition'):
        ids=np.flatnonzero(mm[name]);oid=int(ids[len(ids)//2]);result.append(oid);labels.setdefault(oid,[]).append(name)
    for name,ijk in [('theta_seam',(n//2,0,n//3)),('eta_seam',(n//2,n//3,n-1))]:
        oid=int(geometry.raw_owner[np.ravel_multi_index(ijk,(n,)*3)]);result.append(oid);labels.setdefault(oid,[]).append(name)
    return np.array(sorted(set(result)),np.int64),{str(k):v for k,v in labels.items()}

def closure(geometry,owners):
    n=geometry.resolution;raw=np.flatnonzero(np.isin(geometry.raw_owner,owners));kk=set()
    for rid in raw:
        p=list(np.unravel_index(rid,(n,)*3))
        for ax in range(3):
            for side in (0,1):
                a=p.copy();a[ax]+=side;kk.add((ax,*a))
    k=np.array(sorted(kk));return raw,np.sort(face_indices(n,k))
