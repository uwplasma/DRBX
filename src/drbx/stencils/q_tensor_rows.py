"""Exact captured factors of outer Q rows with five eta planes.

P's exact-bit table helper is reused; its historical four-plane schema remains
unchanged. Inner and wall Q rows remain dense. Expansion preserves the builder's
left association and accumulation order; sum-factorized actions are exploratory
host operations with a separately checked floating reduction budget.
"""
from dataclasses import dataclass, fields
import numpy as np
from .tensor_rows import _dedupe


@dataclass(frozen=True)
class QTensorCapture:
    family: str
    radial: np.ndarray                 # (targets,4)
    radial_derivative: np.ndarray
    eta: np.ndarray                    # (targets,5)
    eta_derivative: np.ndarray
    owner: np.ndarray                  # (4,5,7), original block order
    theta: np.ndarray                  # (4,5,targets,7)
    theta_derivative: np.ndarray

    def select_targets(self, indices):
        indices = np.asarray(indices)
        return QTensorCapture(self.family, self.radial[indices].copy(),
            self.radial_derivative[indices].copy(), self.eta[indices].copy(),
            self.eta_derivative[indices].copy(), self.owner.copy(),
            self.theta[:,:,indices].copy(), self.theta_derivative[:,:,indices].copy())


@dataclass(frozen=True)
class QTensorRows:
    """Source-major table indices; each source has a common number of targets."""
    family: tuple[str, ...]
    target_count: int
    has_gradient: bool
    radial: np.ndarray
    radial_derivative: np.ndarray
    eta: np.ndarray
    eta_derivative: np.ndarray
    theta: np.ndarray
    theta_derivative: np.ndarray
    owner: np.ndarray
    t_radial: np.ndarray                 # (source,target)
    t_eta: np.ndarray                    # (source,target)
    t_theta: np.ndarray                  # (source,4,5,target)
    t_owner: np.ndarray                  # (source,4,5)

    @property
    def nbytes(self):
        return sum(getattr(self, f.name).nbytes for f in fields(self)
                   if isinstance(getattr(self, f.name), np.ndarray))

    def validate(self):
        ns, nt = len(self.family), self.target_count
        if ns == 0 or nt <= 0 or any(f not in ('singleton','ringwise') for f in self.family):
            raise ValueError('invalid Q tensor source family/count')
        for key, shape in (('t_radial',(ns,nt)), ('t_eta',(ns,nt)),
                           ('t_theta',(ns,4,5,nt)), ('t_owner',(ns,4,5))):
            ids = getattr(self,key)
            table = getattr(self,key[2:])
            if (ids.shape != shape or ids.dtype.kind not in 'iu' or
                np.any(ids < 0) or np.any(ids >= len(table))):
                raise ValueError(f'invalid Q tensor {key}')
        for key, width in (('radial',4),('eta',5),('theta',7),('owner',7)):
            a = getattr(self,key)
            if a.ndim != 2 or a.shape[1] != width or not np.isfinite(a).all():
                raise ValueError(f'invalid Q tensor {key} table')
        if self.owner.dtype.kind not in 'iu' or np.any(self.owner < 0):
            raise ValueError('invalid Q tensor owner identity')
        for key in ('radial','eta','theta'):
            d = getattr(self,key+'_derivative')
            if (self.has_gradient and d.shape != getattr(self,key).shape or
                not self.has_gradient and d.shape != (0,getattr(self,key).shape[1]) or
                not np.isfinite(d).all()):
                raise ValueError('invalid Q tensor derivative table')
        return self


def _small(ids):
    ids = np.asarray(ids)
    maximum = int(ids.max()) if ids.size else 0
    dtype = np.uint8 if maximum <= 255 else np.uint16 if maximum <= 65535 else np.uint32
    return ids.astype(dtype)


def build_q_tensor_rows(captures, *, include_gradients=False):
    """Build exact-bit deduplicated tables from builder captures only."""
    captures = tuple(captures)
    if not captures or any(c is None for c in captures):
        raise ValueError('Q tensor factors unavailable for non-outer sources')
    nt = len(captures[0].radial); ns = len(captures)
    if any(len(c.radial) != nt for c in captures):
        raise ValueError('Q tensor target count mismatch')
    result = {}
    for key, shape in (('radial',(ns,nt)), ('eta',(ns,nt)), ('theta',(ns,4,5,nt))):
        source = np.concatenate([getattr(c,key).reshape(-1,getattr(c,key).shape[-1]) for c in captures])
        if include_gradients:
            derivative = np.concatenate([getattr(c,key+'_derivative').reshape(-1,source.shape[1]) for c in captures])
            (table, dtable), ids = _dedupe(source, derivative)
        else:
            (table,), ids = _dedupe(source)
            dtable = np.zeros((0,source.shape[1]), table.dtype)
        result[key] = table
        result[key+'_derivative'] = dtable
        result['t_'+key] = _small(ids.reshape(shape))
    owners = np.concatenate([c.owner.reshape(-1,7) for c in captures])
    if np.any(owners < 0) or np.any(owners > np.iinfo(np.int32).max):
        raise ValueError('Q tensor donor identities do not fit int32')
    (result['owner'],), ids = _dedupe(owners)
    result['owner'] = result['owner'].astype(np.int32)
    result['t_owner'] = _small(ids.reshape(ns,4,5))
    return QTensorRows(tuple(c.family for c in captures), nt, bool(include_gradients), **result).validate()


def expand_q_tensor_rows(rows):
    """Return source (sorted donor,value,gradient|None) tuples, exact builder order."""
    rows.validate(); nt = rows.target_count; result = []
    for s in range(len(rows.family)):
        col = {}
        L = rows.radial[rows.t_radial[s]]
        ev = rows.eta[rows.t_eta[s]]
        if rows.has_gradient:
            D = rows.radial_derivative[rows.t_radial[s]]
            ed = rows.eta_derivative[rows.t_eta[s]]
        for l in range(4):
            for e in range(5):
                ids = rows.owner[rows.t_owner[s,l,e]]
                vv = rows.theta[rows.t_theta[s,l,e]]
                value = L[:,l,None]*ev[:,e,None]*vv
                if rows.has_gradient:
                    dd = rows.theta_derivative[rows.t_theta[s,l,e]]
                    block = np.stack((value, D[:,l,None]*ev[:,e,None]*vv,
                        L[:,l,None]*ev[:,e,None]*dd, L[:,l,None]*ed[:,e,None]*vv),axis=1)
                else:
                    block = value[:,None]
                for z,o in enumerate(ids):
                    if int(o) not in col:
                        col[int(o)] = np.zeros((nt,4 if rows.has_gradient else 1))
                    col[int(o)] += block[:,:,z]
        donor = np.array(sorted(col),np.int32)
        packed = np.stack([col[o] for o in donor],axis=-1)
        result.append((donor,packed[:,0],packed[:,1:] if rows.has_gradient else None))
    return result


def verified_q_tensor_rows(captures, expected, *, include_gradients=False, require_smaller=True):
    """Return (factors,report) or (None,report) for nonexact/nonbeneficial storage."""
    captures,expected=tuple(captures),tuple(expected)
    if len(captures)!=len(expected):
        raise ValueError('Q tensor expected source count mismatch')
    report = dict(exact=False, selected=False, dense_bytes=0, factor_bytes=None)
    if any(c is None for c in captures):
        report['reason'] = 'unsupported source retains dense rows'
        return None,report
    rows = build_q_tensor_rows(captures, include_gradients=include_gradients)
    report['factor_bytes'] = rows.nbytes
    report['dense_bytes'] = sum(d.nbytes+v.nbytes+(g.nbytes if include_gradients else 0)
                                for d,v,g in expected)
    report['replaced_row_bytes'] = sum(v.nbytes+(g.nbytes if include_gradients else 0) for _,v,g in expected)
    report['shared_donor_net_saving_bytes'] = report['replaced_row_bytes']-rows.nbytes
    for (ad,av,ag),(ed,ev,eg) in zip(expand_q_tensor_rows(rows),expected):
        pairs = [(ad.astype(ed.dtype),ed),(av,ev)]
        if include_gradients:
            pairs.append((ag,eg))
        if any(a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes() for a,b in pairs):
            report['reason'] = 'nonexact expansion retains dense rows'
            return None,report
    report['exact'] = True
    if require_smaller and rows.nbytes >= report['replaced_row_bytes']:
        report['reason'] = 'factor payload does not reduce bytes; retains dense rows'
        return None,report
    report.update(selected=True,reason='exact expansion and measured smaller payload')
    return rows,report


def apply_q_tensor_values_numpy(rows, owner_fields):
    """Exploratory sum factorization; changes floating reduction order.

    Shape ``(...,owner)`` returns ``(...,source,target)``. No dense weights are
    expanded. This host benchmark is not selected by a Q runtime plan.
    """
    rows.validate(); x=np.asarray(owner_fields)
    outputs=[]
    for s in range(len(rows.family)):
        angular=[]
        for l in range(4):
            eta=[]
            for e in range(5):
                ids=rows.owner[rows.t_owner[s,l,e]]
                theta=rows.theta[rows.t_theta[s,l,e]]
                eta.append(np.einsum('...j,tj->...t',x[...,ids],theta))
            angular.append(np.stack(eta,axis=-1))
        block=np.stack(angular,axis=-2)  # (...,target,radial,eta)
        ev=rows.eta[rows.t_eta[s]]; L=rows.radial[rows.t_radial[s]]
        outputs.append(np.sum(np.sum(block*ev[...,None,:],axis=-1)*L,axis=-1))
    return np.stack(outputs,axis=-2)
