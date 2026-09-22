"""Frozen P05 nodewise cubic-jump correction to the qualified matched A action."""
import hashlib
import numpy as np

BIAS = 0.75
FIELDS = ('actual_vorticity', 'smooth_regular_scalar', 'smooth_eta_varying_scalar')
POLICY = {'name':'p05-recentered-nodewise-cubic-jump-v1', 'bias':BIAS,
          'support':'frozen centered selection-v3 support; donor hash verified on every face',
          'face_quadrature':3, 'side_fit':'SVD weighted least squares of actual cubic owner moments',
          'weights_squared':'(1+d^2)^-2 * (1 +/- 0.75*tanh(s))',
          'common_value':'unchanged qualified q_star',
          'wall':'legacy center-to-boundary jump, no exterior fit',
          'collapsed_axis':'zero', 'limiter':'none; smooth static accuracy experiment',
          'required_fields':list(FIELDS[1:]), 'vorticity':'diagnostic only'}


def fit_pair(numeric, context, point, axis, eta_index):
    c = numeric.cubic
    for level, policy in enumerate(c.POLICY['deficient_row_expansion_schedule']):
        donors, _, diag, _ = c._row_batch(context, axis, eta_index % context.resolution,
            point[None,:], exact_query=False, count=int(policy['donors_per_plane']),
            pool_count=int(policy['candidate_pool_per_plane']))
        if (diag['rank'][0] == len(c.EXPONENTS) and
            np.max(diag['residual'][0]) <= c.base.REPRODUCTION_TOLERANCE and
            diag['minimum_coverage'][0] >= c.POLICY['minimum_distinct_angular_columns_per_plane']):
            break
    else:
        raise RuntimeError('frozen cubic support remains deficient')
    donors = donors[0]; scale = diag['scale'][0]
    center = np.array([point[0]*np.cos(point[1]), point[0]*np.sin(point[1]), point[2]])
    P = c._centered_observations(context, donors[None,:], center[None,:], scale[None,:])[0]
    eta = c.base._unwrap_periodic(context.owner_eta[donors], center[2], context.eta_period)
    displacement = np.column_stack((context.arrays['owner_centroid_xy'][donors]-center[:2], eta-center[2]))
    distance = np.sqrt(np.sum((displacement/scale)**2,axis=1))
    weight2 = (1/(1+distance**2))**2
    normal = (np.array([np.cos(point[1]),np.sin(point[1]),0.]) if axis==0 else
              np.array([-np.sin(point[1]),np.cos(point[1]),0.]) if axis==1 else np.array([0.,0.,1.]))
    s = displacement@normal / np.sqrt(np.sum((normal*scale)**2))
    values = np.asarray(context.arrays['owner_values'])[:,donors]
    # The common/generator fit retains the original normal-equation arithmetic;
    # only the two NEW side fits use rank-revealing least squares.
    central = np.linalg.solve(P.T@(weight2[:,None]*P), P.T@(weight2[:,None]*values.T)).T
    fits=[]; conditions=[]; reproduction=[]
    for sign in (-1.,1.):
        root = np.sqrt(weight2*(1+sign*BIAS*np.tanh(s)))
        design = P*root[:,None]
        # One factorization for physical data and polynomial-reproduction RHS.
        solution, _, rank, singular = np.linalg.lstsq(design,
            np.column_stack((values.T,P))*root[:,None],rcond=None)
        if rank != len(c.EXPONENTS):
            raise RuntimeError('side fit lost cubic rank')
        conditions.append(float(singular[0]/singular[-1]))
        reproduction.append(float(np.max(abs(solution[:,len(values):]-np.eye(len(c.EXPONENTS))))))
        fits.append(numeric.Fit(point,center,scale,solution[:,:len(values)].T,{}))
    return numeric.Fit(point,center,scale,central,{}), fits, {
        'donor_hash':hashlib.sha256(np.asarray(donors,dtype='<i8').tobytes()).hexdigest(),
        'donor_count':len(donors), 'fallback':level, 'condition':max(conditions),
        'reproduction':max(reproduction),
    }


def compute(numeric, context, reference, prepare, indices):
    n=context.resolution; keys=numeric._face_keys(n,indices)
    points,weights=numeric._face_quadrature(context,keys)
    count=len(indices)
    common=np.zeros((3,count,9)); gradient=np.zeros((count,9,3)); jump=np.zeros_like(common)
    hashes=np.full(count,'collapsed',dtype='U64')
    condition=np.zeros(count);reproduction=np.zeros(count);donor_count=np.zeros(count,dtype=int)
    fallback=np.full(count,255,dtype=np.uint8)
    center=prepare['center'].reshape((4,n,n,n))
    for row,(axis,i,j,k) in enumerate(keys):
        if axis==0 and i==0:
            continue
        if axis==0 and i==n:
            common[:,row]=prepare['boundary_value'][1:4,j,k,None]
            gradient[row]=prepare['boundary_gradient'][0,j,k,None,:]
            jump[:,row]=(prepare['boundary_value'][1:4,j,k]-center[1:4,n-1,j,k])[:,None]
            hashes[row]=hashlib.sha256(np.asarray(prepare['boundary_donors'][j,k],dtype='<i8').tobytes()).hexdigest()
            continue
        central,sides,diag=fit_pair(numeric,context,points[row,4],int(axis),int(k))
        values,grad=numeric._evaluate_fit(central,points[row],context.eta_period)
        common[:,row]=values[1:4];gradient[row]=grad[0]
        left=numeric._evaluate_fit(sides[0],points[row],context.eta_period)[0][1:4]
        right=numeric._evaluate_fit(sides[1],points[row],context.eta_period)[0][1:4]
        jump[:,row]=right-left
        hashes[row]=diag['donor_hash'];condition[row]=diag['condition']
        reproduction[row]=diag['reproduction'];donor_count[row]=diag['donor_count'];fallback[row]=diag['fallback']
    regular=np.flatnonzero(~((keys[:,0]==0)&(keys[:,1]==0)))
    h=np.zeros((count,9,3))
    if len(regular):
        h[regular]=numeric.bounded._h_without_rho(reference,points[regular].reshape(-1,3)).reshape(len(regular),9,3)
    U=np.cross(h,gradient)[np.arange(count)[:,None],np.arange(9)[None,:],keys[:,0,None]]
    delta=np.sum(-.5*weights[None,:,:]*abs(U)[None,:,:]*jump,axis=2)
    f0=np.sum(weights[None,:,:]*U[None,:,:]*common,axis=2)
    if not np.isfinite(delta).all():
        raise RuntimeError('nonfinite material correction')
    return dict(indices=np.asarray(indices), delta_flux=delta, central_flux=f0,
                donor_hash=hashes,condition=condition,reproduction=reproduction,
                donor_count=donor_count,fallback=fallback)


def scatter(numeric,n,raw_owner,indices,flux,out):
    """Use the original raw-grid endpoint incidence, including periodic copies.

    j=0/j=N and k=0/k=N are stored as separate endpoint faces in the frozen
    campaign. Each acts on its ONE endpoint cell; wrapping both ends here
    would double-count the periodic seam.
    """
    keys=numeric._face_keys(n,indices)
    for axis in range(3):
        rows=np.flatnonzero(keys[:,0]==axis)
        coords=keys[rows,1:].copy()
        for sign,offset in ((1.,-1),(-1.,0)):
            cell=coords.copy();cell[:,axis]+=offset
            keep=np.all((cell>=0)&(cell<n),axis=1)
            raw=(cell[keep,0]*n+cell[keep,1])*n+cell[keep,2]
            owner=raw_owner[raw]
            for field in range(len(flux)):
                np.add.at(out[field],owner,sign*flux[field,rows[keep]])
