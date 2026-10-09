"""N-generic frozen Q operator, original-L checks and P Krylov reuse."""
from pathlib import Path
import json,time
from . import common as c
REPO=None
np=jax=jnp=sla=E=None
sha=c.sha
peak_gib=c.peak_gib

def initialize(run,repo,core_index=None):
    global REPO,np,jax,jnp,sla,E
    REPO=Path(repo)
    backend=c.bootstrap(run,repo,core_index=core_index)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import scipy.sparse.linalg as sla
    from . import eigcore as E
    return backend

def topology(folder):
    """Read canonical topology; no physical geometry evaluation or tracing."""
    from drbx.stencils.q_parallel import topology_from_arrays
    with np.load(Path(folder) / 'base_geometry.npz', allow_pickle=False) as z:
        centers = tuple(z[f'grid.{a}.centers'].copy() for a in 'xyz')
    with np.load(Path(folder) / 'rlp_topology.npz', allow_pickle=False) as z:
        return topology_from_arrays(centers, z['is_active_owner'], z['aggregate_id'], z['raw_volume'])


def observe(t, owners, values):
    """Physical-volume average over complete members of requested owners only."""
    members=np.concatenate([t.order[t.starts[o]:t.starts[o+1]] for o in owners])
    lookup=np.full(len(t.vol),-1,int);lookup[owners]=np.arange(len(owners))
    output=None
    for start in range(0,len(members),4096):
        raw=members[start:start+4096];v=np.asarray(values(t.pts[raw]))
        if output is None:output=np.zeros((len(owners),*v.shape[1:]),dtype=v.dtype)
        weights=t.rv[raw].reshape((len(raw),)+(1,)*(v.ndim-1))
        np.add.at(output,lookup[t.ro[raw]],weights*v)
    return output/t.vol[owners].reshape((len(owners),)+(1,)*(output.ndim-1))


class Operator:
    def __init__(self, args):
        from dataclasses import replace
        from drbx.stencils.q_artifact import load_q_bank
        from drbx.stencils.q_plan import lower_q_plan
        from drbx.stencils.q_parallel import QRuntime
        from drbx.native.q_parallel import apply_q, homogeneous_boundary
        from drbx.native.q_plan import apply_q_plan
        from scripts.q09_evolved_mms.mms import smooth_fields, COEFFICIENTS, BASE
        from scripts.q08_extraction_global.common import boundaries
        from scripts.q08_rhs_mms_global.science import REGIONS, masks

        self.args = args
        self.bank = bank = load_q_bank(args.bank)
        self.t = t = topology(args.topology)
        if bank.metadata['n'] != t.n or bank.metadata['n_owner'] != len(t.vol):
            raise ValueError('canonical topology size mismatch')
        np.testing.assert_array_equal(t.ro[bank.raw], bank.owners[bank.raw_to_owner])
        np.testing.assert_allclose(bank.raw_weight, t.rv[bank.raw]/t.vol[t.ro[bank.raw]], rtol=2e-13)
        np.testing.assert_allclose(bank.diagnostics['slot_points'][:,2], t.pts[bank.raw], atol=2e-14, rtol=0)
        self.owners = bank.owners if args.owners is None else np.array(json.loads(Path(args.owners).read_text()), dtype=int)
        if not np.array_equal(self.owners, np.unique(self.owners)) or not np.isin(self.owners, bank.owners).all():
            raise ValueError('owners must be sorted, unique and supplied by bank')
        self.full = (np.array_equal(self.owners, np.arange(len(t.vol))) and
                     np.array_equal(np.sort(bank.raw), np.arange(t.n**3)))
        if args.scope == 'full' and not self.full:
            raise ValueError('FULL scope requires all owners and all raw members; no pinned halos')
        if args.command == 'verify' and self.full:
            raise ValueError('verification is bounded-only')
        self.fields = 1 if args.operator == 'diffusion' else 6
        self.shape = (self.fields, len(self.owners))
        self.size = int(np.prod(self.shape))
        if args.command == 'verify' and self.size > args.dense_limit:
            raise ValueError('verification dense size limit exceeded')
        self.volume = t.vol[self.owners]
        self.H = np.tile(self.volume, self.fields)
        self.sqrtH = np.sqrt(self.H)
        output_index = np.searchsorted(bank.owners, self.owners)
        # Relabel required donors only; physical rows and owner projection stay
        # untouched. For FULL scope this is the full owner set. For a subset,
        # exterior perturbations are zero and explicitly labeled artificial.
        needed = np.unique(np.r_[bank.donor.ravel(), self.owners])
        local_donor = np.searchsorted(needed, bank.donor)
        local_variable = np.searchsorted(needed, self.owners)
        q0 = observe(t, needed, lambda p: np.asarray(smooth_fields(p, args.base_time)[0])).T
        phi = observe(t, needed, lambda p: np.asarray(smooth_fields(p, args.base_time)[2]))
        self.coordinates = observe(t, self.owners, lambda p: np.column_stack((p[:,0]*np.cos(p[:,1]), p[:,0]*np.sin(p[:,1]), p[:,2])))
        radial = (t.pts[t.order[t.starts[self.owners]],0]*t.n).astype(int)
        if args.last_aggregate is None:
            regional_plan=json.loads(args.regional_plan.read_text())[str(t.n)]
            if regional_plan['n_owner']!=len(t.vol) or regional_plan['n_raw']!=t.n**3:
                raise ValueError('regional plan topology mismatch')
            args.last_aggregate=int(regional_plan['last_aggregate'])
        self.regions = dict(zip(REGIONS, masks(radial,t.n,args.last_aggregate), strict=True))
        self.regions.update(axis_rlp_aggregates=radial<=args.last_aggregate,
                            last_two_rings=radial>=t.n-2)
        self.q0 = jnp.asarray(q0[:,local_variable] if self.fields==6 else np.zeros(self.shape)).reshape(-1)
        self.calls = dict(matvec=0, rmatvec=0)
        self.start = time.perf_counter()

        if self.fields == 1:
            ai = 1  # accepted h/32, independent of six-field channel constants
            nr = len(bank.raw); wall = bank.wall_index
            dd = np.asarray(bank.diffusion_D[ai]); nn = dd.copy()
            nn[wall] = bank.diffusion_N_wall[ai]
            lifts = {}
            for key in ('boundary_D_node','boundary_D_tangent','boundary_N_normal'):
                source = getattr(bank,key)[ai]
                a = np.zeros((nr,*source.shape[1:])); a[wall] = source
                lifts[key] = a
            rt = QRuntime(dict(bank.metadata,n_owner=len(needed)),bank.raw,local_donor,dd,nn,
                lifts['boundary_D_node'],lifts['boundary_D_tangent'],lifts['boundary_N_normal'],
                bank.owner_raw,bank.owner_weight)
            keys = tuple(k for k in rt.__dataclass_fields__ if k!='metadata')
            arrays = tuple(jnp.asarray(getattr(rt,k)) for k in keys)
            bc = homogeneous_boundary(rt)
            self.data = (arrays,bc)
            def numerical(x, data):
                arrays, bc = data
                runtime = QRuntime(rt.metadata,**dict(zip(keys,arrays,strict=True)))
                u = jnp.zeros(len(needed),dtype=x.dtype).at[local_variable].set(x)
                return args.diffusion_coefficient*apply_q(runtime,u,bc,kind=args.bc)[output_index]
        else:
            with np.load(args.geometry,allow_pickle=False) as z:
                geometry = {k:z[k].copy() for k in ('magnetic_L','b_eta','eta_step','bmag')}
                # Fixture files also carry owner/raw identities; enforce them.
                for key in ('owners','raw'):
                    if key in z.files:
                        np.testing.assert_array_equal(z[key],getattr(bank,key))
            np.testing.assert_allclose(geometry['b_eta'],bank.magnetic_b[:,2,2],rtol=2e-13,atol=1e-10)
            np.testing.assert_allclose(geometry['eta_step'],np.pi/t.n/32,rtol=2e-13,atol=0)
            np.testing.assert_allclose(geometry['magnetic_L'][:,:2],bank.magnetic_L[1,:,:2],rtol=2e-13,atol=1e-10)
            plan = lower_q_plan(bank,diffusion_span=1/32,**geometry)
            plan = replace(plan,n_owner=len(needed),donor=local_donor)
            bi,bo,pb = boundaries(bank,1); zi,zo,zp = boundaries(bank,0)
            amplitude = 1+.1*np.sin(args.base_time)
            # Base wall data are exact Q09 MMS data at t*. They are constants
            # of differentiation: perturbations have homogeneous D/N data.
            bi,bo,pb = jax.tree.map(lambda v,b:b+amplitude*(v-b),(bi,bo,pb),(zi,zo,zp))
            self.data = jax.tree.map(jnp.asarray,(plan,q0,phi,bi,bo,pb))
            def numerical(x,data):
                plan,base,phi,bi,bo,pb = data
                state = base.at[:,local_variable].set(x.reshape(self.shape))
                result = apply_q_plan(plan,state,bi,bo,phi,pb,COEFFICIENTS,kinds=(args.bc,)*6,
                    phi_kind=args.bc,tau=1.,mu=1836.,characteristic_method='polynomial')
                return result.combined[output_index].T.reshape(-1)
            # Admissibility must pass before accepting a base-state derivative.
            result = apply_q_plan(*self.data[:1], self.data[1], self.data[3],self.data[4],
                self.data[2],self.data[5],COEFFICIENTS,kinds=(args.bc,)*6,phi_kind=args.bc,
                tau=1.,mu=1836.,characteristic_method='polynomial')
            if not np.asarray(result.inputs_valid & result.eigensystem_admissible).all():
                raise ValueError('base reconstruction/characteristic split is inadmissible')
            del result
        self.numerical = jax.jit(numerical)
        self.jvp = jax.jit(lambda v,data,x0:jax.jvp(lambda x:numerical(x,data),(x0,),(v,))[1])
        self.vjp = jax.jit(lambda v,data,x0:jax.vjp(lambda x:numerical(x,data),x0)[1](v)[0])
        self.edges = self.make_edges()

    def real_action(self,v,transpose=False):
        v = np.asarray(v,dtype=np.float64).reshape(self.size)
        fn = self.vjp if transpose else self.jvp
        self.calls['rmatvec' if transpose else 'matvec'] += 1
        out = np.asarray(fn(jnp.asarray(v),self.data,self.q0))
        if not np.isfinite(out).all():
            raise ValueError('nonfinite linear action')
        return out

    def matvec(self,v):
        if np.iscomplexobj(v):
            return self.real_action(np.real(v))+1j*self.real_action(np.imag(v))
        return self.real_action(v)

    def rmatvec(self,v):
        if np.iscomplexobj(v):
            return self.real_action(np.real(v),True)+1j*self.real_action(np.imag(v),True)
        return self.real_action(v,True)

    def symmetric(self,u):
        """S=(HL+L^T H)/2, in physical coordinates."""
        return .5*(self.H*self.matvec(u)+self.rmatvec(self.H*u))

    def normalized_symmetric(self,y):
        """Euclidean symmetric representation H^-1/2 S H^-1/2."""
        return .5*(self.sqrtH*self.matvec(y/self.sqrtH)+self.rmatvec(self.sqrtH*y)/self.sqrtH)

    def energy(self,u):
        return float(np.real(np.vdot(u,self.H*self.matvec(u)))/np.real(np.vdot(u,self.H*u)))

    def make_edges(self):
        t=self.t; lookup=np.full(len(t.vol),-1,int);lookup[self.owners]=np.arange(len(self.owners))
        ro=t.ro.reshape((t.n,)*3); output={}
        for axis,name in enumerate(('radial','theta','eta')):
            neighbor=np.roll(ro,-1,axis=axis)
            a,b=lookup[ro].ravel(),lookup[neighbor].ravel()
            good=(a>=0)&(b>=0)&(a!=b)
            if axis==0:
                valid=np.ones(ro.shape,bool);valid[-1]=False;good &= valid.ravel()
            pairs=np.unique(np.sort(np.column_stack((a[good],b[good])),axis=1),axis=0)
            output[name]=pairs
        return output

    def localization(self,v):
        x=v.reshape(self.shape);amplitude=np.abs(x)**2*self.volume[None,:];total=amplitude.sum()
        shares={name:float(amplitude[:,mask].sum()/total) for name,mask in self.regions.items()}
        edges={}
        for name,pairs in self.edges.items():
            if len(pairs)==0:
                edges[name]=dict(edges=0,roughness=None,opposite_phase_fraction=None);continue
            a,b=pairs.T;w=np.minimum(self.volume[a],self.volume[b])[None,:]
            activity=w*(abs(x[:,a])**2+abs(x[:,b])**2)
            denominator=float(activity.sum())
            edges[name]=dict(edges=len(pairs),roughness=float((w*abs(x[:,b]-x[:,a])**2).sum()/denominator) if denominator else None,
                opposite_phase_fraction=float(activity[np.real(x[:,a]*np.conj(x[:,b]))<0].sum()/denominator) if denominator else None)
        return dict(region_fractions=shares,regions_overlap=True,
            field_fractions=(amplitude.sum(axis=1)/total).tolist(),
            field_names=['scalar'] if self.fields==1 else ['n','Te','Ti','Vi','Ve','omega'],
            adjacency=edges,grid_scale_definition='roughness approaches 2 for alternating neighbors; 0 for equal values',
            peak_owner=int(self.owners[np.argmax(amplitude.sum(axis=0))]))

    def probe_vectors(self,seeds):
        for seed in seeds:
            rng=np.random.default_rng(seed)
            for name in ('smooth','grid_scale'):
                if name=='grid_scale':u=rng.normal(size=self.shape)
                else:
                    coefficients=rng.normal(size=(self.fields,7))
                    def values(p):
                        r,theta,eta=p.T;x=r*np.cos(theta);y=r*np.sin(theta)
                        basis=np.column_stack((np.ones(len(r)),x,y,x*np.cos(eta),y*np.sin(eta),x*np.sin(2*eta),y*np.cos(2*eta)))
                        if self.args.bc=='D':basis *= (1-r*r)[:,None]**2
                        return basis@coefficients.T
                    u=observe(self.t,self.owners,values).T
                yield seed,name,u.reshape(-1)

    def energy_probes(self,seeds):
        return [dict(seed=seed,kind=name,energy_rate=self.energy(u),artificial_boundary_warning=not self.full)
            for seed,name,u in self.probe_vectors(seeds)]

    def metadata(self):
        return dict(n=self.t.n,owner_count=len(self.owners),fields=self.fields,size=self.size,
            full_domain=self.full,physics_evidence=self.full,artificial_closure=not self.full,
            closure='none' if self.full else 'exterior perturbations zero; tooling verification only',
            base_time=self.args.base_time,base='physical-volume member observation of Q09 smooth MMS',
            prescribed_phi='fixed at base time',base_boundary='fixed exact MMS; homogeneous perturbation',source='excluded',
            derivative_contract='linear scalar' if self.fields==1 else 'Q09 live matrix, stop_gradient characteristic projectors',
            true_nonlinear_jacobian_qualified=self.fields==1,
            H='physical owner volumes, repeated per field; no extra Jacobian; diagnostic fluctuation norm',
            bank_identity=self.bank.identity,input_hashes={str(p):sha(p) for p in (self.args.bank,self.args.topology/'rlp_topology.npz',self.args.topology/'base_geometry.npz')},
            geometry_sha256=sha(self.args.geometry) if self.fields==6 else None,
            source_hashes={str(p):sha(p) for p in (
                Path(__file__),Path(E.__file__),REPO/'scripts/q09_evolved_mms/bootstrap.py',
                REPO/'scripts/q09_evolved_mms/mms.py',
                REPO/'scripts/q08_extraction_global/common.py',
                REPO/'scripts/q08_rhs_mms_global/science.py',
                REPO/'scripts/q09_evolved_mms/runtime/drbx/native/q_plan.py',
                REPO/'scripts/q09_evolved_mms/runtime/drbx/native/q_parallel.py',
                REPO/'scripts/q09_evolved_mms/runtime/drbx/native/q_characteristic_polynomial.py')},
            calls=self.calls.copy(),peak_rss_gib=peak_gib(),elapsed_seconds=time.perf_counter()-self.start)


def spectrum(op,args,rho=None):
    """Reuse P's restarted propagator KS; verify every candidate with original L.

    These are candidates, not proof that every rightmost eigenvalue was found.
    Multiple seeds and residuals are retained. Zero modes use an absolute gate
    in addition to the requested relative residual (undefined at lambda=0).
    """
    if rho is None:
        linear=sla.LinearOperator((op.size,)*2,matvec=op.matvec,dtype=np.float64)
        lm=sla.eigs(linear,k=2,which='LM',ncv=min(args.ncv,op.size),tol=1e-3,
                    maxiter=args.maxiter,v0=np.random.default_rng(7).normal(size=op.size),return_eigenvectors=False)
        rho=float(max(abs(lm)))
    if not np.isfinite(rho) or rho<=0:raise ValueError('positive spectral scale required')
    dt=.3/rho
    def propagator(v):
        a=op.matvec(v);b=op.matvec(v+.5*dt*a);c=op.matvec(v+.5*dt*b);d=op.matvec(v+dt*c)
        return v+dt*(a+2*b+2*c+d)/6
    records=[];restarts=[];vectors=[]
    for seed in args.seeds:
        initial=np.random.default_rng(seed).normal(size=op.size)
        if args.seed_vector is not None:
            initial=np.asarray(args.seed_vector,dtype=float)
        if args.seed_json:
            saved=json.loads(Path(args.seed_json).read_text())
            if saved['metadata']['bank_identity']!=op.bank.identity or saved['metadata']['size']!=op.size:
                raise ValueError('saved seed identity/shape mismatch')
            pair=saved['pairs'][args.seed_index]
            initial=np.asarray(pair['vector_real'])+np.asarray(pair['vector_imag'])
        mu,X,est,receipt=E.krylov_schur(propagator,op.size,initial,k=min(args.k,op.size),
            m=min(args.ncv,op.size),tol=args.krylov_tol,max_seconds=args.krylov_seconds)
        restarts.append(dict(seed=seed,**receipt))
        for j in range(len(mu)):
            x=E.phase_fix(X[:,j]);lx=op.matvec(x)
            lam=complex(np.vdot(x,lx)/np.vdot(x,x));absolute=float(np.linalg.norm(lx-lam*x)/np.linalg.norm(x))
            relative=absolute/abs(lam) if abs(lam)>rho*1e-13 else None
            accepted=relative<=args.residual_tol if relative is not None else absolute<=args.residual_tol*rho*1e-8
            pair=dict(lam=[lam.real,lam.imag],resid=relative,absolute_residual=absolute,accepted=bool(accepted),seed=seed,
                propagator_eigenvalue=[float(mu[j].real),float(mu[j].imag)],propagator_ritz_estimate=float(est[j]),
                energy_rate=op.energy(x),localization=op.localization(x),vector_real=x.real.tolist(),vector_imag=x.imag.tolist())
            duplicate=next((i for i,p in enumerate(records) if abs(complex(*p['lam'])-lam)<1e-7*max(abs(lam),1)
                and abs(np.vdot(vectors[i],x))>1-1e-5),None)
            if duplicate is None:records.append(pair);vectors.append(x)
            elif absolute<records[duplicate]['absolute_residual']:records[duplicate]=pair;vectors[duplicate]=x
    records.sort(key=lambda p:p['lam'][0],reverse=True)
    result=dict(rho_estimate=rho,rk4_dt=dt,pairs=records[:args.k],krylov_runs=restarts,
        search_complete=False,ordering='largest propagator modulus; original-L Rayleigh quotient and residual',
        rightmost_verified=E.rightmost(dict(pairs=records)),
        interpretation='negative candidates do not prove absence of positive modes')
    return result


def abscissa(op,args):
    s=sla.LinearOperator((op.size,)*2,matvec=op.normalized_symmetric,dtype=np.float64)
    try:
        values,vectors=sla.eigsh(s,k=1,which='LA',ncv=min(args.ncv,op.size),tol=args.lanczos_tol,
            maxiter=args.maxiter,v0=np.random.default_rng(17).normal(size=op.size))
        y=vectors[:,0];lam=float(values[0]);residual=float(np.linalg.norm(op.normalized_symmetric(y)-lam*y))
        return dict(value=lam,residual=residual,converged=True,vector_y=y.tolist(),energy_rate=op.energy(y/op.sqrtH),
            localization=op.localization(y/op.sqrtH))
    except sla.ArpackNoConvergence as e:
        return dict(value=None,residual=None,converged=False,partial_values=e.eigenvalues.tolist())


def verification(op,args):
    rng=np.random.default_rng(260109);u=rng.normal(size=op.size);u/=np.linalg.norm(u)
    v=rng.normal(size=op.size);v/=np.linalg.norm(v)
    start=time.perf_counter();lu=op.matvec(u);first=time.perf_counter()-start
    start=time.perf_counter();ltv=op.rmatvec(v);transpose_first=time.perf_counter()-start
    lhs=float(np.dot(lu,v));rhs=float(np.dot(u,ltv))
    adjoint=abs(lhs-rhs)/max(np.linalg.norm(lu)*np.linalg.norm(v),np.linalg.norm(u)*np.linalg.norm(ltv),1e-300)
    energy_lhs=float(np.dot(u,op.H*lu));energy_rhs=float(np.dot(u,op.symmetric(u)))
    energy_error=abs(energy_lhs-energy_rhs)/max(np.linalg.norm(u)*np.linalg.norm(op.H*lu),abs(energy_rhs),1e-300)
    fd=[]
    for h in (1e-3,3e-4,1e-4,3e-5,1e-5):
        a=np.asarray(op.numerical(op.q0+h*u,op.data));b=np.asarray(op.numerical(op.q0-h*u,op.data))
        approximation=(a-b)/(2*h)
        fd.append(dict(h=h,relative_error=float(np.linalg.norm(approximation-lu)/np.linalg.norm(lu))))
    matrix=np.column_stack([op.matvec(np.eye(op.size)[:,i]) for i in range(op.size)])
    ev,_=np.linalg.eig(matrix);rho=float(max(abs(ev)))
    dtol=args.ncv;args.ncv=op.size  # bounded complete Krylov space, not a dense matvec
    spec=spectrum(op,args,rho=rho);abc=abscissa(op,args);args.ncv=dtol
    dense_s=.5*(op.sqrtH[:,None]*matrix/op.sqrtH[None,:]+op.sqrtH[None,:]*matrix.T/op.sqrtH[:,None])
    exact_abc=float(np.linalg.eigvalsh(dense_s)[-1])
    expected=np.sort(ev.real)[-min(args.k,op.size):][::-1]
    # Match each candidate to an actual dense eigenvalue. Also compare the
    # rightmost value; repeated eigenvalues need not have identical vectors.
    candidate_errors=[float(min(abs(complex(*p['lam'])-ev))/max(rho,1)) for p in spec['pairs']]
    rightmost_error=abs(max(p['lam'][0] for p in spec['pairs'])-max(ev.real))/max(rho,1)
    actual=np.array(sorted((p['lam'][0] for p in spec['pairs']),reverse=True))
    top_six_error=float(max(abs(actual-expected))/max(rho,1)) if len(actual)==len(expected) else None
    abc_error=abs(abc['value']-exact_abc)/max(abs(exact_abc),rho,1) if abc['converged'] else None
    gates=dict(jvp_central_fd=min(x['relative_error'] for x in fd)<2e-7,
        adjoint=adjoint<5e-13,energy_identity=energy_error<5e-13,
        krylov_dense=rightmost_error<2e-7 and top_six_error is not None and top_six_error<2e-7
            and max(candidate_errors)<2e-7 and all(p['accepted'] for p in spec['pairs']),
        abscissa_dense=abc_error is not None and abc_error<2e-9)
    warm=[]
    for i in range(5):
        tick=time.perf_counter();op.matvec(u);warm.append(time.perf_counter()-tick)
    return dict(passed=all(gates.values()),gates=gates,operator=args.operator,bc=args.bc,
        scope='bounded artificial closure: tooling only',metadata=op.metadata(),
        finite_differences=fd,adjoint_scaled_error=adjoint,adjoint_absolute_error=abs(lhs-rhs),
        energy_identity_scaled_error=energy_error,krylov_dense_scaled_errors=candidate_errors,
        krylov_rightmost_scaled_error=float(rightmost_error),dense_top_real_parts=expected.tolist(),
        krylov_top_six_real_scaled_error=top_six_error,
        lanczos_dense_scaled_error=abc_error,dense_abscissa=exact_abc,abscissa=abc,spectrum=spec,
        energy_probes=op.energy_probes(args.seeds),timing=dict(first_jvp_seconds=first,first_vjp_seconds=transpose_first,
            warm_matvec_median_seconds=float(np.median(warm))))

def verify_restarts(args):
    """Exercise the imported KS restart path on a known separated spectrum."""
    diagonal=np.linspace(-40.,5.,64);dt=.3/max(abs(diagonal));z=dt*diagonal
    polynomial=1+z+z*z/2+z**3/6+z**4/24
    runs=[]
    for seed in args.seeds:
        mu,X,est,receipt=E.krylov_schur(lambda v:polynomial*v,len(diagonal),
            np.random.default_rng(seed).normal(size=len(diagonal)),k=6,m=16,tol=1e-10,max_seconds=5.)
        values=[];errors=[]
        for v in X.T:
            lam=np.vdot(v,diagonal*v)/np.vdot(v,v)
            values.append(float(lam.real))
            errors.append(float(np.linalg.norm(diagonal*v-lam*v)/max(abs(lam)*np.linalg.norm(v),1e-300)))
        mismatch=float(max(abs(np.sort(values)-diagonal[-6:])))
        runs.append(dict(seed=seed,**receipt,original_residuals=errors,dense_top_six_max_error=mismatch,
            passed=receipt['restarts']>0 and max(errors)<1e-6 and mismatch<1e-6))
    return dict(passed=all(r['passed'] for r in runs),scope='synthetic algorithm control only',runs=runs)


def compare_derivatives(op,args):
    """Estimate the true nonlinear derivative by refined central differences.

    Calls to the original nonlinear RHS recompute projectors; no monkeypatch,
    alternate stencil or replacement characteristic splitter is used. Report
    FD refinement uncertainty separately from the frozen-projector mismatch.
    This quantifies selected directions, not the norm of the omitted operator.
    """
    probes=[]
    for seed,kind,u in op.probe_vectors(args.seeds):
        u=u/max(np.max(abs(u)),1e-300)
        frozen=op.matvec(u);estimates=[];steps=(1e-3,3e-4,1e-4,3e-5)
        for h in steps:
            plus=np.asarray(op.numerical(op.q0+h*u,op.data))
            minus=np.asarray(op.numerical(op.q0-h*u,op.data))
            estimates.append((plus-minus)/(2*h))
        # Richardson extrapolation for unequal adjacent h, choosing the pair
        # with the smallest observed change in H norm, rather than assuming
        # the smallest h is best in a stiff/cancellation-sensitive RHS.
        norm=lambda v:float(np.sqrt(np.real(np.vdot(v,op.H*v))))
        changes=[norm(estimates[i+1]-estimates[i]) for i in range(len(steps)-1)]
        i=int(np.argmin(changes));ratio=(steps[i]/steps[i+1])**2
        true=(ratio*estimates[i+1]-estimates[i])/(ratio-1)
        denominator=max(norm(true),1e-300)
        probes.append(dict(seed=seed,kind=kind,relative_difference_H=norm(frozen-true)/denominator,
            relative_difference_euclidean=float(np.linalg.norm(frozen-true)/max(np.linalg.norm(true),1e-300)),
            relative_fd_refinement_uncertainty_H=changes[i]/denominator,
            selected_steps=[steps[i],steps[i+1]],steps=list(steps),
            differences_by_step_H=[norm(frozen-d)/max(norm(d),1e-300) for d in estimates],
            relative_adjacent_fd_changes_H=[changes[j]/max(norm(estimates[j+1]),1e-300) for j in range(len(changes))]))
    return dict(metadata=op.metadata(),probes=probes,
        frozen='Q09 live-matrix/frozen-projector JVP',
        true_derivative_estimate='central FD of original nonlinear RHS, projectors recomputed, Richardson extrapolation',
        harmlessness_gate=None,limitation='directional evidence only; report refinement uncertainty; no universal operator bound')

