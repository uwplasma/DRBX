"""Independent time-dependent smooth MMS with fixed spatial reference geometry."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from scripts.q08_extraction_global import common as catalogue
from drbx.native.q_parallel import QBoundaryData

FIELDS = ('n', 'Te', 'Ti', 'Vi', 'Ve', 'omega')
BASE = np.r_[catalogue.BASE, .2]
COEFFICIENTS = catalogue.COEFF.copy()


def amplitude(t):
    """a(0)=1, 0.9 <= a(t) <= 1.1; contract interval [0,1]."""
    return 1 + .1*jnp.sin(t)


def amplitude_dt(t):
    return .1*jnp.cos(t)


def smooth_fields(points, t):
    """JAX smooth catalogue values and coordinate gradients (...,6[,3])."""
    p = jnp.asarray(points)
    u, th, eta = jnp.moveaxis(p, -1, 0)
    x, y = u*jnp.cos(th), u*jnp.sin(th)
    sx, sy = x*jnp.cos(eta), y*jnp.sin(2*eta)
    gx = jnp.stack((jnp.cos(th)*jnp.cos(eta), -y*jnp.cos(eta), -x*jnp.sin(eta)), -1)
    gy = jnp.stack((jnp.sin(th)*jnp.sin(2*eta), x*jnp.sin(2*eta), 2*y*jnp.cos(2*eta)), -1)
    f = jnp.stack((sx, sy, sx, sy, sx), -1)
    g = jnp.stack((gx, gy, gx, gy, gx), -2)
    v = jnp.asarray(catalogue.BASE) + amplitude(t)*jnp.asarray(catalogue.AMP)*f
    g = amplitude(t)*jnp.asarray(catalogue.AMP)[:, None]*g
    omega = catalogue.WOFF + v@jnp.asarray(catalogue.WC)
    og = jnp.einsum('...fa,f->...a', g, jnp.asarray(catalogue.WC))
    phi = amplitude(t)*(.07*sx + .04*sy)
    pg = amplitude(t)*(.07*gx + .04*gy)
    return jnp.concatenate((v, omega[..., None]), -1), jnp.concatenate((g, og[..., None, :]), -2), phi, pg


@dataclass(frozen=True)
class Manufactured:
    """Owner observations from physical-volume weighted raw members, never midpoints."""
    owner_initial: object
    phi_initial: object
    boundary_initial: tuple
    boundary_constant: tuple

    @classmethod
    def from_members(cls, bank, points, raw_to_owner, raw_volume, owner_volume):
        points, ids, w, vol = map(np.asarray, (points, raw_to_owner, raw_volume, owner_volume))
        bank.validate()
        if len(bank.raw) != bank.metadata['n']**3 or not np.array_equal(bank.owners, np.arange(bank.metadata['n_owner'])):
            raise ValueError('member observations require full-domain owner/raw closure')
        if not np.array_equal(ids, bank.raw_to_owner) or not np.array_equal(points, bank.diagnostics['slot_points'][:, 2]):
            raise ValueError('member observation bank coordinate/owner mismatch')
        no = int(bank.metadata['n_owner'])
        if points.shape != (len(ids), 3) or w.shape != ids.shape or vol.shape != (no,):
            raise ValueError('member observation shape mismatch')
        if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= no):
            raise ValueError('member owner identity mismatch')
        if not all(np.isfinite(a).all() for a in (points, w, vol)) or np.any(w <= 0) or np.any(vol <= 0):
            raise ValueError('invalid observation volume/points')
        if not np.allclose(w/vol[ids], bank.raw_weight, rtol=2e-13, atol=0):
            raise ValueError('member observation bank weights mismatch')
        totals = np.bincount(ids, weights=w, minlength=no)
        if not np.allclose(totals, vol, rtol=2e-13, atol=0):
            raise ValueError('incomplete owner member volume coverage')
        if np.any(points[:, 0] < 0) or np.any(points[:, 0] > 1):
            raise ValueError('smooth positivity contract requires 0 <= u <= 1')
        observed = np.zeros((no, 6)); phi = np.zeros(no)
        for start in range(0, len(points), 4096):
            sl = slice(start, start+4096)
            v, _, p, _ = smooth_fields(points[sl], 0.)
            np.add.at(observed, ids[sl], w[sl, None]*np.asarray(v))
            np.add.at(phi, ids[sl], w[sl]*np.asarray(p))
        observed /= vol[:, None]; phi /= vol
        # Preserve even nonwall affine padding conventions from Q08 verbatim.
        bc = tuple(jax.tree.map(jnp.asarray, b) for b in catalogue.boundaries(bank, 1))
        zero = tuple(jax.tree.map(jnp.asarray, b) for b in catalogue.boundaries(bank, 0))
        return cls(jnp.asarray(observed.T), jnp.asarray(phi), bc, zero)

    def state(self, t):
        return jnp.asarray(BASE)[:, None] + amplitude(t)*(self.owner_initial-jnp.asarray(BASE)[:, None])

    def derivative(self, t):
        return amplitude_dt(t)*(self.owner_initial-jnp.asarray(BASE)[:, None])

    def phi(self, t):
        return amplitude(t)*self.phi_initial

    def boundaries(self, t):
        # Exact temporal formula on every original bank query, no interpolation.
        return jax.tree.map(lambda v, b: b+amplitude(t)*(v-b), self.boundary_initial, self.boundary_constant)


@dataclass(frozen=True)
class ContinuumReference:
    """Independent fourth-order coordinate-flux reference; no numerical Q action."""
    values: object
    gradients: object
    phi_gradient: object
    diffusion: object
    kappa: object
    bmag: object
    owner_raw: object
    owner_weight: object
    diagnostics: dict

    @classmethod
    def from_saved_continuum(cls, bank, geometry, geom, action, diagnostics):
        """Reuse checked Q08 R diffusion; independently recover raw material data.

        Only smooth R columns 12:24 are consumed. The superseded standalone Ti
        diagnostic column 29 is not used. No numerical Q row supplies forcing.
        """
        action = np.asarray(action)
        if len(bank.raw) > 128 or action.shape != (catalogue.NF, len(bank.owners), 31):
            raise ValueError('saved continuum shape/chunk size mismatch')
        if not np.isfinite(action).all():
            raise ValueError('nonfinite saved continuum')
        p = bank.diagnostics['slot_points'][:, 2]
        J, b, B = map(np.asarray, geom(p))
        np.testing.assert_allclose(b[:, 2], geometry['b_eta'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(B, geometry['bmag'], atol=1e-10, rtol=0)
        h = 1e-4
        pp = np.broadcast_to(p[:, None, None, :], (len(p), 3, 4, 3)).copy()
        for d in range(3):
            pp[:, d, :, d] += np.array([-2, -1, 1, 2])*h
        jj, bb, _ = geom(pp.reshape(-1, 3))
        jb = (np.asarray(jj)[:, None]*np.asarray(bb)).reshape(len(p), 3, 4, 3)
        weights = np.array([1, -8, 8, -1])/(12*h)
        kap = sum(np.einsum('rq,q->r', jb[:, d, :, d], weights) for d in range(3))/J
        values, gradients, _, pg = map(np.asarray, smooth_fields(p, 0.))
        result = cls(values, np.einsum('ra,rfa->rf', b, gradients),
            np.einsum('ra,ra->r', b, pg), action[1, :, 12:18].copy(), kap, B,
            np.asarray(bank.owner_raw), np.asarray(bank.owner_weight), dict(diagnostics))
        actual = np.asarray(result.rhs(0., 'complete')).T
        # Same cross-platform action replay budget as accepted Q08; this is
        # an input replay gate, not a scientific solution-error tolerance.
        np.testing.assert_allclose(actual, action[1, :, 18:24], atol=1e-8, rtol=1e-11)
        result.diagnostics.update(saved_complete_replay_max=float(np.max(abs(actual-action[1, :, 18:24]))),
            saved_diffusion_reused=True, independent_raw_kappa_step=h)
        return result

    @classmethod
    def prepare(cls, bank, geometry, geom):
        # Reuse the authoritative continuum implementation for the linear
        # diffusion coefficient and reference-step sensitivity at t=0.
        if len(bank.raw) > 128:
            raise ValueError('reference prepare requires <=128 raw rows; use prepare_chunks for full-domain banks')
        from scripts.q08_rhs_mms_global.science import continuum
        action, diagnostics = continuum(bank, geometry, geom)
        p = bank.diagnostics['slot_points'][:, 2]
        J, b, B = map(np.asarray, geom(p))
        v, g = catalogue.six_fields(p)
        pg = catalogue.phi_fields(p)[1][:, 1]
        h = 1e-4
        pp = np.broadcast_to(p[:, None, None, :], (len(p), 3, 4, 3)).copy()
        for d in range(3):
            pp[:, d, :, d] += np.array([-2, -1, 1, 2])*h
        jj, bb, _ = geom(pp.reshape(-1, 3))
        jb = (np.asarray(jj)[:, None]*np.asarray(bb)).reshape(len(p), 3, 4, 3)
        weights = np.array([1, -8, 8, -1])/(12*h)
        kap = sum(np.einsum('rq,q->r', jb[:, d, :, d], weights) for d in range(3))/J
        result = cls(*map(jnp.asarray, (v[:, 1], np.einsum('ra,rfa->rf', b, g[:, 1]),
            np.einsum('ra,ra->r', b, pg), action[1, :, 12:18], kap, B,
            bank.owner_raw, bank.owner_weight)), diagnostics)
        np.testing.assert_allclose(result.rhs(0., 'complete').T, action[1, :, 18:24], rtol=1e-12, atol=1e-10)
        return result

    @classmethod
    def prepare_chunks(cls, bank, chunks, geom):
        """Stream bounded whole-owner reference views in exact merged order."""
        refs = []; raws = []; owners = []
        for view, geometry in chunks:
            refs.append(cls.prepare(view, geometry, geom))
            raws.extend(np.asarray(view.raw).tolist()); owners.extend(np.asarray(view.owners).tolist())
        if not refs or not np.array_equal(raws, bank.raw) or not np.array_equal(owners, bank.owners):
            raise ValueError('reference chunk full-domain order/coverage mismatch')
        names = ('values', 'gradients', 'phi_gradient', 'diffusion', 'kappa', 'bmag')
        values = {name: jnp.concatenate([getattr(r, name) for r in refs], axis=0) for name in names}
        return cls(**values, owner_raw=jnp.asarray(bank.owner_raw), owner_weight=jnp.asarray(bank.owner_weight),
            diagnostics={'chunks': [r.diagnostics for r in refs]})

    def project(self, raw):
        return jnp.sum(jnp.take(raw, self.owner_raw, axis=0)*self.owner_weight[..., None], axis=1).T

    def rhs(self, t, mode):
        if mode not in ('diffusion', 'complete'):
            raise ValueError('mode must be diffusion or complete')
        diffusion = amplitude(t)*self.diffusion.T
        if mode == 'diffusion':
            return diffusion
        V = jnp.asarray(BASE)+amplitude(t)*(self.values-jnp.asarray(BASE))
        G = amplitude(t)*self.gradients
        n, te, ti, vi, ve, om = jnp.moveaxis(V, -1, 0)
        gn, gt, gi, gvi, gve, go = jnp.moveaxis(G, -1, 0)
        kap = self.kappa
        dj = (vi-ve)*gn+n*(gvi-gve)+kap*n*(vi-ve)
        mu, tau = catalogue.MU, catalogue.TAU
        material = jnp.stack((-(ve*gn+n*gve+kap*n*ve),
            -ve*gt+2*te/(3*n)*(.71*dj-n*(gve+kap*ve)),
            -vi*gi+2*ti/(3*n)*(dj-n*(gvi+kap*vi)),
            -vi*gvi-((te+tau*ti)*gn+n*(gt+tau*gi))/n,
            -ve*gve-mu*(te*gn+n*gt)/n-.71*mu*gt+mu*amplitude(t)*self.phi_gradient,
            -vi*go+self.bmag**2/n*dj), -1)
        return self.project(material)+diffusion

    def source(self, manufactured, t, mode):
        return manufactured.derivative(t)-self.rhs(t, mode)
