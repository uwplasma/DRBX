"""Immutable-cell band cache with original predicates, order and ties, and an optional check."""
import numpy as np
TAU=2*np.pi

class CachedLocator:
    """Same predicates, tolerances, band order and tie rule as pkg.locate_fast.locate; only the per-call
    O(cells) work is moved to a one-time per-cells precompute, and the reference spot check is evaluated
    only on cells that can satisfy its radial predicate (others never change its result)."""
    def __init__(self, c, reference='vector'):
        c = np.asarray(c, float); self.c = c; self.reference = reference
        keys = np.round(c[:, :2], 12); _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True); inv = np.asarray(inv).ravel()
        self.BIG = len(c) + 1; self.bands = []
        for b in sorted(range(len(first)), key=lambda b: first[b]):
            js = np.flatnonzero(inv == b)
            st = np.mod(c[js, 2], TAU); o = np.argsort(st, kind='stable')
            self.bands.append((js, st[o], o, len(js)))
        self.rlo = np.array([c[js, 0].min() - 2e-12 for js, *_ in self.bands])
        self.rhi = np.array([c[js, 1].max() + 2e-12 for js, *_ in self.bands])
        self.r0m = c[:, 0] - 2e-12; self.r1p = c[:, 1] + 2e-12

    def reference_restricted(self, qs):
        c = self.c
        K = np.flatnonzero((self.r0m <= qs[:, 0].max()) & (self.r1p >= qs[:, 0].min()))
        if self.reference == 'loop':
            out = np.full(len(qs), -1, int)
            for j in K:
                r0, r1, t0, t1 = c[j]
                t = (qs[:, 1] - t0) % TAU
                hit = (qs[:, 0] >= r0 - 2e-12) & (qs[:, 0] <= r1 + 2e-12) & (t <= t1 - t0 + 2e-12)
                out[(out < 0) & hit] = j
            return out
        cc = c[K]
        t = (qs[:, 1, None] - cc[None, :, 2]) % TAU
        hit = (qs[:, 0, None] >= cc[None, :, 0] - 2e-12) & (qs[:, 0, None] <= cc[None, :, 1] + 2e-12) & (t <= cc[None, :, 3] - cc[None, :, 2] + 2e-12)
        any_ = hit.any(1); out = np.full(len(qs), -1, int); out[any_] = K[hit[any_].argmax(1)]
        return out

    def __call__(self, q, check=None, check_all=False):
        q = np.asarray(q); c = self.c; n = len(q); out = np.full(n, -1, int); r = q[:, 0]; th = q[:, 1]
        order = np.argsort(r, kind='stable'); rs = r[order]
        LO = np.searchsorted(rs, self.rlo, 'left'); HI = np.searchsorted(rs, self.rhi, 'right')
        BIG = self.BIG
        for bi in np.flatnonzero(HI > LO):
            js, sto, o, ns = self.bands[bi]
            cand = order[LO[bi]:HI[bi]]; cand = cand[out[cand] < 0]
            if not len(cand): continue
            base = np.searchsorted(sto, np.mod(th[cand], TAU), 'right') - 1
            best = np.full(len(cand), BIG)
            for d in (-2, -1, 0, 1, 2):
                j = js[o[(base + d) % ns]]
                ok = (r[cand] >= c[j, 0] - 2e-12) & (r[cand] <= c[j, 1] + 2e-12) & (((th[cand] - c[j, 2]) % TAU) <= (c[j, 3] - c[j, 2]) + 2e-12)
                best = np.where(ok & (j < best), j, best)
            got = best < BIG; out[cand[got]] = best[got]
        assert np.all(out >= 0), (q[out < 0])
        if check is None:check = False
        if n and check:
            idx = np.arange(n) if check_all else np.unique(np.linspace(0, n - 1, min(n, 2048)).astype(int))
            assert np.array_equal(out[idx], self.reference_restricted(q[idx])), 'vectorised locate disagrees with the reference'
        return out

_LAST=None
def locator(c):
 global _LAST
 c=np.asarray(c,float)
 if c.flags.writeable:return CachedLocator(c) # Mutable callers retain fresh-input semantics.
 if _LAST is None or _LAST.c is not c:_LAST=CachedLocator(c)
 return _LAST

def locate(q,c):return locator(c)(q)
