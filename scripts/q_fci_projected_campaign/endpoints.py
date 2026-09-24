"""Equivalent cached endpoint preparation for the projected-FCI campaign.

The frozen OwnerMoments fit, candidate scoring and fallbacks are inherited
unchanged. Each worker owns one model; the cache lasts for one endpoint call.
"""
import math

import numpy as np

from scripts.q_fci_return_campaign.endpoints import OwnerMoments


class CachedOwnerMoments(OwnerMoments):
    @classmethod
    def from_model(cls, model):
        """Reuse the already prepared immutable geometry/moments without rebuilding."""
        result = cls.__new__(cls)
        result.__dict__ = model.__dict__.copy()
        result._endpoint_cache = None
        result._in_endpoint = False
        return result

    def endpoint_pair(self, point, *, include_control=True):
        if getattr(self, '_in_endpoint', False):
            raise RuntimeError('endpoint model is already in use; use one model per worker')
        self._in_endpoint = True
        self._endpoint_cache = None
        try:
            return super().endpoint_pair(point, include_control=include_control)
        finally:
            self._endpoint_cache = None
            self._in_endpoint = False

    def geometry(self, point):
        point = np.asarray(point, dtype=float)
        k, plane_delta = self.plane(point[2])
        xy = np.array((point[0] * np.cos(point[1]), point[0] * np.sin(point[1])))
        du = 1.0 / self.N
        scale = np.array((du, max(point[0] * 2 * np.pi / self.N, du)))
        eligible, tree = self.by_plane[k]
        _d, local = tree.query(xy, k=min(96, len(eligible)))
        order = eligible[np.atleast_1d(local)]
        i = min(max(int(np.searchsorted(self.grid.x.faces, point[0], side='right') - 1), 0), self.N - 1)
        j = int((np.searchsorted(self.grid.y.faces, point[1] % (2 * np.pi), side='right') - 1) % self.N)
        containing = int(self.labels[i, j, k])
        # by_plane[k] is exactly the set selected by self.eta == k.
        if self.eta[containing] == k:
            order = np.r_[containing, order[order != containing]]
        if getattr(self, '_in_endpoint', False):
            ids = np.sort(order)
            self._endpoint_cache = {
                'owners': ids, 'local': {}, 'indices': {},
                'absolute': {pq: a[ids] for pq, a in self.absolute.items()},
            }
        return point, k, plane_delta, xy, scale, order, containing

    def local(self, owners, xy, scale, exps):
        cache = getattr(self, '_endpoint_cache', None)
        if cache is None:
            return super().local(owners, xy, scale, exps)
        key = tuple(exps)
        if key not in cache['local']:
            x0, y0 = xy
            sx, sy = scale
            out = []
            for p, q in exps:
                value = np.zeros(len(cache['owners']))
                # Preserve the frozen binomial arithmetic and summation order.
                for i in range(p + 1):
                    for j in range(q + 1):
                        value += (math.comb(p, i) * math.comb(q, j) * (-x0) ** (p - i) *
                                  (-y0) ** (q - j) * cache['absolute'][i, j])
                out.append(value / (sx**p * sy**q))
            cache['local'][key] = np.column_stack(out)
        donor_key = tuple(owners)
        if donor_key not in cache['indices']:
            idx = np.searchsorted(cache['owners'], owners)
            if not np.array_equal(cache['owners'][idx], owners):
                raise RuntimeError('endpoint support is outside the prepared owner pool')
            cache['indices'][donor_key] = idx
        return cache['local'][key][cache['indices'][donor_key]]

    def directional(self, order, xy, scale, count):
        delta = (self.centroid[order] - xy) / scale
        angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * np.pi)
        chosen = [0]
        used = np.zeros(len(order), bool)
        used[0] = True
        for sector in range(8):
            hit = np.flatnonzero((angle >= sector * np.pi / 4) & (angle < (sector + 1) * np.pi / 4))
            if hit.size and not used[int(hit[0])]:
                chosen.append(int(hit[0]))
                used[int(hit[0])] = True
        chosen.extend(np.flatnonzero(~used).tolist())
        return order[np.asarray(chosen[:count], dtype=int)]
