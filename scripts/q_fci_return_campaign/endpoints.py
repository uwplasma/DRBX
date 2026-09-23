"""Frozen audited owner-midpoint cubic endpoint policy; research preparation only.
Extracted from parallel_q03_selector_corrected_global_20260920/run_comparison.py.
"""
import math
import numpy as np
EXPS2 = tuple((p,q) for degree in range(3) for p in range(degree+1) for q in (degree-p,))
EXPS3 = tuple((p,q) for degree in range(4) for p in range(degree+1) for q in (degree-p,))
EXPS4 = tuple((p,q) for degree in range(5) for p in range(degree+1) for q in (degree-p,))

class OwnerMoments:
    """Actual raw-volume owner moments through degree four in regular x/y."""

    def __init__(self, context):
        artifact = context["artifact"]; shape = tuple(artifact.geometry.shape)
        self.context = context; self.N = context["N"]
        self.labels = np.asarray(context["topology"]["compact_raw_owner"], dtype=int).reshape(shape)
        self.weight = np.asarray(artifact.polar_angular_geometry.raw_volume, dtype=float).reshape(shape)
        grid = artifact.geometry.grid
        u, theta, _eta = np.meshgrid(grid.x.centers, grid.y.centers, grid.z.centers, indexing="ij")
        x = u * np.cos(theta); y = u * np.sin(theta); n = len(context["volume"])
        self.absolute = {}
        for p, q in EXPS4:
            self.absolute[p, q] = np.bincount(self.labels.ravel(),
                weights=(self.weight * x**p * y**q).ravel(), minlength=n) / np.asarray(context["volume"])
        self.centroid = np.column_stack((self.absolute[1, 0], self.absolute[0, 1]))
        self.eta = np.asarray(context["topology"]["owner_eta"], dtype=int)
        self.grid = grid; self.by_plane = {}
        from scipy.spatial import cKDTree
        for k in range(self.N):
            owners = np.flatnonzero(self.eta == k); self.by_plane[k] = (owners, cKDTree(self.centroid[owners]))

    def plane(self, eta):
        centers = np.asarray(self.grid.z.centers)
        delta = np.abs((centers - eta + np.pi) % (2*np.pi) - np.pi); k = int(np.argmin(delta))
        if delta[k] > 2e-10:
            raise RuntimeError(f"endpoint eta is off stored center planes by {delta[k]:.3e}")
        return k, float(delta[k])

    def local(self, owners, xy, scale, exps):
        owners = np.asarray(owners, dtype=int); x0, y0 = xy; sx, sy = scale
        out = []
        for p, q in exps:
            value = np.zeros(len(owners))
            for i in range(p + 1):
                for j in range(q + 1):
                    value += (math.comb(p, i) * math.comb(q, j) * (-x0) ** (p - i) *
                              (-y0) ** (q - j) * self.absolute[i, j][owners])
            out.append(value / (sx**p * sy**q))
        return np.column_stack(out)

    def geometry(self, point):
        point = np.asarray(point, dtype=float); k, plane_delta = self.plane(point[2])
        xy = np.array((point[0] * np.cos(point[1]), point[0] * np.sin(point[1])))
        du = 1.0 / self.N; scale = np.array((du, max(point[0] * 2 * np.pi / self.N, du)))
        eligible, tree = self.by_plane[k]; _d, local = tree.query(xy, k=min(96, len(eligible)))
        order = eligible[np.atleast_1d(local)]
        i = min(max(int(np.searchsorted(self.grid.x.faces, point[0], side="right") - 1), 0), self.N - 1)
        j = int((np.searchsorted(self.grid.y.faces, point[1] % (2 * np.pi), side="right") - 1) % self.N)
        containing = int(self.labels[i, j, k])
        if containing in eligible:
            order = np.r_[containing, order[order != containing]]
        return point, k, plane_delta, xy, scale, order, containing

    def directional(self, order, xy, scale, count):
        delta = (self.centroid[order] - xy) / scale
        angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * np.pi); chosen = [0]
        for sector in range(8):
            hit = np.flatnonzero((angle >= sector * np.pi / 4) & (angle < (sector + 1) * np.pi / 4))
            if hit.size and int(hit[0]) not in chosen: chosen.append(int(hit[0]))
        chosen.extend(i for i in range(len(order)) if i not in chosen)
        return order[np.asarray(chosen[:count], dtype=int)]

    def fit(self, donors, xy, scale, degree, label):
        exps = EXPS3 if degree == 3 else EXPS2
        P = self.local(donors, xy, scale, exps)
        distance = np.linalg.norm((self.centroid[donors] - xy) / scale, axis=1)
        weight = 1.0 / (1.0 + distance * distance) ** 2
        B = P.T * weight[None, :]; U, singular, Vt = np.linalg.svd(B, full_matrices=False)
        tol = np.finfo(float).eps * max(B.shape) * singular[0]; rank = int(np.count_nonzero(singular > tol))
        target = np.zeros(len(exps)); target[0] = 1.0
        coefficient = weight * (Vt[:rank].T @ ((U[:, :rank].T @ target) / singular[:rank]))
        residual = float(np.linalg.norm(P.T @ coefficient - target))
        R4 = self.local(donors, xy, scale, ((4, 0), (2, 2), (0, 4)))
        radial_fourth = R4[:, 0] + 2 * R4[:, 1] + R4[:, 2]
        l1 = float(np.sum(np.abs(coefficient))); extent = float(np.max(distance))
        remainder = float(np.sum(np.abs(coefficient) * np.maximum(radial_fourth, 0.0)))
        roundoff = float(np.finfo(float).eps * (singular[0] / singular[rank - 1]) * l1) if rank else float("inf")
        return {"label": label, "degree": degree, "donors": np.asarray(donors), "coefficient": coefficient,
                "rank": rank, "condition": float(singular[0] / singular[rank - 1]) if rank else float("inf"),
                "residual": residual, "coefficient_l1": l1, "max_scaled_distance": extent,
                "roundoff_indicator": roundoff,
                "degree4_owner_remainder_proxy": remainder,
                "score": float((remainder + .05 * l1) * (1 + .1 * extent)),
                "feasible": bool(rank == len(exps) and residual <= 1e-11)}

    def endpoint_pair(self, point, *, include_control=True):
        point, k, plane_delta, xy, scale, order, containing = self.geometry(point)
        supports = [("nearest24", order[:24]), ("directional24", self.directional(order, xy, scale, 24)),
                    ("nearest48", order[:48])]
        trials = [self.fit(d, xy, scale, 3, label) for label, d in supports]
        usable = [x for x in trials if x["feasible"] and x["roundoff_indicator"] <= 1e-8]
        if not usable:
            trials.extend((self.fit(self.directional(order, xy, scale, 48), xy, scale, 3, "directional48_fallback"),
                           self.fit(order[:96], xy, scale, 3, "nearest96_fallback")))
            usable = [x for x in trials if x["feasible"] and x["roundoff_indicator"] <= 1e-8]
        if not usable: raise RuntimeError(f"no numerically usable cubic endpoint fit at {point.tolist()}")
        chosen = min(usable, key=lambda x: (x["score"], x["max_scaled_distance"], x["coefficient_l1"]))
        control = self.fit(chosen["donors"], xy, scale, 2, "quadratic_on_G3_support") if include_control else None
        if control is not None and not control["feasible"]: raise RuntimeError("matched quadratic support failed")
        public = lambda x: {key: value for key, value in x.items() if key not in ("donors", "coefficient")}
        meta = {"point": point.tolist(), "plane": k, "plane_delta": plane_delta, "scale": scale.tolist(),
                "containing_owner": containing, "containing_owner_in_support": bool(containing in chosen["donors"]),
                "chosen": public(chosen), "quadratic_control": public(control) if control is not None else None,
                "trials": [public(x) for x in trials]}
        return (chosen["donors"], chosen["coefficient"], control["coefficient"] if control is not None else None, meta)
