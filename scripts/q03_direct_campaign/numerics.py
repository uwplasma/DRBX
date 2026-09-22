"""Field-independent scalar-owner moments and integrated face functionals.

The spatial tree only prunes candidates: final distances and ordering use the
same anisotropic chart as the original exhaustive prototype.
"""
import itertools
import math
import numpy as np
from scipy.spatial import cKDTree

EXPS = tuple(e for degree in range(4)
             for e in itertools.product(range(degree + 1), repeat=3)
             if sum(e) == degree)
XY = tuple((i, j) for degree in range(4) for i in range(degree + 1)
           for j in [degree - i])
POLICY = {
    "degree": 3, "owners_per_plane": 24, "plane_offsets": [-2, -1, 0, 1, 2],
    "owner_sampling": "raw-midpoint physical-volume-weighted means",
    "selection": "exact local radial/tangential distance, owner-id tie; incident owners retained",
    "weight": "(1+radial_tangential_eta_distance_squared)^(-3/2)",
    "target": "continuous geometry integrated q9 parallel face flux",
    "boundary": "prescribed continuum face flux, identical to high-order reference",
}


def delta(x, origin):
    return (np.asarray(x) - origin + np.pi) % (2 * np.pi) - np.pi


def centered_owner_moments(xyz, labels, raw_volume, volume, centers, resolution):
    """Ten centered XY moments; eta is constant inside each angular owner."""
    if np.max(np.abs(delta(xyz[:, 2], centers[labels, 2]))) > 1e-12:
        raise ValueError("owner spans eta planes; planar-owner moment contract violated")
    z = (xyz[:, :2] - centers[labels, :2]) * resolution
    return np.column_stack([
        np.bincount(labels, weights=raw_volume * z[:, 0]**i * z[:, 1]**j,
                    minlength=len(volume)) / volume for i, j in XY
    ])


class OwnerIndex:
    def __init__(self, centers, planes, resolution):
        self.centers = centers
        self.planes = planes
        self.N = resolution
        self.ids = [np.flatnonzero(planes == p) for p in range(resolution)]
        if any(len(ids) < POLICY["owners_per_plane"] for ids in self.ids):
            raise ValueError("fewer than 24 owners on an eta plane")
        self.trees = [cKDTree(centers[ids, :2]) for ids in self.ids]

    def select(self, center, plane, minus, plus, exhaustive=False):
        theta = np.arctan2(center[1], center[0])
        sr = 1 / self.N
        st = max(np.hypot(*center[:2]) * 2 * np.pi / self.N, sr)
        ct, sn = np.cos(theta), np.sin(theta)

        def distances(ids):
            xy = self.centers[ids, :2] - center[:2]
            return ((xy[:, 0] * ct + xy[:, 1] * sn) / sr)**2 + ((-xy[:, 0] * sn + xy[:, 1] * ct) / st)**2

        donors, candidate_counts = [], []
        for offset in POLICY["plane_offsets"]:
            p = (int(plane) + offset) % self.N
            ids = self.ids[p]
            if exhaustive:
                candidates = ids
            else:
                # Any 24 candidates give an upper bound on the true 24th
                # anisotropic distance. A Euclidean ball of r*max(sr,st)
                # contains the entire corresponding anisotropic ellipse.
                _, first = self.trees[p].query(center[:2], k=24)
                radius = math.sqrt(float(np.max(distances(ids[first])))) * max(sr, st)
                # Outward roundoff guard affects pruning, never final ordering.
                radius += 64 * np.finfo(float).eps * max(1., np.linalg.norm(center[:2]), radius)
                candidates = ids[self.trees[p].query_ball_point(center[:2], radius)]
            dd = distances(candidates)
            ranked = candidates[np.lexsort((candidates, dd))]
            chosen = list(dict.fromkeys(int(j) for j in (minus, plus)
                                        if j >= 0 and self.planes[j] == p))
            for j in ranked:
                if len(chosen) == 24:
                    break
                if j not in chosen:
                    chosen.append(int(j))
            if len(chosen) != 24:
                raise RuntimeError("candidate bound omitted required donors")
            donors.extend(chosen)
            candidate_counts.append(len(candidates))
        donors = np.asarray(donors, dtype=np.int64)
        distance2 = distances(donors) + (delta(self.centers[donors, 2], center[2]) / (2*np.pi/self.N))**2
        return donors, distance2, candidate_counts


def owner_matrix(donors, centers, moments, face_center, face_scale, resolution):
    """Translate centered raw-owner moments, avoiding per-face raw-cell gathers.

    Inputs may be batched: donors (faces,120), center/scale (faces,3).
    Centering before precomputation avoids subtraction of large raw moments.
    """
    shift = (centers[donors] - face_center[..., None, :]) / face_scale[..., None, :]
    shift[..., 2] = delta(centers[donors, 2], face_center[..., None, 2]) / face_scale[..., None, 2]
    factor = 1 / (resolution * face_scale)
    m = moments[donors]
    result = np.empty(donors.shape + (20,))
    for k, (a, b, c) in enumerate(EXPS):
        value = np.zeros(donors.shape)
        for i in range(a + 1):
            for j in range(b + 1):
                value += (math.comb(a, i) * math.comb(b, j)
                          * shift[..., 0]**(a-i) * shift[..., 1]**(b-j)
                          * (factor[..., 0]**i * factor[..., 1]**j)[..., None]
                          * m[..., XY.index((i, j))])
        result[..., k] = value * shift[..., 2]**c
    return result


def fit_batch(matrix, distance2, target):
    weight = (1 + distance2)**-1.5
    B = matrix.swapaxes(-1, -2) * weight[..., None, :]
    U, singular, Vt = np.linalg.svd(B, full_matrices=False)
    rank = np.sum(singular > np.finfo(float).eps * max(B.shape[-2:]) * singular[:, :1], axis=1)
    if np.any(rank != 20):
        raise RuntimeError(f"rank-deficient cubic fit: {rank.tolist()}")
    amplitude = np.einsum('fij,fi->fj', U, target) / singular
    coefficients = weight * np.einsum('fji,fj->fi', Vt, amplitude)
    residual = np.linalg.norm(np.einsum('fij,fi->fj', matrix, coefficients) - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-300)
    if not np.all(np.isfinite(coefficients)) or np.any(residual > 1e-10):
        raise RuntimeError(f"incompatible cubic target: max relative defect {residual.max()}")
    return coefficients, {
        "rank": rank, "condition": singular[:, 0] / singular[:, -1],
        "cubic_defect": residual, "constant_flux": np.sum(coefficients, axis=1),
    }
