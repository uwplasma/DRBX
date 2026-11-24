"""Frozen polygon arithmetic, unchanged expression order."""
import numpy as np
TAU=2*np.pi
def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def clip(poly, values):
    if not len(poly):
        return poly
    f = poly @ values
    out = []
    for i in range(len(poly)):
        a = poly[i - 1]
        b = poly[i]
        fa = f[i - 1]
        fb = f[i]
        if (fa >= 0) != (fb >= 0):
            out.append((fb * a - fa * b) / (fb - fa))
        if fb >= 0:
            out.append(b)
    return np.array(out).reshape(-1, 3)

def inpoly(poly, land, quad):
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        poly = clip(poly, cross(b - a, land - a))
        if len(poly) < 3:
            return np.empty((0, 3))
    return poly

def area_centroid(poly, xy):
    if len(poly) < 3:
        return (0, None)
    points = poly @ xy
    w = cross(points[1:-1] - points[0], points[2:] - points[0]) / 2
    a = w.sum()
    if a <= 1e-18:
        return (0, None)
    b = ((poly[0] + poly[1:-1] + poly[2:]) * w[:, None]).sum(0) / (3 * a)
    return (a, b)

def tiles(cells, sub=4):
    out = []
    owner = []
    angles = np.unique(np.round(np.concatenate([np.linspace(t0, t1, sub + 1) % TAU for r0, r1, t0, t1 in cells]), 13))
    for j, (r0, r1, t0, t1) in enumerate(cells):
        local = np.r_[angles - TAU, angles, angles + TAU]
        ts = np.r_[t0, local[(local > t0 + 1e-12) & (local < t1 - 1e-12)], t1]
        ts.sort()
        for a, b in zip(ts[:-1], ts[1:]):
            q = np.array([[r0 * np.cos(a), r0 * np.sin(a)], [r1 * np.cos(a), r1 * np.sin(a)], [r1 * np.cos(b), r1 * np.sin(b)], [r0 * np.cos(b), r0 * np.sin(b)]])
            if r0 == 0:
                q = q[:3]
            out.append(q)
            owner.append(j)
    return (out, np.array(owner), np.array([q.min(0) for q in out]), np.array([q.max(0) for q in out]))

def disk(nr, nt):
    xy = [[0.0, 0.0]]
    polar = [[1e-10, 0.0]]
    for i in range(1, nr + 1):
        for j in range(nt):
            r = i / nr
            th = j * TAU / nt
            xy.append([r * np.cos(th), r * np.sin(th)])
            polar.append([min(r, 1 - 1e-10), round(th, 13)])
    tri = []
    for j in range(nt):
        tri.append([0, 1 + j, 1 + (j + 1) % nt])
    for i in range(1, nr):
        for j in range(nt):
            a = 1 + (i - 1) * nt + j
            b = 1 + i * nt + j
            c = 1 + i * nt + (j + 1) % nt
            d = 1 + (i - 1) * nt + (j + 1) % nt
            tri.extend([[a, b, c], [a, c, d]])
    return (np.array(xy), np.array(polar), np.array(tri))
